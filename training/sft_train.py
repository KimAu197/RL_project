"""SFT training entry point (TRL SFTTrainer) driven by a YAML config.

Supports both variants:

* ``mode: direct``  -> target is ``<sql>...</sql>``
* ``mode: sketch``  -> target is ``<sketch>...</sketch><sql>...</sql>``

Run as ``python -m project.training.sft_train --config configs/sft_direct.yaml``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from datasets import Dataset

from ..data.spider_dataset import load_spider_splits, read_jsonl
from ..data.prompt_builder import PromptSpec
from ..evaluation.utils import (
    configure_logging,
    deep_merge,
    load_config,
    resolve_output_dir,
    snapshot_config,
)
from ..models.loader import LoaderConfig, load_model_and_tokenizer

LOGGER = logging.getLogger("sft_train")


def _records_to_messages(records: list[dict]) -> list[dict]:
    """Convert SpiderRecord dicts into TRL SFTTrainer-style message examples."""
    out: list[dict] = []
    for r in records:
        out.append(
            {
                "messages": [
                    {"role": "system", "content": r["prompt_system"]},
                    {"role": "user", "content": r["prompt_user"]},
                    {"role": "assistant", "content": r["target"]},
                ]
            }
        )
    return out


def _build_dataset(cfg: dict, mode: str, split: str, limit: int | None) -> Dataset:
    data_cfg = cfg["data"]
    processed = data_cfg.get("processed_dir")
    if processed:
        fname = f"spider_{mode}_{split}.jsonl"
        path = Path(processed) / fname
        if path.exists():
            LOGGER.info("loading processed JSONL: %s", path)
            records = read_jsonl(path)
            if limit is not None:
                records = records[:limit]
            return Dataset.from_list(_records_to_messages(records))

    LOGGER.info("materializing Spider records from %s split=%s mode=%s", data_cfg["spider_root"], split, mode)
    records = load_spider_splits(
        data_cfg["spider_root"],
        split=split,
        mode=mode,
        max_schema_chars=int(data_cfg.get("max_schema_chars", 4000)),
        limit=limit,
    )
    return Dataset.from_list(_records_to_messages([r.to_dict() for r in records]))


def run_sft(config_path: str, extra: dict) -> int:
    cfg = load_config(config_path)
    cfg = deep_merge(cfg, extra)

    mode = cfg["prompt"]["mode"]
    assert mode in ("direct", "sketch")

    run_cfg = cfg["run"]
    output_dir = resolve_output_dir(
        base=run_cfg.get("output_root", "runs"),
        round_name=run_cfg.get("round_name", "sft"),
        model_name=run_cfg.get("model_tag") or cfg["model"]["name_or_path"].split("/")[-1],
        dataset=cfg["data"].get("dataset_tag", "spider"),
        count=int(cfg["data"].get("max_train_examples", 0) or 0),
    )
    configure_logging(output_dir / "logging.log")
    snapshot_config(cfg, output_dir / "config.yaml")
    LOGGER.info("output_dir=%s", output_dir)

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("trl is required for SFT. pip install trl") from exc

    loader_cfg = LoaderConfig(**cfg["model"])
    model, tokenizer = load_model_and_tokenizer(loader_cfg)

    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(
            "tokenizer has no chat_template; SFT uses the 'messages' format which "
            "requires a chat template. Set model.chat_template_override in the config "
            "or choose an instruct-tuned base model."
        )

    train_ds = _build_dataset(
        cfg, mode=mode, split="train",
        limit=cfg["data"].get("max_train_examples"),
    )
    eval_ds = _build_dataset(
        cfg, mode=mode, split="dev",
        limit=cfg["data"].get("max_eval_examples"),
    )

    sft_args_cfg = dict(cfg.get("sft", {}))
    sft_args_cfg.setdefault("output_dir", str(output_dir / "checkpoints"))
    sft_args_cfg.setdefault("logging_dir", str(output_dir / "tb"))
    sft_args_cfg.setdefault("report_to", ["none"])
    sft_args = SFTConfig(**sft_args_cfg)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    LOGGER.info("starting SFT: train=%d eval=%d", len(train_ds), len(eval_ds))
    trainer.train()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    LOGGER.info("saved final checkpoint to %s", final_dir)

    summary = {
        "variant": f"sft_{mode}",
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds),
        "final_checkpoint": str(final_dir),
    }
    (output_dir / "summary.txt").write_text(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SFT training (TRL)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, help="override model.name_or_path")
    parser.add_argument("--max-train", type=int, default=None, help="override max train examples")
    parser.add_argument("--max-eval", type=int, default=None, help="override max eval examples")
    parser.add_argument("--round", type=str, default=None, help="override round name")
    args = parser.parse_args(argv)

    extra: dict = {}
    if args.model is not None:
        extra.setdefault("model", {})["name_or_path"] = args.model
    if args.max_train is not None:
        extra.setdefault("data", {})["max_train_examples"] = args.max_train
    if args.max_eval is not None:
        extra.setdefault("data", {})["max_eval_examples"] = args.max_eval
    if args.round is not None:
        extra.setdefault("run", {})["round_name"] = args.round

    return run_sft(args.config, extra)


if __name__ == "__main__":
    sys.exit(main())
