"""GRPO post-training entry point (TRL GRPOTrainer) driven by a YAML config.

Supports both ``direct`` and ``sketch`` prompt variants. Reward is produced
by ``training.reward.build_reward_fn`` and defaults to a pure execution
match signal.

Run as ``python -m project.training.grpo_train --config configs/grpo_direct.yaml``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from datasets import Dataset

from ..data.spider_dataset import load_spider_splits, read_jsonl
from ..evaluation.utils import (
    configure_logging,
    deep_merge,
    load_config,
    resolve_output_dir,
    snapshot_config,
)
from ..models.loader import LoaderConfig, load_model_and_tokenizer
from .logging_utils import configure_wandb_reporting
from .reward import RewardConfig, build_reward_fn

LOGGER = logging.getLogger("grpo_train")


def _records_to_prompt_examples(records: list[dict]) -> list[dict]:
    """Convert records into TRL GRPOTrainer conversational prompts.

    TRL's GRPOTrainer accepts either a ``prompt`` string or a ``prompt`` list
    of chat messages. We use the chat form so it applies the tokenizer's
    chat template consistently with SFT.
    """
    out: list[dict] = []
    for r in records:
        out.append(
            {
                "prompt": [
                    {"role": "system", "content": r["prompt_system"]},
                    {"role": "user", "content": r["prompt_user"]},
                ],
                "db_id": r["db_id"],
                "gold_sql": r["sql"],
                "question": r["question"],
            }
        )
    return out


def _build_prompt_dataset(cfg: dict, mode: str, split: str, limit: int | None) -> Dataset:
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
            return Dataset.from_list(_records_to_prompt_examples(records))

    LOGGER.info("materializing Spider records from %s split=%s mode=%s", data_cfg["spider_root"], split, mode)
    records = load_spider_splits(
        data_cfg["spider_root"],
        split=split,
        mode=mode,
        max_schema_chars=int(data_cfg.get("max_schema_chars", 4000)),
        limit=limit,
    )
    return Dataset.from_list(_records_to_prompt_examples([r.to_dict() for r in records]))


def run_grpo(config_path: str, extra: dict) -> int:
    cfg = load_config(config_path)
    cfg = deep_merge(cfg, extra)

    mode = cfg["prompt"]["mode"]
    assert mode in ("direct", "sketch")

    run_cfg = cfg["run"]
    output_dir = resolve_output_dir(
        base=run_cfg.get("output_root", "runs"),
        round_name=run_cfg.get("round_name", "grpo"),
        model_name=run_cfg.get("model_tag") or cfg["model"]["name_or_path"].split("/")[-1],
        dataset=cfg["data"].get("dataset_tag", "spider"),
        count=int(cfg["data"].get("max_train_examples", 0) or 0),
    )
    configure_logging(output_dir / "logging.log")
    snapshot_config(cfg, output_dir / "config.yaml")
    LOGGER.info("output_dir=%s", output_dir)

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("trl>=0.11 is required for GRPO. pip install -U trl") from exc

    loader_cfg = LoaderConfig(**cfg["model"])
    model, tokenizer = load_model_and_tokenizer(loader_cfg)

    train_ds = _build_prompt_dataset(
        cfg, mode=mode, split="train",
        limit=cfg["data"].get("max_train_examples"),
    )

    reward_cfg = RewardConfig(
        spider_root=cfg["data"]["spider_root"],
        mode=mode,
        timeout_s=float(cfg.get("reward", {}).get("timeout_s", 5.0)),
        match_reward=float(cfg.get("reward", {}).get("match_reward", 1.0)),
        validity_bonus=float(cfg.get("reward", {}).get("validity_bonus", 0.0)),
        sketch_bonus=float(cfg.get("reward", {}).get("sketch_bonus", 0.0)),
        sketch_format_bonus=float(cfg.get("reward", {}).get("sketch_format_bonus", 0.0)),
        sketch_table_bonus=float(cfg.get("reward", {}).get("sketch_table_bonus", 0.0)),
        sketch_agg_bonus=float(cfg.get("reward", {}).get("sketch_agg_bonus", 0.0)),
        no_sql_penalty=float(cfg.get("reward", {}).get("no_sql_penalty", 0.0)),
    )
    reward_fn = build_reward_fn(reward_cfg)

    grpo_args_cfg = dict(cfg.get("grpo", {}))
    grpo_args_cfg.setdefault("output_dir", str(output_dir / "checkpoints"))
    grpo_args_cfg.setdefault("logging_dir", str(output_dir / "tb"))
    # Reward needs db_id / gold_sql from the dataset; never let HF drop them.
    grpo_args_cfg["remove_unused_columns"] = False
    grpo_args_cfg = configure_wandb_reporting(
        trainer_args=grpo_args_cfg,
        wandb_cfg=cfg.get("wandb"),
        output_dir=output_dir,
        run_name=output_dir.name,
    )
    grpo_args = GRPOConfig(**grpo_args_cfg)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_args,
        train_dataset=train_ds,
        reward_funcs=reward_fn,
    )

    LOGGER.info("starting GRPO: train=%d", len(train_ds))
    trainer.train()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    LOGGER.info("saved final checkpoint to %s", final_dir)

    summary = {
        "variant": f"grpo_{mode}",
        "train_examples": len(train_ds),
        "final_checkpoint": str(final_dir),
        "reward_config": reward_cfg.__dict__,
    }
    (output_dir / "summary.txt").write_text(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GRPO post-training (TRL)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--round", type=str, default=None)
    parser.add_argument("--init-ckpt", type=str, default=None, help="override model.name_or_path")
    parser.add_argument("--wandb-enabled", choices=("true", "false"), default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    args = parser.parse_args(argv)

    extra: dict = {}
    if args.max_train is not None:
        extra.setdefault("data", {})["max_train_examples"] = args.max_train
    if args.round is not None:
        extra.setdefault("run", {})["round_name"] = args.round
    if args.init_ckpt is not None:
        extra.setdefault("model", {})["name_or_path"] = args.init_ckpt
    if args.wandb_enabled is not None:
        extra.setdefault("wandb", {})["enabled"] = args.wandb_enabled == "true"
    if args.wandb_project is not None:
        extra.setdefault("wandb", {})["project"] = args.wandb_project
    if args.wandb_entity is not None:
        extra.setdefault("wandb", {})["entity"] = args.wandb_entity

    return run_grpo(args.config, extra)


if __name__ == "__main__":
    sys.exit(main())
