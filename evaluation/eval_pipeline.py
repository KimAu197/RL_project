"""Evaluation loop: generate predictions with a trained model and score them.

This is the only file in ``evaluation/`` that contains the evaluation
loop. All shared helpers live in ``evaluation/utils.py`` and model / SQL
utilities are imported from their dedicated modules.

Outputs (matching the user's workspace convention):

    results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/
        logging.log
        metrics.csv
        summary.txt
        answer.json
        config.yaml
        log/
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..data.prompt_builder import apply_chat_template, build_prompt
from ..data.spider_dataset import load_spider_splits
from ..models.generation import GenerationConfigLite, generate_batched
from ..models.loader import LoaderConfig, load_model_and_tokenizer
from ..sql.metrics import evaluate_predictions
from .utils import (
    configure_logging,
    deep_merge,
    load_config,
    resolve_output_dir,
    snapshot_config,
)

LOGGER = logging.getLogger("eval_pipeline")


_METRIC_COLS = ("idx", "db_id", "valid", "executable", "match", "error")


def _write_metrics_csv(metrics: list[Any], summary: dict[str, float], path: Path) -> None:
    """Write a clean per-example metrics table plus a summary footer.

    ``pred_text`` is intentionally excluded from metrics.csv — full
    predictions live in answer.json. This keeps metrics.csv easy to load
    in pandas / spreadsheets for the ablation table.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_METRIC_COLS)
        for m in metrics:
            d = m.to_dict()
            writer.writerow([d.get(k) for k in _METRIC_COLS])
        writer.writerow([])
        writer.writerow(["# summary"])
        for k, v in summary.items():
            writer.writerow([k, v])


def _write_summary(summary: dict[str, float], variant: str, path: Path) -> None:
    lines = [
        "Sketch-to-SQL evaluation summary",
        "================================",
        f"variant           : {variant}",
        f"n_examples        : {int(summary.get('n', 0))}",
        f"execution_accuracy: {summary.get('execution_accuracy', 0.0):.4f}",
        f"executable_rate   : {summary.get('executable_rate', 0.0):.4f}",
        f"validity_rate     : {summary.get('validity_rate', 0.0):.4f}",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_answer_json(metrics: list[Any], path: Path, detailed: bool) -> None:
    payload = []
    for m in metrics:
        d = m.to_dict()
        if not detailed:
            d.pop("pred_text", None)
        payload.append(d)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def run_eval(config_path: str, cli_overrides: dict) -> int:
    cfg = load_config(config_path)
    cfg = deep_merge(cfg, cli_overrides)

    mode = cfg["prompt"]["mode"]
    assert mode in ("direct", "sketch")

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    gen_cfg_dict = cfg.get("generation", {})
    count = int(data_cfg.get("max_eval_examples", 0) or 0)

    output_dir = resolve_output_dir(
        base=run_cfg.get("output_root", "results"),
        round_name=run_cfg.get("round_name", "eval"),
        model_name=run_cfg.get("model_tag") or cfg["model"]["name_or_path"].split("/")[-1],
        dataset=data_cfg.get("dataset_tag", "spider"),
        count=count,
    )
    configure_logging(output_dir / "logging.log")
    snapshot_config(cfg, output_dir / "config.yaml")
    LOGGER.info("output_dir=%s mode=%s", output_dir, mode)

    LOGGER.info("loading model: %s", cfg["model"]["name_or_path"])
    model, tokenizer = load_model_and_tokenizer(LoaderConfig(**cfg["model"]))

    LOGGER.info("loading spider dev split")
    records = load_spider_splits(
        data_cfg["spider_root"],
        split=data_cfg.get("eval_split", "dev"),
        mode=mode,
        max_schema_chars=int(data_cfg.get("max_schema_chars", 4000)),
        limit=(count or None),
    )
    LOGGER.info("got %d examples", len(records))

    prompts = []
    for r in records:
        spec = build_prompt(r.schema_text, r.question, mode=mode)
        text = apply_chat_template(tokenizer, spec, target=None, add_generation_prompt=True)
        prompts.append(text)

    gen_cfg = GenerationConfigLite(
        max_new_tokens=int(gen_cfg_dict.get("max_new_tokens", 512)),
        temperature=float(gen_cfg_dict.get("temperature", 0.0)),
        top_p=float(gen_cfg_dict.get("top_p", 1.0)),
        do_sample=bool(gen_cfg_dict.get("do_sample", False)),
        num_return_sequences=1,
        repetition_penalty=float(gen_cfg_dict.get("repetition_penalty", 1.0)),
    )
    batch_size = int(gen_cfg_dict.get("batch_size", 4))
    LOGGER.info("running generation: batch_size=%d", batch_size)

    completions_nested = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="generate"):
        chunk = prompts[start : start + batch_size]
        completions_nested.extend(generate_batched(model, tokenizer, chunk, gen_cfg, batch_size=batch_size))

    preds = []
    for i, (r, comps) in enumerate(zip(records, completions_nested)):
        pred_text = comps[0] if comps else ""
        preds.append(
            {
                "idx": i,
                "db_id": r.db_id,
                "question": r.question,
                "gold_sql": r.sql,
                "pred_text": pred_text,
            }
        )

    LOGGER.info("scoring %d predictions", len(preds))
    per_example, summary = evaluate_predictions(
        preds,
        spider_root=data_cfg["spider_root"],
        timeout_s=float(cfg.get("eval", {}).get("timeout_s", 5.0)),
    )

    _write_metrics_csv(per_example, summary, output_dir / "metrics.csv")
    _write_summary(summary, variant=f"{mode}_{run_cfg.get('round_name','eval')}", path=output_dir / "summary.txt")
    detailed = bool(cfg.get("eval", {}).get("detailed", True))
    _write_answer_json(per_example, output_dir / "answer.json", detailed=detailed)

    LOGGER.info(
        "done: exec_acc=%.4f executable=%.4f validity=%.4f",
        summary.get("execution_accuracy", 0.0),
        summary.get("executable_rate", 0.0),
        summary.get("validity_rate", 0.0),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sketch-to-SQL evaluation pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, help="override model.name_or_path")
    parser.add_argument("--round", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--detailed", type=str, default=None, help="true|false")
    args = parser.parse_args(argv)

    overrides: dict = {}
    if args.model is not None:
        overrides["model"] = {"name_or_path": args.model}
    if args.round is not None:
        overrides["run"] = {"round_name": args.round}
    if args.dataset is not None:
        overrides["data"] = {"dataset_tag": args.dataset}
    if args.count is not None:
        overrides.setdefault("data", {})["max_eval_examples"] = args.count
    if args.detailed is not None:
        overrides["eval"] = {"detailed": args.detailed.lower() == "true"}

    return run_eval(args.config, overrides)


if __name__ == "__main__":
    sys.exit(main())
