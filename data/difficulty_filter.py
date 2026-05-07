"""Difficulty filtering for GRPO train data.

The filter samples an SFT model several times per Spider train prompt, scores
each completion with the same execution reward used by GRPO, and keeps examples
whose pass rate is neither saturated nor hopeless.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from ..evaluation.utils import configure_logging, deep_merge, load_config, snapshot_config
from ..models.generation import GenerationConfigLite, generate_batched
from ..models.loader import LoaderConfig, load_model_and_tokenizer
from ..sql.validator import extract_sketch_text, extract_sql, is_valid_sql
from ..training.reward import RewardConfig, build_reward_fn
from .prompt_builder import apply_chat_template, build_prompt
from .spider_dataset import SpiderRecord, load_spider_splits

LOGGER = logging.getLogger("difficulty_filter")

__all__ = [
    "DifficultyFilterConfig",
    "DifficultyStats",
    "compute_difficulty_stats",
    "keep_by_difficulty",
    "run_filter",
]


@dataclass
class DifficultyFilterConfig:
    min_pass_rate: float = 0.125
    max_pass_rate: float = 0.875
    min_reward_std: float = 1.0e-8
    min_has_sql_rate: float = 0.5
    min_valid_sql_rate: float = 0.0
    require_sketch: bool = False


@dataclass
class DifficultyStats:
    num_samples: int
    pass_rate: float
    mean_reward: float
    reward_std: float
    has_sql_rate: float
    valid_sql_rate: float
    has_sketch_rate: float


def _rate(flags: Sequence[bool]) -> float:
    return sum(bool(x) for x in flags) / max(1, len(flags))


def _population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5


def compute_difficulty_stats(
    rewards: Sequence[float],
    pred_sqls: Sequence[str | None],
    valid_flags: Sequence[bool],
    sketch_flags: Sequence[bool],
) -> DifficultyStats:
    """Summarize K sampled completions for one training example."""
    n = max(1, len(rewards))
    return DifficultyStats(
        num_samples=len(rewards),
        pass_rate=sum(float(r >= 1.0) for r in rewards) / n,
        mean_reward=sum(float(r) for r in rewards) / n,
        reward_std=_population_std([float(r) for r in rewards]),
        has_sql_rate=sum(sql is not None for sql in pred_sqls) / max(1, len(pred_sqls)),
        valid_sql_rate=_rate(valid_flags),
        has_sketch_rate=_rate(sketch_flags),
    )


def keep_by_difficulty(stats: DifficultyStats, cfg: DifficultyFilterConfig) -> bool:
    """Return true when an example has useful GRPO reward variation."""
    if stats.pass_rate < cfg.min_pass_rate or stats.pass_rate > cfg.max_pass_rate:
        return False
    if stats.reward_std < cfg.min_reward_std:
        return False
    if stats.has_sql_rate < cfg.min_has_sql_rate:
        return False
    if stats.valid_sql_rate < cfg.min_valid_sql_rate:
        return False
    if cfg.require_sketch and stats.has_sketch_rate <= 0.0:
        return False
    return True


def _score_completions(
    reward_fn,
    completions: list[str],
    db_id: str,
    gold_sql: str,
) -> list[float]:
    return reward_fn(
        completions=completions,
        db_id=[db_id] * len(completions),
        gold_sql=[gold_sql] * len(completions),
    )


def _write_filtered_jsonl(records: list[SpiderRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def _write_stats_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "idx",
        "keep",
        "db_id",
        "question",
        "num_samples",
        "pass_rate",
        "mean_reward",
        "reward_std",
        "has_sql_rate",
        "valid_sql_rate",
        "has_sketch_rate",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_summary(path: Path, total: int, kept: int, filter_cfg: DifficultyFilterConfig) -> None:
    lines = [
        "Difficulty filtering summary",
        "============================",
        f"total_examples : {total}",
        f"kept_examples  : {kept}",
        f"keep_rate      : {kept / max(1, total):.4f}",
        "",
        "Filter config:",
        json.dumps(asdict(filter_cfg), indent=2),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_filter(config_path: str, cli_overrides: dict[str, Any]) -> int:
    cfg = deep_merge(load_config(config_path), cli_overrides)
    mode = cfg["prompt"]["mode"]
    if mode not in ("direct", "sketch"):
        raise ValueError(f"unknown prompt mode: {mode}")

    filter_cfg_dict = cfg.get("difficulty_filter", {})
    filter_cfg = DifficultyFilterConfig(
        min_pass_rate=float(filter_cfg_dict.get("min_pass_rate", 0.125)),
        max_pass_rate=float(filter_cfg_dict.get("max_pass_rate", 0.875)),
        min_reward_std=float(filter_cfg_dict.get("min_reward_std", 1.0e-8)),
        min_has_sql_rate=float(filter_cfg_dict.get("min_has_sql_rate", 0.5)),
        min_valid_sql_rate=float(filter_cfg_dict.get("min_valid_sql_rate", 0.0)),
        require_sketch=bool(filter_cfg_dict.get("require_sketch", mode == "sketch")),
    )

    data_cfg = cfg["data"]
    output_dir = Path(filter_cfg_dict.get("output_dir") or Path(data_cfg.get("processed_dir", "datasets/spider/processed")) / "difficulty_filter")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "logging.log")
    snapshot_config(cfg, output_dir / "config.yaml")

    samples = int(filter_cfg_dict.get("samples", 8))
    if samples <= 0:
        raise ValueError("difficulty_filter.samples must be positive")

    LOGGER.info("loading model: %s", cfg["model"]["name_or_path"])
    model, tokenizer = load_model_and_tokenizer(LoaderConfig(**cfg["model"]))

    limit = data_cfg.get("max_train_examples")
    LOGGER.info("loading Spider train split mode=%s limit=%s", mode, limit)
    records = load_spider_splits(
        data_cfg["spider_root"],
        split="train",
        mode=mode,
        max_schema_chars=int(data_cfg.get("max_schema_chars", 4000)),
        limit=limit,
    )

    prompts = []
    for rec in records:
        spec = build_prompt(rec.schema_text, rec.question, mode=mode)
        prompts.append(apply_chat_template(tokenizer, spec, target=None, add_generation_prompt=True))

    gen_cfg_dict = cfg.get("generation", {})
    temperature = float(gen_cfg_dict.get("temperature", 0.7))
    if temperature <= 0.0:
        raise ValueError("difficulty filtering samples with do_sample=True, so generation.temperature must be > 0")
    gen_cfg = GenerationConfigLite(
        max_new_tokens=int(gen_cfg_dict.get("max_new_tokens", 512)),
        temperature=temperature,
        top_p=float(gen_cfg_dict.get("top_p", 0.95)),
        do_sample=True,
        num_return_sequences=samples,
        repetition_penalty=float(gen_cfg_dict.get("repetition_penalty", 1.0)),
    )
    batch_size = int(gen_cfg_dict.get("batch_size", 1))

    reward_cfg = RewardConfig(
        spider_root=data_cfg["spider_root"],
        mode=mode,
        timeout_s=float(cfg.get("reward", {}).get("timeout_s", 5.0)),
        match_reward=float(cfg.get("reward", {}).get("match_reward", 1.0)),
        validity_bonus=float(cfg.get("reward", {}).get("validity_bonus", 0.0)),
        sketch_bonus=float(cfg.get("reward", {}).get("sketch_bonus", 0.0)),
        no_sql_penalty=float(cfg.get("reward", {}).get("no_sql_penalty", 0.0)),
    )
    reward_fn = build_reward_fn(reward_cfg)

    kept_records: list[SpiderRecord] = []
    stats_rows: list[dict[str, Any]] = []
    LOGGER.info("sampling %d completions per prompt", samples)
    for idx in tqdm(range(0, len(records), batch_size), desc="filter train"):
        batch_records = records[idx : idx + batch_size]
        batch_prompts = prompts[idx : idx + batch_size]
        completions_nested = generate_batched(model, tokenizer, batch_prompts, gen_cfg, batch_size=batch_size)
        for offset, (rec, completions) in enumerate(zip(batch_records, completions_nested)):
            rewards = _score_completions(reward_fn, completions, rec.db_id, rec.sql)
            pred_sqls = [extract_sql(text) for text in completions]
            valid_flags = [bool(sql) and is_valid_sql(sql) for sql in pred_sqls]
            sketch_flags = [extract_sketch_text(text) is not None for text in completions]
            stats = compute_difficulty_stats(rewards, pred_sqls, valid_flags, sketch_flags)
            keep = keep_by_difficulty(stats, filter_cfg)
            if keep:
                kept_records.append(rec)
            stats_rows.append(
                {
                    "idx": idx + offset,
                    "keep": keep,
                    "db_id": rec.db_id,
                    "question": rec.question,
                    **asdict(stats),
                }
            )

    filtered_path = output_dir / f"spider_{mode}_train.jsonl"
    _write_filtered_jsonl(kept_records, filtered_path)
    _write_stats_csv(stats_rows, output_dir / "difficulty_stats.csv")
    _write_summary(output_dir / "summary.txt", total=len(records), kept=len(kept_records), filter_cfg=filter_cfg)
    LOGGER.info("kept %d/%d examples; wrote %s", len(kept_records), len(records), filtered_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter Spider train examples by SFT sampling difficulty")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, help="SFT model path or HF repo id")
    parser.add_argument("--variant", choices=("direct", "sketch"), default=None)
    parser.add_argument("--count", type=int, default=None, help="number of train examples; 0 = full train split")
    parser.add_argument("--samples", type=int, default=None, help="samples per prompt")
    parser.add_argument("--batch-size", type=int, default=None, help="prompts per generation batch")
    parser.add_argument("--temperature", type=float, default=None, help="sampling temperature; must be > 0")
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--min-pass-rate", type=float, default=None)
    parser.add_argument("--max-pass-rate", type=float, default=None)
    parser.add_argument("--min-has-sql-rate", type=float, default=None)
    parser.add_argument("--min-valid-sql-rate", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.model is not None:
        overrides.setdefault("model", {})["name_or_path"] = args.model
    if args.variant is not None:
        overrides.setdefault("prompt", {})["mode"] = args.variant
    if args.count is not None:
        overrides.setdefault("data", {})["max_train_examples"] = None if args.count == 0 else args.count
    for attr, key in (
        ("batch_size", "batch_size"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
    ):
        value = getattr(args, attr)
        if value is not None:
            overrides.setdefault("generation", {})[key] = value
    for attr, key in (
        ("samples", "samples"),
        ("min_pass_rate", "min_pass_rate"),
        ("max_pass_rate", "max_pass_rate"),
        ("min_has_sql_rate", "min_has_sql_rate"),
        ("min_valid_sql_rate", "min_valid_sql_rate"),
        ("output_dir", "output_dir"),
    ):
        value = getattr(args, attr)
        if value is not None:
            overrides.setdefault("difficulty_filter", {})[key] = value

    return run_filter(args.config, overrides)


if __name__ == "__main__":
    sys.exit(main())
