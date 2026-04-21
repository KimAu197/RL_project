"""Materialize processed JSONL files for every (split, mode) pair.

Reading Spider + running sqlglot on every gold SQL for every epoch is
wasteful. This script writes a flat JSONL per split / mode combination so
training scripts can just stream them.

Usage:

    python -m project.data.prepare \
        --spider-root datasets/spider \
        --out datasets/spider/processed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .spider_dataset import load_spider_splits, write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess Spider into prompt/target JSONLs")
    parser.add_argument("--spider-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-schema-chars", type=int, default=4000)
    args = parser.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for mode in ("direct", "sketch"):
        for split in ("train", "dev"):
            records = load_spider_splits(
                args.spider_root,
                split=split,
                mode=mode,
                max_schema_chars=args.max_schema_chars,
            )
            path = out / f"spider_{mode}_{split}.jsonl"
            write_jsonl(records, path)
            total += len(records)
            print(f"wrote {len(records):>6d} records -> {path}")
    print(f"total records: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
