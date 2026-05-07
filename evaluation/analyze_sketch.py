"""Analyze generated sketches vs gold sketches on an evaluation run.

Takes an ``answer.json`` produced by the evaluation pipeline (in sketch
mode) and measures how well the model produced the intermediate sketch:

* sketch_present_rate : fraction of outputs containing a <sketch>...</sketch> block
* table_recall        : fraction of gold tables appearing in the predicted sketch
* agg_match_rate      : fraction of examples whose aggregations set matches gold

This is intended for the qualitative analysis section of the project
report; it does not affect training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..data.sketch_extractor import SketchExtractionError, extract_sketch
from ..sql.validator import extract_sketch_text


def _tables_from_pred_sketch(sketch_text: str) -> set[str]:
    for line in sketch_text.splitlines():
        if line.upper().startswith("TABLES:"):
            raw = line.split(":", 1)[1].strip()
            if raw == "-" or not raw:
                return set()
            return {t.strip() for t in raw.split(",") if t.strip()}
    return set()


def _aggs_from_pred_sketch(sketch_text: str) -> set[str]:
    for line in sketch_text.splitlines():
        if line.upper().startswith("AGGREGATIONS:"):
            raw = line.split(":", 1)[1].strip()
            return {t.strip().upper() for t in raw.split(",") if t.strip()}
    return set()


def analyze(answer_path: str | Path) -> dict:
    records = json.loads(Path(answer_path).read_text())
    n = len(records)
    if n == 0:
        return {"n": 0}

    n_sketch = 0
    table_hits = 0
    table_total = 0
    agg_match = 0
    agg_total = 0
    for r in records:
        pred_text = r.get("pred_text", "") or ""
        sketch_text = extract_sketch_text(pred_text)
        if sketch_text:
            n_sketch += 1

        gold_sql = r.get("gold_sql", "")
        try:
            gold_sketch = extract_sketch(gold_sql)
        except SketchExtractionError:
            continue

        gold_tables = set(gold_sketch.tables)
        if gold_tables:
            pred_tables = _tables_from_pred_sketch(sketch_text or "")
            hits = len(gold_tables & pred_tables)
            table_hits += hits
            table_total += len(gold_tables)

        gold_aggs = set(gold_sketch.aggregations)
        if gold_aggs or sketch_text:
            pred_aggs = _aggs_from_pred_sketch(sketch_text or "")
            if gold_aggs == pred_aggs:
                agg_match += 1
            agg_total += 1

    return {
        "n": n,
        "sketch_present_rate": n_sketch / n,
        "table_recall": (table_hits / table_total) if table_total else 0.0,
        "agg_match_rate": (agg_match / agg_total) if agg_total else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze sketch quality from answer.json")
    parser.add_argument("--answer", type=str, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.answer)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
