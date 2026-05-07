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

from ..sql.validator import extract_sketch_text
from .sketch_metrics import compute_sketch_metrics


def analyze(answer_path: str | Path) -> dict:
    records = json.loads(Path(answer_path).read_text())
    n = len(records)
    if n == 0:
        return {"n": 0}

    n_sketch = 0
    table_recall_sum = 0.0
    table_total = 0
    agg_match = 0
    agg_total = 0
    for r in records:
        pred_text = r.get("pred_text", "") or ""
        sketch_text = extract_sketch_text(pred_text)
        if sketch_text:
            n_sketch += 1

        metrics = compute_sketch_metrics(sketch_text, r.get("gold_sql", ""))
        table_recall_sum += metrics.table_recall
        table_total += 1
        if metrics.agg_match:
            agg_match += 1
        agg_total += 1

    return {
        "n": n,
        "sketch_present_rate": n_sketch / n,
        "table_recall": (table_recall_sum / table_total) if table_total else 0.0,
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
