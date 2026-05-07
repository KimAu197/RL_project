"""Reusable metrics for comparing predicted sketches with gold SQL sketches."""
from __future__ import annotations

from dataclasses import dataclass

from ..data.sketch_extractor import SketchExtractionError, extract_sketch

__all__ = ["SketchMetrics", "compute_sketch_metrics", "parse_sketch_fields"]


@dataclass(frozen=True)
class SketchMetrics:
    sketch_present: bool
    format_ok: bool
    table_recall: float
    agg_match: bool


def parse_sketch_fields(sketch_text: str) -> dict[str, list[str]]:
    """Parse ``LABEL: value`` sketch lines into normalized field lists."""
    fields: dict[str, list[str]] = {}
    for raw_line in (sketch_text or "").splitlines():
        if ":" not in raw_line:
            continue
        label, raw_value = raw_line.split(":", 1)
        label = label.strip().upper()
        value = raw_value.strip()
        if not label or not value or value == "-":
            fields[label] = []
            continue
        fields[label] = [part.strip() for part in value.split(",") if part.strip()]
    return fields


def _normalized_set(items: list[str], *, upper: bool = False) -> set[str]:
    if upper:
        return {item.upper() for item in items if item}
    return {item.lower() for item in items if item}


def compute_sketch_metrics(
    sketch_text: str | None,
    gold_sql: str,
    dialect: str = "sqlite",
) -> SketchMetrics:
    """Compare a predicted sketch block against the deterministic gold sketch."""
    sketch_present = bool((sketch_text or "").strip())
    if not sketch_present:
        return SketchMetrics(False, False, 0.0, False)

    fields = parse_sketch_fields(sketch_text or "")
    format_ok = "TABLES" in fields and "SELECT" in fields

    try:
        gold_sketch = extract_sketch(gold_sql, dialect=dialect)
    except SketchExtractionError:
        return SketchMetrics(sketch_present, format_ok, 0.0, False)

    gold_tables = _normalized_set(gold_sketch.tables)
    pred_tables = _normalized_set(fields.get("TABLES", []))
    table_recall = (len(gold_tables & pred_tables) / len(gold_tables)) if gold_tables else 0.0

    gold_aggs = set(gold_sketch.aggregations)
    pred_aggs = _normalized_set(fields.get("AGGREGATIONS", []), upper=True)
    agg_match = gold_aggs == pred_aggs

    return SketchMetrics(
        sketch_present=sketch_present,
        format_ok=format_ok,
        table_recall=table_recall,
        agg_match=agg_match,
    )
