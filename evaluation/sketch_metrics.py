"""Reusable metrics for comparing predicted sketches with gold SQL sketches."""
from __future__ import annotations

from dataclasses import dataclass

from ..data.sketch_extractor import Sketch, SketchExtractionError, extract_sketch, sketch_to_text

__all__ = ["SketchMetrics", "compute_sketch_metrics", "parse_sketch_fields"]


@dataclass(frozen=True)
class SketchMetrics:
    sketch_present: bool
    format_ok: bool
    table_recall: float
    table_precision: float
    table_f1: float
    agg_match: bool
    select_match: bool
    join_match: bool
    where_match: bool
    group_by_match: bool
    having_match: bool
    subquery_match: bool
    order_by_match: bool
    limit_match: bool
    set_op_match: bool
    set_rhs_match: bool
    agg_relevant: bool
    join_relevant: bool
    where_relevant: bool
    group_by_relevant: bool
    having_relevant: bool
    subquery_relevant: bool
    order_by_relevant: bool
    limit_relevant: bool
    set_op_relevant: bool
    set_rhs_relevant: bool


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


def _f1(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold:
        return 0.0, 0.0, 0.0
    if not pred:
        return 0.0, 0.0, 0.0
    overlap = len(gold & pred)
    recall = overlap / len(gold)
    precision = overlap / len(pred)
    if recall + precision == 0:
        return recall, precision, 0.0
    return recall, precision, 2 * recall * precision / (recall + precision)


def _field_set(fields: dict[str, list[str]], label: str) -> set[str]:
    items = fields.get(label, [])
    out: set[str] = set()
    for item in items:
        parts = item.split("|") if label in {"JOINS", "SUBQUERIES"} else [item]
        for part in parts:
            normalized = " ".join(part.strip().lower().split())
            if normalized:
                out.add(normalized)
    return out


def _field_list(fields: dict[str, list[str]], label: str) -> list[str]:
    return [" ".join(item.strip().lower().split()) for item in fields.get(label, []) if item.strip()]


def _matches(fields: dict[str, list[str]], gold_fields: dict[str, list[str]], label: str) -> bool:
    return _field_set(fields, label) == _field_set(gold_fields, label)


def _matches_ordered(fields: dict[str, list[str]], gold_fields: dict[str, list[str]], label: str) -> bool:
    return _field_list(fields, label) == _field_list(gold_fields, label)


def _relevant(fields: dict[str, list[str]], gold_fields: dict[str, list[str]], label: str) -> bool:
    return bool(_field_set(fields, label) or _field_set(gold_fields, label))


def _empty_metrics(sketch_present: bool, format_ok: bool) -> SketchMetrics:
    return SketchMetrics(
        sketch_present, format_ok, 0.0, 0.0, 0.0, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
    )


def _gold_fields(gold_sketch: Sketch) -> dict[str, list[str]]:
    return parse_sketch_fields(sketch_to_text(gold_sketch))


def compute_sketch_metrics(
    sketch_text: str | None,
    gold_sql: str,
    dialect: str = "sqlite",
) -> SketchMetrics:
    """Compare a predicted sketch block against the deterministic gold sketch."""
    sketch_present = bool((sketch_text or "").strip())
    if not sketch_present:
        return _empty_metrics(False, False)

    fields = parse_sketch_fields(sketch_text or "")
    format_ok = "TABLES" in fields and "SELECT" in fields

    try:
        gold_sketch = extract_sketch(gold_sql, dialect=dialect)
    except SketchExtractionError:
        return _empty_metrics(sketch_present, format_ok)
    gold_fields = _gold_fields(gold_sketch)

    gold_tables = _normalized_set(gold_sketch.tables)
    pred_tables = _normalized_set(fields.get("TABLES", []))
    table_recall, table_precision, table_f1 = _f1(pred_tables, gold_tables)

    gold_aggs = set(gold_sketch.aggregations)
    pred_aggs = _normalized_set(fields.get("AGGREGATIONS", []), upper=True)
    agg_match = gold_aggs == pred_aggs
    agg_relevant = bool(gold_aggs or pred_aggs)

    return SketchMetrics(
        sketch_present=sketch_present,
        format_ok=format_ok,
        table_recall=table_recall,
        table_precision=table_precision,
        table_f1=table_f1,
        agg_match=agg_match,
        select_match=_matches_ordered(fields, gold_fields, "SELECT"),
        join_match=_matches(fields, gold_fields, "JOINS"),
        where_match=_matches(fields, gold_fields, "WHERE"),
        group_by_match=_matches(fields, gold_fields, "GROUP_BY"),
        having_match=_matches(fields, gold_fields, "HAVING"),
        subquery_match=_matches(fields, gold_fields, "SUBQUERIES"),
        order_by_match=_matches(fields, gold_fields, "ORDER_BY"),
        limit_match=_matches(fields, gold_fields, "LIMIT"),
        set_op_match=_matches(fields, gold_fields, "SET_OP"),
        set_rhs_match=_matches(fields, gold_fields, "SET_RHS"),
        agg_relevant=agg_relevant,
        join_relevant=_relevant(fields, gold_fields, "JOINS"),
        where_relevant=_relevant(fields, gold_fields, "WHERE"),
        group_by_relevant=_relevant(fields, gold_fields, "GROUP_BY"),
        having_relevant=_relevant(fields, gold_fields, "HAVING"),
        subquery_relevant=_relevant(fields, gold_fields, "SUBQUERIES"),
        order_by_relevant=_relevant(fields, gold_fields, "ORDER_BY"),
        limit_relevant=_relevant(fields, gold_fields, "LIMIT"),
        set_op_relevant=_relevant(fields, gold_fields, "SET_OP"),
        set_rhs_relevant=_relevant(fields, gold_fields, "SET_RHS"),
    )
