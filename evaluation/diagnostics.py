"""Lightweight diagnostics for generated model outputs."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..sql.validator import extract_sql, extract_sketch_text, is_valid_sql

__all__ = ["compute_diagnostics"]


_SQL_OPEN = re.compile(r"<sql>", re.IGNORECASE)
_SQL_CLOSE = re.compile(r"</sql>", re.IGNORECASE)
_SKETCH_OPEN = re.compile(r"<sketch>", re.IGNORECASE)
_SKETCH_CLOSE = re.compile(r"</sketch>", re.IGNORECASE)


def _rate(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def _has_unclosed_block(text: str) -> bool:
    return (
        len(_SQL_OPEN.findall(text)) > len(_SQL_CLOSE.findall(text))
        or len(_SKETCH_OPEN.findall(text)) > len(_SKETCH_CLOSE.findall(text))
    )


@dataclass
class _DiagnosticCounts:
    sql_block: int = 0
    sketch_block: int = 0
    sketch_open_no_sql: int = 0
    no_sql: int = 0
    invalid_sql: int = 0
    unclosed: int = 0
    near_limit: int = 0


def _has_closed_sql_block(text: str) -> bool:
    return _SQL_OPEN.search(text) is not None and _SQL_CLOSE.search(text) is not None


def _near_token_limit(pred: dict[str, Any], threshold: int) -> bool:
    return bool(threshold and int(pred.get("new_token_count", 0) or 0) >= threshold)


def _update_counts(counts: _DiagnosticCounts, pred: dict[str, Any], threshold: int) -> None:
    text = pred.get("pred_text", "") or ""
    pred_sql = extract_sql(text)
    has_sketch_open = _SKETCH_OPEN.search(text) is not None

    counts.sql_block += int(_has_closed_sql_block(text))
    counts.sketch_block += int(bool(extract_sketch_text(text)))
    counts.unclosed += int(_has_unclosed_block(text))
    counts.near_limit += int(_near_token_limit(pred, threshold))

    if pred_sql is None:
        counts.no_sql += 1
        counts.sketch_open_no_sql += int(has_sketch_open)
    elif not is_valid_sql(pred_sql):
        counts.invalid_sql += 1


def compute_diagnostics(preds: list[dict[str, Any]], max_new_tokens: int) -> dict[str, float]:
    """Compute format and truncation diagnostics for generated predictions."""
    n = len(preds)
    counts = _DiagnosticCounts()
    near_limit_threshold = int(max_new_tokens * 0.95) if max_new_tokens > 0 else 0

    for pred in preds:
        _update_counts(counts, pred, near_limit_threshold)

    return {
        "n": float(n),
        "sql_block_present_rate": _rate(counts.sql_block, n),
        "sketch_block_present_rate": _rate(counts.sketch_block, n),
        "sketch_open_no_sql_rate": _rate(counts.sketch_open_no_sql, n),
        "no_sql_extracted_rate": _rate(counts.no_sql, n),
        "invalid_sql_rate": _rate(counts.invalid_sql, n),
        "unclosed_block_rate": _rate(counts.unclosed, n),
        "near_token_limit_rate": _rate(counts.near_limit, n),
    }
