"""Execution-based evaluation metrics for Text-to-SQL.

The primary metric is *execution match*: the predicted query is considered
correct iff it executes successfully and its result set equals the gold
result set on the same database. We use a permutation-invariant comparison
by default (set of row tuples) and optionally fall back to an ordered
comparison for queries that have an ORDER BY clause.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .executor import ExecutionResult, execute_sql, spider_db_path
from .validator import extract_sql, is_valid_sql

__all__ = [
    "execution_match",
    "validity_rate",
    "evaluate_predictions",
    "PerExampleMetric",
]


_ORDER_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


def results_equal(pred: ExecutionResult, gold: ExecutionResult, gold_sql: str) -> bool:
    """Permutation-invariant result-set equality, ordered when gold has ORDER BY."""
    if not gold.ok or not pred.ok:
        return False
    if _ORDER_RE.search(gold_sql or ""):
        return list(pred.rows or []) == list(gold.rows or [])
    return pred.as_set() == gold.as_set()


# Backwards-compatible alias kept for internal callers.
_results_equal = results_equal


def execution_match(
    pred_sql: str,
    gold_sql: str,
    db_path: str | Path,
    *,
    timeout_s: float = 5.0,
) -> tuple[bool, ExecutionResult, ExecutionResult]:
    """Run both queries and check result-set equality.

    Returns ``(match, pred_result, gold_result)`` so callers can log
    intermediate state (useful for debugging and for reward shaping).
    """
    pred_res = execute_sql(pred_sql, db_path, timeout_s=timeout_s) if pred_sql else ExecutionResult(ok=False, error="no sql")
    gold_res = execute_sql(gold_sql, db_path, timeout_s=timeout_s) if gold_sql else ExecutionResult(ok=False, error="no gold")
    return _results_equal(pred_res, gold_res, gold_sql), pred_res, gold_res


@dataclass
class PerExampleMetric:
    idx: int
    db_id: str
    question: str
    gold_sql: str
    pred_text: str
    pred_sql: Optional[str]
    valid: bool
    executable: bool
    match: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_predictions(
    predictions: list[dict],
    spider_root: str | Path,
    *,
    timeout_s: float = 5.0,
) -> tuple[list[PerExampleMetric], dict[str, float]]:
    """Score a list of predictions against Spider gold SQL via execution match.

    Each prediction dict must have: ``idx``, ``db_id``, ``question``,
    ``gold_sql``, and ``pred_text`` (the raw model output).
    """
    spider_root = Path(spider_root)
    per_example: list[PerExampleMetric] = []
    n_valid = 0
    n_exec = 0
    n_match = 0

    for p in predictions:
        pred_text = p.get("pred_text", "") or ""
        pred_sql = extract_sql(pred_text)
        valid = bool(pred_sql) and is_valid_sql(pred_sql)
        db_path = spider_db_path(spider_root, p["db_id"])

        match = False
        executable = False
        error: Optional[str] = None
        if pred_sql:
            ok, pred_res, _gold_res = execution_match(
                pred_sql, p["gold_sql"], db_path, timeout_s=timeout_s
            )
            executable = pred_res.ok
            match = ok
            if not pred_res.ok:
                error = pred_res.error
        else:
            error = "no sql extracted"

        per_example.append(
            PerExampleMetric(
                idx=p["idx"],
                db_id=p["db_id"],
                question=p["question"],
                gold_sql=p["gold_sql"],
                pred_text=pred_text,
                pred_sql=pred_sql,
                valid=valid,
                executable=executable,
                match=match,
                error=error,
            )
        )
        n_valid += int(valid)
        n_exec += int(executable)
        n_match += int(match)

    n = max(1, len(predictions))
    summary = {
        "n": float(len(predictions)),
        "execution_accuracy": n_match / n,
        "executable_rate": n_exec / n,
        "validity_rate": n_valid / n,
    }
    return per_example, summary


def validity_rate(predictions: list[dict]) -> float:
    n = max(1, len(predictions))
    valid = 0
    for p in predictions:
        sql = extract_sql(p.get("pred_text", "") or "")
        if sql and is_valid_sql(sql):
            valid += 1
    return valid / n
