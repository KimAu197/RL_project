"""Parse model outputs to extract SQL (and optional sketch), and validate syntax."""
from __future__ import annotations

import re
from typing import Optional

import sqlglot

__all__ = ["extract_sql", "extract_sketch_text", "is_valid_sql"]


_SQL_BLOCK = re.compile(r"<sql>\s*(.*?)\s*</sql>", re.IGNORECASE | re.DOTALL)
_SKETCH_BLOCK = re.compile(r"<sketch>\s*(.*?)\s*</sketch>", re.IGNORECASE | re.DOTALL)
_FENCE_SQL = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SKETCH_OPEN = re.compile(r"<sketch>", re.IGNORECASE)
_SKETCH_LABEL = re.compile(
    r"^\s*(TABLES|JOINS|SELECT|AGGREGATIONS|WHERE|GROUP_BY|HAVING|"
    r"SUBQUERIES|ORDER_BY|LIMIT|SET_OP|SET_RHS)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
_SQL_KEYWORDS = ("select", "with ", "insert", "update", "delete")


def extract_sql(text: str) -> Optional[str]:
    """Return the SQL query found in the model response, or None if missing.

    Resolution order:

    1. Last well-formed ``<sql>...</sql>`` block (matches what we train on).
    2. Last fenced code block (``` or ```sql) appearing AFTER any ``<sketch>``
       block, so sketch content is never mistaken for SQL.
    3. Plain SELECT-ish text fallback, but ONLY when the response shows no
       sketch markers (open ``<sketch>`` tag or sketch labels like
       ``TABLES:`` / ``SELECT:`` at the start of a line). This prevents the
       fallback from grabbing the sketch text when ``<sql>`` is missing or
       truncated.
    """
    if not text:
        return None

    sql_matches = list(_SQL_BLOCK.finditer(text))
    if sql_matches:
        return sql_matches[-1].group(1).strip().rstrip(";")

    sketch_open = _SKETCH_OPEN.search(text)
    sketch_match = _SKETCH_BLOCK.search(text)
    if sketch_match is not None:
        fence_search_start = sketch_match.end()
    elif sketch_open is not None:
        fence_search_start = sketch_open.end()
    else:
        fence_search_start = 0
    fence_matches = [
        m for m in _FENCE_SQL.finditer(text) if m.start() >= fence_search_start
    ]
    if fence_matches:
        return fence_matches[-1].group(1).strip().rstrip(";")

    if sketch_open is not None or _SKETCH_LABEL.search(text):
        return None

    stripped = text.strip().rstrip(";")
    lower = stripped.lower()
    if any(kw in lower for kw in _SQL_KEYWORDS):
        return stripped
    return None


def extract_sketch_text(text: str) -> Optional[str]:
    """Return the raw sketch text if present, else None."""
    if not text:
        return None
    m = _SKETCH_BLOCK.search(text)
    if m:
        return m.group(1).strip()
    return None


def is_valid_sql(sql: str, dialect: str = "sqlite") -> bool:
    if not sql:
        return False
    try:
        trees = sqlglot.parse(sql, read=dialect)
    except Exception:
        return False
    return any(t is not None for t in trees)
