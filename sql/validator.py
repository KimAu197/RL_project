"""Parse model outputs to extract SQL (and optional sketch), and validate syntax."""
from __future__ import annotations

import re
from typing import Optional

import sqlglot

__all__ = ["extract_sql", "extract_sketch_text", "is_valid_sql"]


_SQL_BLOCK = re.compile(r"<sql>\s*(.*?)\s*</sql>", re.IGNORECASE | re.DOTALL)
_SKETCH_BLOCK = re.compile(r"<sketch>\s*(.*?)\s*</sketch>", re.IGNORECASE | re.DOTALL)
_FENCE_SQL = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> Optional[str]:
    """Return the SQL query found in the model response, or None if missing.

    Tries, in order:

    1. ``<sql>...</sql>`` block (the format we train on).
    2. Fenced code block ```sql ... ``` as a fallback.
    3. The last SELECT-ish line in the text.
    """
    if not text:
        return None

    m = _SQL_BLOCK.search(text)
    if m:
        return m.group(1).strip().rstrip(";")

    m = _FENCE_SQL.search(text)
    if m:
        return m.group(1).strip().rstrip(";")

    stripped = text.strip().rstrip(";")
    lower = stripped.lower()
    if any(kw in lower for kw in ("select", "with ", "insert", "update", "delete")):
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
