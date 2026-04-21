"""Format Spider database schemas into compact prompt-ready strings.

Spider ships a `tables.json` describing every database. We convert each
database into a token-efficient textual schema that the model can read.

Format (per database):

    # Schema for <db_id>
    Table singer(singer_id, name, country, age)
    Table concert(concert_id, singer_id, year)
    FK: concert.singer_id -> singer.singer_id
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["load_tables_json", "format_schema"]


def load_tables_json(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load tables.json and index it by ``db_id``."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {rec["db_id"]: rec for rec in records}


def format_schema(db_entry: dict[str, Any], max_chars: int | None = None) -> str:
    """Render a single db entry (from tables.json) as compact text."""
    db_id = db_entry["db_id"]
    table_names: list[str] = db_entry["table_names_original"]
    columns: list[list[Any]] = db_entry["column_names_original"]
    column_types: list[str] = db_entry.get("column_types") or [""] * len(columns)
    fkeys: list[list[int]] = db_entry.get("foreign_keys") or []

    # Group columns by their owning table index; skip the special (-1, "*") row.
    grouped: dict[int, list[tuple[str, str]]] = {i: [] for i in range(len(table_names))}
    for (tbl_idx, col_name), col_type in zip(columns, column_types):
        if tbl_idx < 0:
            continue
        grouped[tbl_idx].append((col_name, col_type))

    lines: list[str] = [f"# Schema for {db_id}"]
    for tbl_idx, tbl_name in enumerate(table_names):
        col_strs = [f"{c}:{t}" if t else c for (c, t) in grouped.get(tbl_idx, [])]
        lines.append(f"Table {tbl_name}({', '.join(col_strs)})")

    for src, dst in fkeys:
        # columns is flattened: each entry has an index implicitly equal to its
        # position in the global column list; resolve table + column names.
        src_tbl_idx, src_col = columns[src]
        dst_tbl_idx, dst_col = columns[dst]
        if src_tbl_idx < 0 or dst_tbl_idx < 0:
            continue
        src_tbl = table_names[src_tbl_idx]
        dst_tbl = table_names[dst_tbl_idx]
        lines.append(f"FK: {src_tbl}.{src_col} -> {dst_tbl}.{dst_col}")

    text = "\n".join(lines)
    if max_chars is not None and len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text
