"""Deterministic extraction of structured sketches from SQL queries.

The sketch describes the skeleton of a query: which tables are used, what
joins tie them together, the logical role of each selected column, which
aggregations appear, and the structural shape of WHERE / GROUP BY / ORDER BY
clauses. Concrete literals are replaced by a placeholder so the sketch is
schema-aware but value-agnostic.

We use sqlglot's parser instead of regex because it handles subqueries,
aliasing, and nested expressions correctly.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import sqlglot
from sqlglot import expressions as exp


class SketchExtractionError(Exception):
    """Raised when a SQL string cannot be parsed into a sketch."""


_PLACEHOLDER = "?"
_AND = " AND "

_AGGS = {"SUM", "AVG", "MIN", "MAX", "COUNT"}


@dataclass
class Sketch:
    tables: list[str] = field(default_factory=list)
    joins: list[str] = field(default_factory=list)
    select: list[str] = field(default_factory=list)
    aggregations: list[str] = field(default_factory=list)
    where: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    having: list[str] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    order_by: list[str] = field(default_factory=list)
    limit: Optional[int] = None
    set_op: Optional[str] = None
    set_rhs: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _col_name(e: exp.Expression) -> str:
    """Render a column reference as `table.col` (or `col` when no table)."""
    if isinstance(e, exp.Column):
        table = e.table
        if not table and isinstance(e.this, exp.Identifier) and e.this.args.get("quoted"):
            return _PLACEHOLDER
        name = e.name
        return f"{table}.{name}" if table else name
    if isinstance(e, exp.Alias):
        return _col_name(e.this)
    if isinstance(e, exp.Star):
        return "*"
    return _expr_to_placeholder(e)


def _expr_to_placeholder(e: exp.Expression) -> str:
    """Render an expression with literal values replaced by a placeholder."""
    if isinstance(e, (exp.Literal, exp.Boolean, exp.Null)):
        return _PLACEHOLDER
    if isinstance(e, exp.Column):
        return _col_name(e)
    if isinstance(e, exp.Star):
        return "*"
    if isinstance(e, exp.Alias):
        return _expr_to_placeholder(e.this)
    if isinstance(e, exp.Func):
        return _render_func(e)
    if isinstance(e, exp.Paren):
        return f"({_expr_to_placeholder(e.this)})"
    if isinstance(e, exp.Distinct):
        inner = e.args.get("expressions") or []
        return "DISTINCT " + ", ".join(_expr_to_placeholder(x) for x in inner)
    if isinstance(e, exp.Binary):
        return _render_binary(e)
    if isinstance(e, exp.Not):
        return f"NOT {_expr_to_placeholder(e.this)}"
    if isinstance(e, exp.Between):
        col = _expr_to_placeholder(e.this)
        return f"{col} BETWEEN {_PLACEHOLDER} AND {_PLACEHOLDER}"
    if isinstance(e, exp.In):
        return _render_in(e)
    if isinstance(e, exp.Exists):
        return _render_exists(e)
    if isinstance(e, exp.Subquery):
        return _summarize_query(e)
    # Fallback: render via sqlglot but strip literals
    try:
        rendered = e.sql()
    except Exception:
        rendered = str(e)
    return rendered


def _render_func(e: exp.Func) -> str:
    name = e.sql_name().upper() if hasattr(e, "sql_name") else e.key.upper()
    args = [_expr_to_placeholder(a) for a in (e.args.get("expressions") or [])]
    if not args:
        inner = e.args.get("this")
        if inner is not None:
            args = [_expr_to_placeholder(inner)]
    return f"{name}({', '.join(args)})"


def _render_binary(e: exp.Binary) -> str:
    left = _expr_to_placeholder(e.this)
    right = _expr_to_placeholder(e.expression)
    op = e.key.upper() if e.key else type(e).__name__.upper()
    op_map = {
        "EQ": "=", "NEQ": "!=", "GT": ">", "LT": "<", "GTE": ">=", "LTE": "<=",
        "LIKE": "LIKE", "ILIKE": "ILIKE", "IN": "IN", "IS": "IS", "AND": "AND", "OR": "OR",
        "ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/",
    }
    op_str = op_map.get(op, op)
    return f"{left} {op_str} {right}"


def _render_in(e: exp.In) -> str:
    col = _expr_to_placeholder(e.this)
    query = e.args.get("query")
    if query is not None:
        return f"{col} IN ({_summarize_query(query)})"
    return f"{col} IN ({_PLACEHOLDER})"


def _render_exists(e: exp.Exists) -> str:
    query = e.this
    if query is not None:
        return f"EXISTS ({_summarize_query(query)})"
    return "EXISTS (SUBQUERY)"


def _collect_tables(query: exp.Expression) -> list[str]:
    seen: list[str] = []
    for t in query.find_all(exp.Table):
        name = t.name
        if name and name not in seen:
            seen.append(name)
    return seen


def _collect_joins(query: exp.Expression) -> list[str]:
    joins: list[str] = []
    for j in query.find_all(exp.Join):
        on = j.args.get("on")
        if on is not None:
            joins.append(_expr_to_placeholder(on))
        else:
            side = j.args.get("side") or ""
            kind = j.args.get("kind") or "JOIN"
            t = j.this
            tname = t.name if isinstance(t, exp.Table) else _expr_to_placeholder(t)
            joins.append(f"{side} {kind} {tname}".strip())
    return joins


def _collect_select(query: exp.Select) -> tuple[list[str], list[str]]:
    selects: list[str] = []
    aggs: list[str] = []
    for e in query.expressions:
        rendered = _expr_to_placeholder(e)
        selects.append(rendered)
        for f in e.find_all(exp.Func):
            name = f.sql_name().upper() if hasattr(f, "sql_name") else f.key.upper()
            if name in _AGGS and name not in aggs:
                aggs.append(name)
    if selects and query.args.get("distinct") is not None:
        selects[0] = f"DISTINCT {selects[0]}"
    return selects, aggs


def _summarize_query(query: exp.Expression) -> str:
    """Compactly render a nested SELECT without concrete literal values."""
    if isinstance(query, exp.Subquery):
        query = query.this
    if isinstance(query, (exp.Union, exp.Intersect, exp.Except)):
        left = _summarize_query(query.this)
        right = _summarize_query(query.expression)
        return f"{left} SET_OP {_set_op_name(query)} SET_RHS {right}"
    if not isinstance(query, exp.Select):
        inner = query.find(exp.Select)
        if inner is None:
            return "SUBQUERY"
        query = inner

    selects, _ = _collect_select(query)
    parts = ["SELECT " + (", ".join(selects) if selects else "-")]
    tables = _collect_tables(query)
    if tables:
        parts.append("FROM " + ", ".join(tables))
    joins = _collect_joins(query)
    if joins:
        parts.append("JOINS " + " | ".join(joins))
    where = _collect_where(query)
    if where:
        parts.append("WHERE " + _AND.join(where))
    group_by = _collect_group(query)
    if group_by:
        parts.append("GROUP_BY " + ", ".join(group_by))
    having = _collect_having(query)
    if having:
        parts.append("HAVING " + _AND.join(having))
    order_by = _collect_order(query)
    if order_by:
        parts.append("ORDER_BY " + ", ".join(order_by))
    limit = _collect_limit(query)
    if limit is not None:
        parts.append(f"LIMIT {limit}")
    return " ".join(parts)


def _set_op_name(query: exp.Expression) -> str:
    if isinstance(query, exp.Union):
        return "UNION_ALL" if query.args.get("distinct") is False else "UNION"
    if isinstance(query, exp.Intersect):
        return "INTERSECT"
    if isinstance(query, exp.Except):
        return "EXCEPT"
    return type(query).__name__.upper()


def _collect_where(query: exp.Select) -> list[str]:
    where = query.args.get("where")
    if where is None:
        return []
    return _split_conjunction(where.this)


def _split_conjunction(e: exp.Expression) -> list[str]:
    if isinstance(e, exp.And):
        return _split_conjunction(e.this) + _split_conjunction(e.expression)
    return [_expr_to_placeholder(e)]


def _collect_group(query: exp.Select) -> list[str]:
    g = query.args.get("group")
    if g is None:
        return []
    return [_expr_to_placeholder(x) for x in g.expressions]


def _collect_having(query: exp.Select) -> list[str]:
    h = query.args.get("having")
    if h is None:
        return []
    return _split_conjunction(h.this)


def _collect_subqueries(query: exp.Select) -> list[str]:
    seen: list[str] = []
    for subquery in query.find_all(exp.Subquery):
        rendered = _summarize_query(subquery)
        if rendered and rendered not in seen:
            seen.append(rendered)
    return seen


def _collect_order(query: exp.Select) -> list[str]:
    o = query.args.get("order")
    if o is None:
        return []
    out: list[str] = []
    for item in o.expressions:
        direction = "DESC" if item.args.get("desc") else "ASC"
        out.append(f"{_expr_to_placeholder(item.this)} {direction}")
    return out


def _collect_limit(query: exp.Select) -> Optional[int]:
    lim = query.args.get("limit")
    if lim is None:
        return None
    val = lim.expression if hasattr(lim, "expression") else lim.this
    if isinstance(val, exp.Literal):
        try:
            return int(val.this)
        except (TypeError, ValueError):
            return None
    return None


def extract_sketch(sql: str, dialect: str = "sqlite") -> Sketch:
    """Parse ``sql`` and return a deterministic structured sketch."""
    try:
        trees = sqlglot.parse(sql, read=dialect)
    except Exception as exc:
        raise SketchExtractionError(f"parse error: {exc}") from exc

    trees = [t for t in trees if t is not None]
    if not trees:
        raise SketchExtractionError("empty parse tree")

    tree = trees[0]

    set_op: Optional[str] = None
    set_rhs: Optional[str] = None
    if isinstance(tree, (exp.Union, exp.Intersect, exp.Except)):
        set_op = _set_op_name(tree)
        set_rhs = _summarize_query(tree.expression)
        tree = tree.this

    if not isinstance(tree, exp.Select):
        inner = tree.find(exp.Select)
        if inner is None:
            raise SketchExtractionError(f"unsupported top-level expression: {type(tree).__name__}")
        tree = inner

    selects, aggs = _collect_select(tree)
    sketch = Sketch(
        tables=_collect_tables(tree),
        joins=_collect_joins(tree),
        select=selects,
        aggregations=aggs,
        where=_collect_where(tree),
        group_by=_collect_group(tree),
        having=_collect_having(tree),
        subqueries=_collect_subqueries(tree),
        order_by=_collect_order(tree),
        limit=_collect_limit(tree),
        set_op=set_op,
        set_rhs=set_rhs,
    )
    return sketch


def sketch_to_text(sketch: Sketch) -> str:
    """Render a sketch as a deterministic multi-line string for training targets."""
    lines: list[str] = []
    lines.append("TABLES: " + (", ".join(sketch.tables) if sketch.tables else "-"))
    if sketch.joins:
        lines.append("JOINS: " + " | ".join(sketch.joins))
    lines.append("SELECT: " + (", ".join(sketch.select) if sketch.select else "-"))
    if sketch.aggregations:
        lines.append("AGGREGATIONS: " + ", ".join(sketch.aggregations))
    if sketch.where:
        lines.append("WHERE: " + _AND.join(sketch.where))
    if sketch.group_by:
        lines.append("GROUP_BY: " + ", ".join(sketch.group_by))
    if sketch.having:
        lines.append("HAVING: " + _AND.join(sketch.having))
    if sketch.subqueries:
        lines.append("SUBQUERIES: " + " | ".join(sketch.subqueries))
    if sketch.order_by:
        lines.append("ORDER_BY: " + ", ".join(sketch.order_by))
    if sketch.limit is not None:
        lines.append(f"LIMIT: {sketch.limit}")
    if sketch.set_op:
        lines.append(f"SET_OP: {sketch.set_op}")
    if sketch.set_rhs:
        lines.append(f"SET_RHS: {sketch.set_rhs}")
    return "\n".join(lines)
