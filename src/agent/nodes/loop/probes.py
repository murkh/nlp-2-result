"""
Read-only schema probes for value grounding.

The model supplies identifiers only; every probe statement is written here from
the pruned schema's own canonical names, so no model-authored SQL reaches an
engine through this path. An identifier outside the pruned schema is rejected
before any statement is built.
"""

from typing import Any, Dict, List, Optional, Tuple

INSPECT_VALUES = "inspect_values"
SAMPLE_ROWS = "sample_rows"
PROBE_TOOLS = (INSPECT_VALUES, SAMPLE_ROWS)

DISTINCT_VALUE_LIMIT = 20
SAMPLE_ROW_LIMIT = 5


class ProbeRejected(ValueError):
    """A probe named a tool, table, or column that is not in the pruned schema."""


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _resolve_table(table: str, retained_columns: Dict[str, List[str]]) -> str:
    for name in retained_columns:
        if name.lower() == (table or "").lower():
            return name
    raise ProbeRejected(f"table '{table}' is not in the pruned schema")


def _resolve_column(column: str, table: str, retained_columns: Dict[str, List[str]]) -> str:
    for name in retained_columns.get(table, []):
        if name.lower() == (column or "").lower():
            return name
    raise ProbeRejected(f"column '{column}' is not in the pruned schema for table '{table}'")


def build_probe_sql(
    tool: str,
    table: str,
    column: Optional[str],
    retained_columns: Dict[str, List[str]],
) -> Tuple[str, str]:
    """
    Build the SQL for one probe. Returns (sql, label).

    Raises ProbeRejected for an unknown tool or an identifier outside the schema.
    """
    if tool not in PROBE_TOOLS:
        raise ProbeRejected(f"unknown probe tool '{tool}'")

    resolved_table = _resolve_table(table, retained_columns)

    if tool == SAMPLE_ROWS:
        sql = f"SELECT * FROM {_quote(resolved_table)} LIMIT {SAMPLE_ROW_LIMIT}"
        return sql, resolved_table

    resolved_column = _resolve_column(column, resolved_table, retained_columns)
    sql = (
        f"SELECT DISTINCT {_quote(resolved_column)} FROM {_quote(resolved_table)} "
        f"WHERE {_quote(resolved_column)} IS NOT NULL LIMIT {DISTINCT_VALUE_LIMIT}"
    )
    return sql, f"{resolved_table}.{resolved_column}"


def summarize_values(rows: List[Dict[str, Any]]) -> str:
    """Flatten distinct-value rows into a short comma-separated list."""
    values = [str(next(iter(row.values()), "")) for row in rows if row]
    return ", ".join(v for v in values if v)


def summarize_rows(columns: List[str], rows: List[Dict[str, Any]]) -> str:
    """Render sample rows compactly, one line per row."""
    if not rows:
        return "no rows"
    header = " | ".join(columns)
    lines = [" | ".join(str(row.get(c, "")) for c in columns) for row in rows]
    return "\n".join([header, *lines])
