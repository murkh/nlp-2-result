"""Helpers shared by the structured loop nodes."""

from typing import Any, Dict, Optional


def schema_summary(schema_context: Dict[str, Any]) -> str:
    """Compact table/column listing with column roles."""
    retained = schema_context.get("retained_columns") or {}
    roles = schema_context.get("column_roles") or {}
    lines = []
    for table, cols in retained.items():
        rendered = ", ".join(f"{c} ({roles[c]})" if c in roles else c for c in cols)
        lines.append(f"- {table}: {rendered}")
    return "\n".join(lines)


def add_tokens(
    telemetry: Optional[Dict[str, Any]], prompt_tokens: int, completion_tokens: int
) -> Dict[str, Any]:
    """Roll a node's own LLM spend into the reported totals."""
    updated = dict(telemetry or {})
    updated["prompt_tokens"] = updated.get("prompt_tokens", 0) + prompt_tokens
    updated["completion_tokens"] = updated.get("completion_tokens", 0) + completion_tokens
    updated["total_tokens"] = updated.get("total_tokens", 0) + prompt_tokens + completion_tokens
    return updated
