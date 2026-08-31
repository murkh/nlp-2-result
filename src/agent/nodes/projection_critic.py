"""
Projection Critic Node.

An id alone is not an answer an analyst can act on. This node checks whether the
generated SQL projects the columns needed to verify the result -- the entity's
human-readable identifier, the columns being compared, and the measures -- and
widens the SELECT list when it does not.

A deterministic gate runs first, so the node makes no LLM call and no second
database round trip on the common case where the projection is already adequate.
Every failure path falls through to the original result: the critic must never
turn a working answer into an error.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from src.agent.nodes.loop._shared import (
    SQL_BLOCK_LANGS,
    extract_code_block,
    schema_summary,
)
from src.agent.state import AgentState
from src.config import Settings, get_settings
from src.llm import require_openai_client

logger = logging.getLogger(__name__)

# Clause boundaries used to slice a SELECT list and a predicate region without a
# full SQL parse. Deliberately conservative: anything that does not match cleanly
# is treated as "not thin" and left alone.
_SELECT_RE = re.compile(r"^\s*SELECT\s+(?:DISTINCT\s+)?(.*?)\s+FROM\s", re.IGNORECASE | re.DOTALL)
_PREDICATE_RE = re.compile(
    r"\b(?:WHERE|HAVING|ORDER\s+BY)\b(.*?)(?=\b(?:GROUP\s+BY|ORDER\s+BY|LIMIT|OFFSET)\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_AGG_RE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)
_GROUP_BY_RE = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Reserved words that look like identifiers when scanning a predicate region.
_SQL_KEYWORDS = frozenset(
    {
        "and", "or", "not", "in", "is", "null", "like", "ilike", "between", "as",
        "asc", "desc", "case", "when", "then", "else", "end", "cast", "distinct",
        "true", "false", "on", "select", "from", "where", "having", "group", "order",
        "by", "limit", "offset", "join", "left", "right", "inner", "outer", "full",
        "union", "all", "exists", "any", "count", "sum", "avg", "min", "max", "coalesce",
    }
)

# Clauses the critic is not allowed to touch. A widened projection must return the
# same rows -- only more columns of them.
_IMMUTABLE_CLAUSE_RE = re.compile(
    r"\b(FROM|JOIN|WHERE|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET)\b(.*)$",
    re.IGNORECASE | re.DOTALL,
)


def _split_select_list(select_list: str) -> List[str]:
    """Split a SELECT list on top-level commas, ignoring commas inside parentheses."""
    items: List[str] = []
    depth = 0
    current = ""
    for ch in select_list:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        items.append(current.strip())
    return items


def _projected_identifiers(select_list: str) -> Set[str]:
    """Base column names appearing anywhere in the SELECT list, aliases included."""
    return {m.group(0).lower() for m in _IDENT_RE.finditer(select_list)} - _SQL_KEYWORDS


def _predicate_identifiers(sql: str) -> Set[str]:
    """Base column names referenced in WHERE / HAVING / ORDER BY."""
    found: Set[str] = set()
    for match in _PREDICATE_RE.finditer(sql):
        found |= {m.group(0).lower() for m in _IDENT_RE.finditer(match.group(1))}
    return found - _SQL_KEYWORDS


def _known_columns(schema_context: Dict[str, Any]) -> Set[str]:
    retained = schema_context.get("retained_columns") or {}
    return {c.lower() for cols in retained.values() for c in cols}


def is_projection_thin(
    sql: str, schema_context: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Decide, without an LLM, whether the SELECT list omits columns an analyst needs.

    Returns (thin, missing_columns). Conservative by design -- anything unparseable
    or already wide returns (False, []) so the node stays a free pass-through.
    """
    if not sql or not schema_context:
        return False, []

    select_match = _SELECT_RE.search(sql)
    if not select_match:
        return False, []

    select_list = select_match.group(1)
    if "*" in select_list:
        return False, []

    # A pure scalar aggregate is already the whole answer; widening it is wrong.
    if _AGG_RE.search(select_list) and not _GROUP_BY_RE.search(sql):
        if len(_split_select_list(select_list)) == 1:
            return False, []

    known = _known_columns(schema_context)
    if not known:
        return False, []

    projected = _projected_identifiers(select_list)
    wanted: Set[str] = _predicate_identifiers(sql) & known

    # Every entity whose id we project should come back with its display column.
    roles = {k.lower(): v for k, v in (schema_context.get("column_roles") or {}).items()}
    if projected:
        wanted |= {col for col, role in roles.items() if role == "display"}

    missing = sorted(c for c in wanted - projected if c in known)
    return bool(missing), missing


def _widen_sql(
    sql: str,
    missing: List[str],
    schema_summary: str,
    settings: Settings,
) -> Tuple[Optional[str], Tuple[int, int]]:
    """
    Ask the critic model to rewrite the SELECT list only. Returns (sql, tokens).

    Uses plain chat completions and ```sql block extraction -- the same contract the
    generators use -- so it works over any OpenAI-compatible gateway (bedrock-mantle)
    without depending on structured-output support.
    """
    client = require_openai_client(settings)
    model = settings.critic_model or settings.openai_model

    prompt = (
        "You are a SQL projection critic. The query below is correct but returns too "
        "few columns for an analyst to verify the answer.\n"
        f"Schema:\n{schema_summary}\n"
        f"SQL:\n{sql}\n"
        f"Missing columns an analyst needs: {', '.join(missing)}\n\n"
        "Rewrite the query adding those columns to the SELECT list.\n"
        "Rules:\n"
        "1. Change ONLY the SELECT list. Keep FROM, JOIN, WHERE, GROUP BY, HAVING, "
        "ORDER BY and LIMIT byte-for-byte identical.\n"
        "2. Keep every column that is already selected.\n"
        "3. If the query uses GROUP BY, add a column only if it is already grouped or "
        "you wrap it in an aggregate.\n"
        "4. Reply with ONLY a ```sql block containing the rewritten query."
    )
    prompt_tokens = max(1, len(prompt) // 4)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw_text = resp.choices[0].message.content or ""
    comp_tokens = resp.usage.completion_tokens if resp.usage else max(1, len(raw_text) // 4)

    widened = extract_code_block(raw_text, SQL_BLOCK_LANGS)
    if widened is None:
        logger.warning("Projection critic response contained no ```sql block.")
    return widened, (prompt_tokens, comp_tokens)


def _tail_clauses(sql: str) -> str:
    """Everything from the first FROM/JOIN/WHERE onward, normalized for comparison."""
    match = _IMMUTABLE_CLAUSE_RE.search(sql)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(0)).strip().rstrip(";").lower()


def is_widening_safe(original_sql: str, widened_sql: str) -> Tuple[bool, Optional[str]]:
    """
    Verify the rewrite only added columns: same row-selecting clauses, no dropped
    projections, and no forbidden statement smuggled in.
    """
    from src.engines.duckdb_engine import validate_duckdb_security

    if not widened_sql:
        return False, "empty rewrite"

    is_safe, sec_err = validate_duckdb_security(widened_sql)
    if not is_safe:
        return False, sec_err

    if _tail_clauses(original_sql) != _tail_clauses(widened_sql):
        return False, "rewrite altered a row-selecting clause"

    orig_select = _SELECT_RE.search(original_sql)
    new_select = _SELECT_RE.search(widened_sql)
    if not orig_select or not new_select:
        return False, "could not parse SELECT list"

    dropped = _projected_identifiers(orig_select.group(1)) - _projected_identifiers(
        new_select.group(1)
    )
    if dropped:
        return False, f"rewrite dropped projected column(s): {', '.join(sorted(dropped))}"

    return True, None


def projection_critic_node(state: AgentState) -> Dict[str, Any]:
    """
    Widen a thin SELECT list so the result carries the columns an analyst needs.

    Pass-through (no LLM, no re-execution) unless the deterministic gate fires.
    """
    settings = get_settings()
    sql = state.get("generated_code")
    schema_context = state.get("pruned_tables") or {}

    # Pass through: disabled, non-SQL strategy, upstream failure, or nothing returned.
    if not settings.projection_critic_enabled:
        return {}
    if state.get("suggested_strategy") == "pandas_sandbox":
        return {}
    if state.get("execution_error") or not sql:
        return {}
    if not state.get("execution_result"):
        return {}
    if not isinstance(schema_context, dict) or not schema_context.get("retained_columns"):
        return {}

    thin, missing = is_projection_thin(sql, schema_context)
    if not thin:
        return {}

    start = time.perf_counter()
    try:
        widened_sql, tokens = _widen_sql(
            sql, missing, schema_summary(schema_context), settings
        )
    except Exception as exc:
        # Includes LLMUnavailableError. A critic failure must never cost the user a
        # working answer, so every exception falls through to the original result.
        logger.warning("Projection critic LLM call failed, keeping original result: %s", exc)
        return {}

    if widened_sql is None:
        return {}

    safe, reason = is_widening_safe(sql, widened_sql)
    if not safe:
        logger.info("Projection critic rewrite rejected (%s), keeping original result.", reason)
        return {}

    columns, rows, exec_err = _re_execute(widened_sql, schema_context, state)
    if exec_err or not rows:
        logger.info(
            "Projection critic re-execution failed (%s), keeping original result.", exec_err
        )
        return {}

    latency_ms = (time.perf_counter() - start) * 1000.0
    telemetry = dict(state.get("telemetry") or {})
    # Roll the critic's own call into the reported totals: it is real spend.
    telemetry["prompt_tokens"] = telemetry.get("prompt_tokens", 0) + tokens[0]
    telemetry["completion_tokens"] = telemetry.get("completion_tokens", 0) + tokens[1]
    telemetry["total_tokens"] = telemetry.get("total_tokens", 0) + tokens[0] + tokens[1]
    telemetry["projection_critic"] = {
        "fired": True,
        "added_columns": missing,
        "latency_ms": round(latency_ms, 2),
        "model": settings.critic_model or settings.openai_model,
        "prompt_tokens": tokens[0],
        "completion_tokens": tokens[1],
    }

    return {
        "generated_code": widened_sql,
        "execution_result": rows,
        "execution_columns": columns,
        # The engine synthesized its one-line preview from the narrow result. Drop it
        # so the synthesizer rebuilds the answer over the widened columns.
        "final_answer": None,
        "telemetry": telemetry,
    }


def _re_execute(
    sql: str, schema_context: Dict[str, Any], state: AgentState
) -> Tuple[List[str], List[Dict[str, Any]], Optional[str]]:
    """Re-run the widened SQL on the engine that produced the original result."""
    strategy = state.get("suggested_strategy") or "duckdb"
    try:
        if strategy == "dedicated_db":
            from src.engines.dedicated_db import DedicatedDBEngine

            return DedicatedDBEngine().execute_sql(sql, schema_context)

        from src.engines.duckdb_engine import DuckDBQueryEngine

        return DuckDBQueryEngine().execute_sql(sql, schema_context)
    except Exception as exc:
        return [], [], str(exc)
