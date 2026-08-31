"""
Exploration nodes: plan probes, then run them.

Probes exist to ground literals. The pruned schema names columns but the model
cannot see what is stored in them, so a filter on a status or a category is a
guess until it looks. Probes run on the first attempt only; on a retry the
execution error is the stronger signal and costs nothing extra.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from src.agent.nodes.loop._shared import add_tokens, extract_code_block, schema_summary
from src.agent.nodes.loop.engine_adapter import get_adapter, probes_available
from src.agent.nodes.loop.probes import (
    INSPECT_VALUES,
    PROBE_TOOLS,
    ProbeRejected,
    build_probe_sql,
    summarize_rows,
    summarize_values,
)
from src.agent.state import AgentState
from src.config import get_settings
from src.feedback import PROBE, PROBE_REJECTED, observation
from src.llm import require_openai_client

logger = logging.getLogger(__name__)


def _plan_prompt(query: str, schema: str, budget: int) -> str:
    return (
        "You are about to write a SQL query. Before you do, you may inspect the data to "
        "learn the exact stored values, so your filters match.\n"
        f"Schema:\n{schema}\n"
        f"Question: {query}\n\n"
        f"Choose at most {budget} inspections, fewer is better. Choose none if the question "
        "needs no literal filter.\n"
        f"Tools: {INSPECT_VALUES} (distinct values of one column), sample_rows (first rows of one table).\n"
        "Reply with ONLY a ```json block containing a list, for example:\n"
        '[{"tool": "inspect_values", "table": "orders", "column": "status"}]\n'
        "Use [] for none."
    )


def _parse_plan(raw_text: str, budget: int) -> Optional[List[Dict[str, Any]]]:
    """Probe requests from the model's reply, or None if it did not produce a list."""
    block = extract_code_block(raw_text, ("json", ""))
    if block is None:
        return None
    try:
        parsed = json.loads(block)
    except ValueError:
        return None
    if not isinstance(parsed, list):
        return None

    requests = [entry for entry in parsed if isinstance(entry, dict) and entry.get("tool")]
    return requests[:budget]


def explorer_planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Open a loop iteration, and on the first one decide which probes to run.

    This is the only node that writes `loop_iterations`. Keeping the counter in a
    single sequential node is what makes the budget well-defined.
    """
    settings = get_settings()
    attempt = state.get("loop_iterations", 0) + 1
    update: Dict[str, Any] = {"loop_iterations": attempt, "probe_plan": []}

    schema_context = state.get("pruned_tables") or {}
    adapter = get_adapter(state.get("suggested_strategy"))

    skip_probes = (
        attempt > 1
        or not settings.schema_exploration_enabled
        or not probes_available(adapter, schema_context)
    )
    if skip_probes:
        return update

    prompt = _plan_prompt(
        state.get("query", ""),
        schema_summary(schema_context),
        settings.loop_tool_call_budget,
    )
    client = require_openai_client(settings)
    response = client.chat.completions.create(
        model=settings.critic_model or settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw_text = response.choices[0].message.content or ""
    completion_tokens = (
        response.usage.completion_tokens if response.usage else max(1, len(raw_text) // 4)
    )

    update["telemetry"] = add_tokens(
        state.get("telemetry"), max(1, len(prompt) // 4), completion_tokens
    )

    plan = _parse_plan(raw_text, settings.loop_tool_call_budget)
    if plan is None:
        logger.info("Probe plan was not a JSON list; continuing without probes.")
        return update

    update["probe_plan"] = plan
    return update


def has_probes(state: AgentState) -> str:
    """Route past the tool executor when no probes were requested."""
    return "probe" if state.get("probe_plan") else "generate"


def tool_executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Run the planned probes and record what the data actually contains.

    Every probe statement is written by `build_probe_sql` from the pruned
    schema's own identifiers. A name the model invented is rejected here and
    never reaches an engine.
    """
    schema_context = state.get("pruned_tables") or {}
    retained = schema_context.get("retained_columns") or {}
    adapter = get_adapter(state.get("suggested_strategy"))

    observations: List[Dict[str, Any]] = []
    executed = 0

    for request in state.get("probe_plan") or []:
        tool = str(request.get("tool") or "")
        try:
            sql, label = build_probe_sql(
                tool, str(request.get("table") or ""), request.get("column"), retained
            )
        except ProbeRejected as exc:
            observations.append(observation(PROBE_REJECTED, reason=str(exc)))
            continue

        columns, rows, error = adapter.execute(sql, schema_context)
        executed += 1

        if error:
            observations.append(observation(PROBE_REJECTED, reason=f"{label}: {error}"))
            continue

        result = (
            summarize_values(rows)
            if tool == INSPECT_VALUES
            else summarize_rows(columns, rows)
        )
        observations.append(observation(PROBE, label=label, result=result or "no values"))

    telemetry = dict(state.get("telemetry") or {})
    loop_telemetry = dict(telemetry.get("loop") or {})
    loop_telemetry["tool_calls"] = loop_telemetry.get("tool_calls", 0) + executed
    telemetry["loop"] = loop_telemetry

    return {"observations": observations, "probe_plan": [], "telemetry": telemetry}


__all__ = [
    "PROBE_TOOLS",
    "explorer_planner_node",
    "has_probes",
    "tool_executor_node",
]
