"""
Generation, validation and execution nodes.

Each is a separate node so the graph shows where a run failed and so the
validator can fast-fail before spending a database round trip.
"""

import logging
from typing import Any, Dict

from src.agent.nodes.loop._shared import add_tokens
from src.agent.nodes.loop.engine_adapter import get_adapter
from src.agent.state import AgentState
from src.feedback import EXECUTION_ERROR, classify_error, observation

logger = logging.getLogger(__name__)


def code_generator_node(state: AgentState) -> Dict[str, Any]:
    """Write the candidate statement from the pruned schema plus observations."""
    adapter = get_adapter(state.get("suggested_strategy"))
    schema_context = state.get("pruned_tables") or {}

    code, thought, tokens = adapter.generate(
        state.get("query", ""),
        state.get("schema_ddl", ""),
        list(state.get("observations") or []),
        schema_context,
    )
    code = adapter.prepare(code, schema_context)

    return {
        "generated_code": code,
        "routing_reason": thought or state.get("routing_reason", ""),
        "execution_error": None,
        "telemetry": add_tokens(state.get("telemetry"), tokens[0], tokens[1]),
    }


def code_validator_node(state: AgentState) -> Dict[str, Any]:
    """Reject a statement the engine would refuse, without paying for the trip."""
    adapter = get_adapter(state.get("suggested_strategy"))
    code = state.get("generated_code") or ""

    is_safe, error = adapter.validate(code)
    if is_safe:
        return {"execution_error": None}

    attempt = state.get("loop_iterations") or 1
    return {
        "execution_error": error,
        "observations": [
            observation(
                EXECUTION_ERROR,
                attempt=attempt,
                error=error,
                correction_class="misc",
            )
        ],
    }


def is_valid(state: AgentState) -> str:
    """Skip execution when validation already failed."""
    return "reflect" if state.get("execution_error") else "execute"


def code_executor_node(state: AgentState) -> Dict[str, Any]:
    """Run the statement. Both the rows and the error are refinement signals."""
    adapter = get_adapter(state.get("suggested_strategy"))
    schema_context = state.get("pruned_tables") or {}
    code = state.get("generated_code") or ""
    attempt = state.get("loop_iterations") or 1

    columns, rows, error = adapter.execute(code, schema_context)

    if error:
        correction_class = classify_error(error)
        return {
            "execution_result": [],
            "execution_columns": [],
            "execution_error": error,
            "observations": [
                observation(
                    EXECUTION_ERROR,
                    attempt=attempt,
                    error=error,
                    correction_class=correction_class,
                )
            ],
        }

    return {
        "execution_result": rows,
        "execution_columns": columns,
        "execution_error": None,
    }
