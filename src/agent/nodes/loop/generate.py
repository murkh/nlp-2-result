"""
Generation, validation and execution nodes.

Each is a separate node so the graph shows where a run failed and so the
validator can fast-fail before spending a sandbox subprocess.

The nodes call the Pandas sandbox engine's own public methods, so the guardrails
and execution path stay exactly the ones `/query/pandas-sandbox` uses.
"""

import logging
from typing import Any, Dict

from src.agent.nodes.loop._shared import add_tokens
from src.agent.state import AgentState
from src.database.connection import get_db_manager
from src.engines.pandas_sandbox.ast_validator import validate_python_code
from src.feedback import EXECUTION_ERROR, classify_error, observation

logger = logging.getLogger(__name__)


def get_sandbox_engine() -> Any:
    """The engine behind the loop. Built per call so routing costs no connection."""
    from src.engines.pandas_sandbox.engine import PandasSandboxEngine

    return PandasSandboxEngine(db_manager=get_db_manager())


def code_generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Write the candidate statement from the pruned schema plus observations.

    This is the entry point of every attempt -- the first pass and each retry --
    so it owns the loop counter the reflector spends.
    """
    engine = get_sandbox_engine()
    schema_context = state.get("pruned_tables") or {}
    attempt = (state.get("loop_iterations") or 0) + 1

    code, thought, tokens = engine.generate_code(
        state.get("query", ""),
        state.get("schema_ddl", ""),
        list(state.get("observations") or []),
        file_paths=schema_context.get("file_paths") or {},
    )
    code = engine.apply_dataset_loader(
        code,
        file_paths=schema_context.get("file_paths") or {},
        table_names=schema_context.get("table_names") or [],
    )

    return {
        "generated_code": code,
        "loop_iterations": attempt,
        "routing_reason": thought or state.get("routing_reason", ""),
        "execution_error": None,
        "telemetry": add_tokens(state.get("telemetry"), tokens[0], tokens[1]),
    }


def code_validator_node(state: AgentState) -> Dict[str, Any]:
    """Reject code the sandbox would refuse, without paying for the subprocess."""
    code = state.get("generated_code") or ""

    is_safe, violations = validate_python_code(code)
    if is_safe:
        return {"execution_error": None}

    # The validator reports a list; downstream error classification and the
    # failure message both need a single string.
    error = "; ".join(violations)
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
    engine = get_sandbox_engine()
    code = state.get("generated_code") or ""
    attempt = state.get("loop_iterations") or 1

    columns, rows, error = engine.execute_code(code)

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
