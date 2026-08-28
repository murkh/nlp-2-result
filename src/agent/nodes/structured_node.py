"""
Structured Sub-Agent Node for SQL & DataFrame Query Execution.
Dispatches queries to Strategy A (PostgreSQL), Strategy B (DuckDB), or Strategy C (Pandas Sandbox).
"""

from typing import Any, Dict

from src.agent.state import AgentState
from src.api.schemas import (
    QueryDedicatedDBRequest,
    QueryDuckDBRequest,
    QueryPandasSandboxRequest,
)
from src.database.connection import get_db_manager
from src.engines.dedicated_db import DedicatedDBEngine
from src.engines.duckdb_engine import DuckDBQueryEngine
from src.engines.pandas_sandbox.engine import PandasSandboxEngine
from src.observability.telemetry import get_tracer


def structured_node(state: AgentState) -> Dict[str, Any]:
    """
    Structured Sub-Agent Node:
    Executes natural language queries against structured data using selected strategy.
    """
    query = state.get("query", "")
    session_id = state.get("session_id", "")
    strategy = state.get("suggested_strategy") or "duckdb"
    db_manager = get_db_manager()
    tracer = get_tracer()

    with tracer.start_trace(name=f"structured_agent_{strategy}", session_id=session_id) as trace:
        span = trace.start_span(
            "execute_structured_engine", input_data={"query": query, "strategy": strategy}
        )

        generated_code = ""
        execution_result = []
        execution_columns = []
        execution_error = None
        prompt_tokens = 0
        completion_tokens = 0
        raw_answer = ""

        try:
            if strategy == "dedicated_db":
                engine = DedicatedDBEngine(db_manager=db_manager)
                req = QueryDedicatedDBRequest(query=query)
                res = engine.execute_query(req)
                generated_code = res.sql_query
                execution_result = res.tabular_result.rows
                execution_columns = res.tabular_result.columns
                raw_answer = res.answer
                execution_error = res.error
                prompt_tokens = res.token_usage.prompt_tokens
                completion_tokens = res.token_usage.completion_tokens

            elif strategy == "pandas_sandbox":
                engine = PandasSandboxEngine(db_manager=db_manager)
                req = QueryPandasSandboxRequest(query=query)
                res = engine.execute_query(req)
                generated_code = res.python_code
                execution_result = res.tabular_result.rows
                execution_columns = res.tabular_result.columns
                raw_answer = res.answer
                execution_error = res.error
                prompt_tokens = res.token_usage.prompt_tokens
                completion_tokens = res.token_usage.completion_tokens

            else:  # default Strategy B: duckdb
                strategy = "duckdb"
                engine = DuckDBQueryEngine(db_manager=db_manager)
                req = QueryDuckDBRequest(query=query)
                res = engine.execute_query(req)
                generated_code = res.sql_query
                execution_result = res.tabular_result.rows
                execution_columns = res.tabular_result.columns
                raw_answer = res.answer
                execution_error = res.error
                prompt_tokens = res.token_usage.prompt_tokens
                completion_tokens = res.token_usage.completion_tokens

            span.record_tokens(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            span.end(output_data={"rows_count": len(execution_result), "error": execution_error})

        except Exception as exc:
            execution_error = str(exc)
            span.end(status="ERROR", error=execution_error)

        telemetry = state.get("telemetry", {}) or {}
        telemetry.update(
            {
                "trace_id": trace.trace_id,
                "route": "STRUCTURED_QUERY",
                "strategy_used": strategy,
                "prompt_tokens": telemetry.get("prompt_tokens", 0) + prompt_tokens,
                "completion_tokens": telemetry.get("completion_tokens", 0) + completion_tokens,
                "total_tokens": telemetry.get("total_tokens", 0)
                + prompt_tokens
                + completion_tokens,
                "latency_ms": round(trace.latency_ms, 2),
                "execution_success": execution_error is None,
                "error": execution_error,
            }
        )

        return {
            "suggested_strategy": strategy,
            "generated_code": generated_code,
            "execution_result": execution_result,
            "execution_columns": execution_columns,
            "execution_error": execution_error,
            "final_answer": raw_answer if raw_answer else None,
            "telemetry": telemetry,
        }
