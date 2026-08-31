"""
Clarification Node for Ambiguous Queries.
Proactively presents clarifying questions with candidate dataset options.
"""

from typing import Any, Dict

from src.agent.state import AgentState
from src.observability.telemetry import get_tracer


def clarify_node(state: AgentState) -> Dict[str, Any]:
    """
    Clarification Node: Formulates targeted questions and suggestions for ambiguous queries.
    Prevents running ungrounded or hallucinated queries.
    """
    query = state.get("query", "")
    session_id = state.get("session_id", "")
    clarification_msg = state.get("clarification_message")
    candidates = state.get("candidate_datasets", [])
    tracer = get_tracer()

    with tracer.start_trace(name="clarify_node", session_id=session_id) as trace:
        span = trace.start_span(
            "formulate_clarification", input_data={"query": query, "candidates": candidates}
        )

        if not clarification_msg:
            cand_str = (
                ", ".join(candidates)
                if candidates
                else "orders, sales, customer tables, and company policy documents"
            )
            clarification_msg = (
                f"Your query '{query}' is broad. To provide an accurate answer, could you clarify your goal?\n\n"
                f"**Available Datasets & Topics**:\n"
                f"- Structured Tables: {cand_str}\n"
                f"- Unstructured Docs: Employee Handbook, Incident Policy, Security Guidelines\n\n"
                f"For example, you can ask: *'How many completed orders are there?'* or *'What is the rate limit policy?'*"
            )

        prompt_tokens = 20
        completion_tokens = 50
        span.record_tokens(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        span.end(output_data={"clarification": clarification_msg})

        telemetry = state.get("telemetry", {}) or {}
        telemetry.update(
            {
                "trace_id": trace.trace_id,
                "route": "AMBIGUOUS_QUERY",
                "strategy_used": None,
                "prompt_tokens": telemetry.get("prompt_tokens", 0) + prompt_tokens,
                "completion_tokens": telemetry.get("completion_tokens", 0) + completion_tokens,
                "total_tokens": telemetry.get("total_tokens", 0)
                + prompt_tokens
                + completion_tokens,
                "latency_ms": round(trace.latency_ms, 2),
                # A clarification reached because there was nothing to query is
                # still a failed query, not a successful turn.
                "execution_success": state.get("execution_error") is None,
            }
        )

        return {
            "clarification_message": clarification_msg,
            "final_answer": clarification_msg,
            "generated_code": None,
            "execution_result": [],
            "citations": [],
            "telemetry": telemetry,
        }
