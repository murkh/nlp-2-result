"""
Chitchat & Greeting Node.
Provides immediate conversational responses without database lookups or tool executions.
"""

from typing import Any, Dict

from src.agent.state import AgentState
from src.observability.telemetry import get_tracer


def chitchat_node(state: AgentState) -> Dict[str, Any]:
    """
    Chitchat Node: Handles greetings, identity, and general capability queries.
    Guaranteed zero tool calls and zero database lookups.
    """
    query = state.get("query", "").lower().strip()
    session_id = state.get("session_id", "")
    tracer = get_tracer()

    with tracer.start_trace(name="chitchat_node", session_id=session_id) as trace:
        span = trace.start_span("generate_chitchat_reply", input_data={"query": query})

        if any(w in query for w in ["who are you", "what are you", "what can you do", "capabilities", "help"]):
            answer = (
                "I am your Multi-Agent Knowledge Base Q&A Assistant. I can help you with:\n"
                "1. **Structured Data Analysis**: Query orders, sales, customer data using PostgreSQL, DuckDB, or Pandas.\n"
                "2. **Unstructured Document Search**: Retrieve and ground answers from policy handbooks, SOPs, and manuals.\n"
                "3. **Multi-Strategy Benchmarking**: Compare query performance and results across 3 processing engines.\n\n"
                "How can I assist you today?"
            )
        elif any(w in query for w in ["bye", "goodbye", "see you"]):
            answer = "Goodbye! Feel free to reach out anytime if you have more questions about your datasets."
        elif any(w in query for w in ["thank", "thanks"]):
            answer = "You're welcome! Let me know if you need anything else."
        else:
            answer = (
                "Hello! I am your Multi-Agent Knowledge Base assistant. "
                "You can ask me questions about your structured datasets (CSV/Parquet) "
                "or unstructured documents (PDF/Markdown). What would you like to explore?"
            )

        prompt_tokens = 15
        completion_tokens = 45
        span.record_tokens(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        span.end(output_data={"answer": answer})

        telemetry = state.get("telemetry", {}) or {}
        telemetry.update({
            "trace_id": trace.trace_id,
            "route": "GREETING_OR_CHITCHAT",
            "strategy_used": None,
            "prompt_tokens": telemetry.get("prompt_tokens", 0) + prompt_tokens,
            "completion_tokens": telemetry.get("completion_tokens", 0) + completion_tokens,
            "total_tokens": telemetry.get("total_tokens", 0) + prompt_tokens + completion_tokens,
            "latency_ms": round(trace.latency_ms, 2),
            "execution_success": True,
        })

        return {
            "final_answer": answer,
            "clarification_message": None,
            "generated_code": None,
            "execution_result": [],
            "citations": [],
            "telemetry": telemetry,
        }
