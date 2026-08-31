"""
Supervisor Router Node for LangGraph State Machine.
Evaluates user query with token-efficient prompt and determines execution path.
"""

from typing import Any, Dict

from src.agent.state import AgentState
from src.database.connection import get_db_manager
from src.observability.telemetry import get_tracer
from src.routing import SemanticRouteIndex, SupervisorRouter

# The anchor index owns the embedding model and the pre-embedded global anchor
# matrix, so it is built once per process and shared. The router itself is cheap
# and rebuilt per call so it always sees the current database manager.
_semantic_index: SemanticRouteIndex = None


def _get_semantic_index() -> SemanticRouteIndex:
    global _semantic_index
    if _semantic_index is None:
        _semantic_index = SemanticRouteIndex()
    return _semantic_index


def supervisor_router_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor Router Node.
    Classifies incoming query into one of four intent branches:
    1. GREETING_OR_CHITCHAT
    2. AMBIGUOUS_QUERY
    3. STRUCTURED_QUERY
    4. UNSTRUCTURED_QUERY
    """
    query = state.get("query", "")
    session_id = state.get("session_id", "")
    tracer = get_tracer()

    with tracer.start_trace(name="supervisor_router_node", session_id=session_id) as trace:
        span = trace.start_span("classify_intent", input_data={"query": query})

        router = SupervisorRouter(
            db_manager=get_db_manager(), semantic_index=_get_semantic_index()
        )
        decision = router.classify_intent(query=query, session_id=session_id)

        # Only the LLM fallback tier spends tokens; the local tiers spend none.
        if decision.route_engine == "llm_fallback":
            prompt_tokens = max(10, len(query.split()) + 35)
            completion_tokens = 25
        else:
            prompt_tokens = 0
            completion_tokens = 0
        span.record_tokens(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        span.end(
            output_data=decision.model_dump() if hasattr(decision, "model_dump") else str(decision)
        )

        telemetry = state.get("telemetry", {}) or {}
        telemetry.update(
            {
                "trace_id": trace.trace_id,
                "route": decision.intent,
                "route_engine": decision.route_engine,
                "strategy_used": decision.suggested_strategy,
                "prompt_tokens": telemetry.get("prompt_tokens", 0) + prompt_tokens,
                "completion_tokens": telemetry.get("completion_tokens", 0) + completion_tokens,
                "total_tokens": telemetry.get("total_tokens", 0)
                + prompt_tokens
                + completion_tokens,
                "latency_ms": round(trace.latency_ms, 2),
                "execution_success": True,
            }
        )

        return {
            "intent": decision.intent,
            "confidence": decision.confidence,
            "routing_reason": decision.reasoning,
            "suggested_strategy": decision.suggested_strategy,
            "candidate_datasets": decision.relevant_datasets,
            "clarification_message": decision.clarification_question,
            "telemetry": telemetry,
        }
