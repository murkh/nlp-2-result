"""
Unstructured Sub-Agent Node for Hybrid Dense + Sparse Document RAG.
Retrieves chunks via Reciprocal Rank Fusion (RRF) and extracts citations.
"""

from typing import Any, Dict

from src.agent.state import AgentState
from src.api.schemas import QueryUnstructuredRAGRequest
from src.database.connection import get_db_manager
from src.engines.hybrid_rag import HybridRAGEngine
from src.observability.telemetry import get_tracer


def unstructured_node(state: AgentState) -> Dict[str, Any]:
    """
    Unstructured Sub-Agent Node:
    Executes hybrid dense vector + sparse search over document chunks.
    """
    query = state.get("query", "")
    session_id = state.get("session_id", "")
    db_manager = get_db_manager()
    tracer = get_tracer()

    with tracer.start_trace(name="unstructured_agent_hybrid_rag", session_id=session_id) as trace:
        span = trace.start_span("execute_hybrid_rag", input_data={"query": query})

        raw_answer = ""
        citations_list = []
        retrieved_chunks = []
        execution_error = None
        prompt_tokens = 0
        completion_tokens = 0

        try:
            engine = HybridRAGEngine(db_manager=db_manager)
            req = QueryUnstructuredRAGRequest(query=query, top_k=5)
            res = engine.execute_query(req)

            raw_answer = res.answer
            execution_error = res.error
            prompt_tokens = res.token_usage.prompt_tokens
            completion_tokens = res.token_usage.completion_tokens

            for c in res.citations:
                c_dict = c.model_dump() if hasattr(c, "model_dump") else (c.dict() if hasattr(c, "dict") else dict(c))
                citations_list.append(
                    f"[Doc: {c_dict.get('document_name')}, Page: {c_dict.get('page_number', 'N/A')}, Chunk: {c_dict.get('chunk_index')}]"
                )
                retrieved_chunks.append(c_dict)

            span.record_tokens(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            span.end(output_data={"chunks_count": len(retrieved_chunks), "error": execution_error})

        except Exception as exc:
            execution_error = str(exc)
            span.end(status="ERROR", error=execution_error)

        telemetry = state.get("telemetry", {}) or {}
        telemetry.update({
            "trace_id": trace.trace_id,
            "route": "UNSTRUCTURED_QUERY",
            "strategy_used": "hybrid_rag",
            "prompt_tokens": telemetry.get("prompt_tokens", 0) + prompt_tokens,
            "completion_tokens": telemetry.get("completion_tokens", 0) + completion_tokens,
            "total_tokens": telemetry.get("total_tokens", 0) + prompt_tokens + completion_tokens,
            "latency_ms": round(trace.latency_ms, 2),
            "execution_success": execution_error is None,
            "error": execution_error,
        })

        return {
            "retrieved_chunks": retrieved_chunks,
            "citations": citations_list,
            "final_answer": raw_answer if raw_answer else None,
            "execution_error": execution_error,
            "telemetry": telemetry,
        }
