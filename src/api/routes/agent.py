"""
Conversational Multi-Agent Supervisor Q&A API Route.
Provides the POST /query/agent endpoint executing the LangGraph multi-agent workflow.
"""

from typing import Any, Dict, Optional
import uuid

from src.agent.graph import run_agent
from src.api.schemas import (
    ExecutionMetrics,
    QueryAgentRequest,
    QueryAgentResponse,
    TabularResult,
    TokenUsage,
)

try:
    from fastapi import APIRouter
except ImportError:
    class APIRouter:
        def __init__(self, *args, **kwargs):
            self.routes = []

        def post(self, path, *args, **kwargs):
            def decorator(func):
                self.routes.append(("POST", path, func))
                return func
            return decorator

        def get(self, path, *args, **kwargs):
            def decorator(func):
                self.routes.append(("GET", path, func))
                return func
            return decorator


router = APIRouter(prefix="/query", tags=["Agent Orchestration"])


@router.post("/agent", response_model=QueryAgentResponse)
async def query_agent_endpoint(request: QueryAgentRequest) -> QueryAgentResponse:
    """
    Conversational LangGraph Multi-Agent Supervisor Q&A Endpoint.
    Routes queries between Greeting/Chitchat, Ambiguous Clarification,
    Structured SQL/DataFrame Engines, and Unstructured Hybrid RAG with Synthesis.
    """
    session_id = request.session_id or str(uuid.uuid4())
    state = run_agent(
        query=request.query,
        session_id=session_id,
        suggested_strategy=request.suggested_strategy,
        dataset_ids=request.dataset_ids,
    )

    telemetry = state.get("telemetry", {}) or {}
    exec_cols = state.get("execution_columns") or []
    exec_rows = state.get("execution_result") or []
    tabular_res = TabularResult(columns=exec_cols, rows=exec_rows)

    prompt_toks = telemetry.get("prompt_tokens", 0)
    compl_toks = telemetry.get("completion_tokens", 0)
    total_toks = telemetry.get("total_tokens", prompt_toks + compl_toks)
    lat_ms = telemetry.get("latency_ms", 0.0)

    token_usage = TokenUsage(
        prompt_tokens=prompt_toks,
        completion_tokens=compl_toks,
        total_tokens=total_toks,
    )
    metrics = ExecutionMetrics(total_latency_ms=lat_ms)

    return QueryAgentResponse(
        query=state.get("query", request.query),
        session_id=session_id,
        intent=state.get("intent", "GREETING_OR_CHITCHAT"),
        confidence=state.get("confidence", 0.95),
        routing_reason=state.get("routing_reason", ""),
        suggested_strategy=state.get("suggested_strategy"),
        answer=state.get("final_answer") or "",
        generated_code=state.get("generated_code"),
        tabular_result=tabular_res,
        citations=state.get("citations", []),
        clarification_message=state.get("clarification_message"),
        candidate_datasets=state.get("candidate_datasets", []),
        metrics=metrics,
        token_usage=token_usage,
        telemetry=telemetry,
        error=state.get("execution_error"),
    )
