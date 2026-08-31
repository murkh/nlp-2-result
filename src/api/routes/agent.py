"""
Conversational Multi-Agent Supervisor Q&A API Route.
Provides the POST /query/agent endpoint executing the LangGraph multi-agent workflow.
"""

import uuid
from typing import Any, Dict, List

from fastapi import APIRouter


from src.agent.graph import run_agent
from src.api.schemas import (
    DecisionStep,
    ExecutionMetrics,
    QueryAgentRequest,
    QueryAgentResponse,
    TabularResult,
    ThinkingProcess,
    TokenUsage,
)
from src.llm import LLMUnavailableError


router = APIRouter(prefix="/query", tags=["Agent Orchestration"])


def _loop_steps(loop: Dict[str, Any], offset: int) -> List[DecisionStep]:
    """One decision step per correction attempt, plus one for the probes."""
    if not loop:
        return []

    steps: List[DecisionStep] = []
    tool_calls = loop.get("tool_calls") or 0
    if tool_calls:
        steps.append(
            DecisionStep(
                step_number=offset + len(steps) + 1,
                title="Schema Value Grounding",
                choice=f"Inspected the data {tool_calls} time(s)",
                reasoning=(
                    "Read the stored values before writing the query, so filters match "
                    "the real spelling and casing instead of a guess."
                ),
                details={"tool_calls": tool_calls},
            )
        )

    attempts = loop.get("attempts") or []
    for record in attempts[1:]:
        steps.append(
            DecisionStep(
                step_number=offset + len(steps) + 1,
                title=f"Self-Correction Attempt {record.get('attempt')}",
                choice=f"Outcome: {record.get('outcome')}",
                reasoning=(
                    "The previous attempt did not produce a usable result, so the error "
                    "was fed back and the statement regenerated."
                ),
                details=record,
            )
        )

    if loop.get("escalated"):
        steps.append(
            DecisionStep(
                step_number=offset + len(steps) + 1,
                title="Escalated to the User",
                choice="Retry budget spent, no answer invented",
                reasoning=(
                    "Every attempt failed to execute, so the query was handed back with "
                    "what was tried and what the data contains."
                ),
                details={"iterations": loop.get("iterations")},
            )
        )

    return steps


@router.post("/agent", response_model=QueryAgentResponse)
async def query_agent_endpoint(request: QueryAgentRequest) -> QueryAgentResponse:
    """
    Conversational LangGraph Multi-Agent Supervisor Q&A Endpoint.
    Routes queries between Greeting/Chitchat, Ambiguous Clarification,
    Structured SQL/DataFrame Engines, and Unstructured Hybrid RAG with Synthesis.
    """
    session_id = request.session_id or str(uuid.uuid4())
    try:
        state = run_agent(
            query=request.query,
            session_id=session_id,
            suggested_strategy=request.suggested_strategy,
            dataset_ids=request.dataset_ids,
        )
    except LLMUnavailableError as exc:
        return QueryAgentResponse(
            query=request.query,
            session_id=session_id,
            intent="ERROR",
            answer=f"Agent routing failed: {exc}",
            error=str(exc),
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

    intent = state.get("intent", "GREETING_OR_CHITCHAT")
    confidence = state.get("confidence", 0.95)
    routing_reason = state.get("routing_reason", "")
    suggested_strategy = state.get("suggested_strategy")
    candidate_datasets = state.get("candidate_datasets", [])
    gen_code = state.get("generated_code")
    citations = state.get("citations", [])

    steps = [
        DecisionStep(
            step_number=1,
            title="Supervisor Intent Classification",
            choice=f"Intent: {intent} ({confidence:.0%})",
            reasoning=routing_reason
            or "Classified user query based on semantic patterns and metadata catalog.",
            details={"intent": intent, "confidence": confidence},
        )
    ]

    if intent == "GREETING_OR_CHITCHAT":
        steps.append(
            DecisionStep(
                step_number=2,
                title="Direct Conversational Bypass",
                choice="0 Tool Calls & 0 DB Queries",
                reasoning="Short-circuited pipeline immediately with conversational greeting to achieve 100% token efficiency.",
            )
        )
    elif intent == "AMBIGUOUS_QUERY":
        cand_str = (
            ", ".join(candidate_datasets)
            if candidate_datasets
            else "Available knowledge datasets"
        )
        steps.append(
            DecisionStep(
                step_number=2,
                title="Proactive Ambiguity Detection",
                choice=f"Suggested Datasets: {cand_str}",
                reasoning="Identified underspecified query. Prevented hallucinated SQL/code generation by formulating targeted clarification questions.",
                details={"candidate_datasets": candidate_datasets},
            )
        )
    elif intent == "STRUCTURED_QUERY":
        strat_name = (
            "DuckDB (Strategy B)"
            if suggested_strategy == "duckdb"
            else (
                "PostgreSQL (Strategy A)"
                if suggested_strategy == "dedicated_db"
                else "Pandas Sandbox (Strategy C)"
            )
        )
        steps.append(
            DecisionStep(
                step_number=2,
                title="Engine Dispatch & Strategy Selection",
                choice=f"Dispatched to {strat_name}",
                reasoning=f"Selected {strat_name} to execute structured aggregation and filtering over tabular datasets.",
                details={"strategy": suggested_strategy},
            )
        )
        if gen_code:
            lang = "Python" if suggested_strategy == "pandas_sandbox" else "SQL"
            steps.append(
                DecisionStep(
                    step_number=3,
                    title=f"{lang} Query & Code Formulation",
                    choice=f"Generated {lang} with LIMIT 20 safety ceiling",
                    reasoning="Synthesized executable transformation code adhering to schema column definitions.",
                    details={"code": gen_code},
                )
            )
        steps.extend(_loop_steps(telemetry.get("loop") or {}, len(steps)))
        steps.append(
            DecisionStep(
                step_number=len(steps) + 1,
                title="Execution & Result Extraction",
                choice=f"Retrieved {len(exec_rows)} record(s)",
                reasoning="Executed query inside sandbox/engine and extracted structured records without mutations.",
            )
        )
        critic = (telemetry or {}).get("projection_critic")
        if critic:
            added = ", ".join(critic.get("added_columns", []))
            steps.append(
                DecisionStep(
                    step_number=len(steps) + 1,
                    title="Projection Critic",
                    choice=f"Widened projection with: {added}",
                    reasoning=(
                        "The initial query returned too few columns to verify the answer. "
                        "Added the human-readable identifier, the compared columns, and the "
                        "measures, then re-executed."
                    ),
                    details=critic,
                )
            )
    elif intent == "UNSTRUCTURED_QUERY":
        steps.append(
            DecisionStep(
                step_number=2,
                title="Hybrid Dense + Sparse RAG Dispatch",
                choice="Hybrid RRF (pgvector HNSW + tsvector BM25)",
                reasoning="Dispatched to Unstructured Sub-Agent for dual semantic vector and keyword search.",
            )
        )
        steps.append(
            DecisionStep(
                step_number=3,
                title="Citation Extraction & Evidence Grounding",
                choice=f"Retrieved {len(citations)} citation(s)",
                reasoning="Ranked excerpts using Reciprocal Rank Fusion (k=60) and grounded response strictly in source excerpts.",
                details={"citations": citations},
            )
        )

    steps.append(
        DecisionStep(
            step_number=len(steps) + 1,
            title="Evidence Synthesis & Formatting",
            choice="Final Natural Language Answer with Evidence",
            reasoning="Formatted output with tabular evidence, bracketed citations, and execution telemetry metrics.",
        )
    )

    thinking_process = ThinkingProcess(
        summary=f"LangGraph Supervisor routed query as [{intent}] with {confidence:.0%} confidence.",
        steps=steps,
    )

    return QueryAgentResponse(
        query=state.get("query", request.query),
        session_id=session_id,
        intent=intent,
        confidence=confidence,
        routing_reason=routing_reason,
        suggested_strategy=suggested_strategy,
        answer=state.get("final_answer") or "",
        generated_code=gen_code,
        tabular_result=tabular_res,
        citations=citations,
        clarification_message=state.get("clarification_message"),
        candidate_datasets=candidate_datasets,
        thinking_process=thinking_process,
        metrics=metrics,
        token_usage=token_usage,
        telemetry=telemetry,
        error=state.get("execution_error"),
    )
