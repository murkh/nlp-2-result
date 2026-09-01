"""
LangGraph Multi-Agent State and Telemetry Schema Definitions.
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict, Union

from langchain_core.messages import BaseMessage

# The only structured engine. Reported in telemetry so runs stay comparable.
STRATEGY_LABEL = "pandas_sandbox"


class TelemetryData(TypedDict, total=False):
    """Execution telemetry recorded across graph nodes."""

    trace_id: str
    route: str
    strategy_used: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    execution_success: bool
    error: Optional[str]
    loop: Dict[str, Any]


class AgentState(TypedDict, total=False):
    """
    Complete state passed between LangGraph nodes during multi-agent conversational Q&A.
    """

    # Conversational Input & Session Context
    query: str
    session_id: str
    messages: Annotated[List[BaseMessage], operator.add]

    # Routing & Intent Classification
    intent: Optional[
        Literal["GREETING_OR_CHITCHAT", "AMBIGUOUS_QUERY", "STRUCTURED_QUERY", "UNSTRUCTURED_QUERY"]
    ]
    confidence: float
    routing_reason: str

    # Schema & Dataset Context
    candidate_datasets: List[str]
    # Pruned schema slice (SchemaContextRef) the loop reasons over.
    pruned_tables: Dict[str, Any]
    # DDL prompt snippet for the generators. Kept out of pruned_tables so engine
    # API responses stay free of DDL text.
    schema_ddl: str
    retrieved_chunks: List[Dict[str, Any]]

    # Execution Artifacts
    generated_code: Optional[str]
    execution_result: Optional[List[Dict[str, Any]]]
    execution_columns: Optional[List[str]]
    execution_error: Optional[str]

    # Self-Correction Loop
    # Accumulates across iterations. Order is significant for prompt
    # reproducibility, so only sequential nodes may append.
    observations: Annotated[List[Dict[str, Any]], operator.add]
    # Written by code_generator only (plus the reset in schema_retriever): a
    # single sequential writer is what makes the retry budget well-defined.
    loop_iterations: int
    reflection_class: Optional[str]

    # Synthesis & Outputs
    clarification_message: Optional[str]
    final_answer: Optional[str]
    citations: List[str]

    # Observability & Metrics
    telemetry: TelemetryData
    thinking_process: Optional[Dict[str, Any]]
