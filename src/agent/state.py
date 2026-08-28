"""
LangGraph Multi-Agent State and Telemetry Schema Definitions.
"""

import operator
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

try:
    from langchain_core.messages import BaseMessage

    MessageItem = BaseMessage
except ImportError:
    MessageItem = Dict[str, Any]


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


class AgentState(TypedDict, total=False):
    """
    Complete state passed between LangGraph nodes during multi-agent conversational Q&A.
    """

    # Conversational Input & Session Context
    query: str
    session_id: str
    messages: Annotated[List[Any], operator.add]

    # Routing & Intent Classification
    intent: Optional[
        Literal["GREETING_OR_CHITCHAT", "AMBIGUOUS_QUERY", "STRUCTURED_QUERY", "UNSTRUCTURED_QUERY"]
    ]
    confidence: float
    routing_reason: str
    suggested_strategy: Optional[Literal["dedicated_db", "duckdb", "pandas_sandbox"]]

    # Schema & Dataset Context
    candidate_datasets: List[str]
    pruned_tables: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]

    # Execution Artifacts
    generated_code: Optional[str]
    execution_result: Optional[List[Dict[str, Any]]]
    execution_columns: Optional[List[str]]
    execution_error: Optional[str]

    # Synthesis & Outputs
    clarification_message: Optional[str]
    final_answer: Optional[str]
    citations: List[str]

    # Observability & Metrics
    telemetry: TelemetryData
    thinking_process: Optional[Dict[str, Any]]
