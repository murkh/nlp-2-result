"""
Strict schemas for the Supervisor Intent Classifier and Router.

Four intent routes:
- GREETING_OR_CHITCHAT: direct conversational reply, zero database lookups.
- AMBIGUOUS_QUERY: proactive clarification with candidate dataset suggestions.
- STRUCTURED_QUERY: sandboxed Python/Pandas DataFrame execution.
- UNSTRUCTURED_QUERY: hybrid dense+sparse document RAG.
"""

from typing import List, Literal, Optional

from src.api.schemas import BaseModel, Field

IntentType = Literal[
    "GREETING_OR_CHITCHAT",
    "AMBIGUOUS_QUERY",
    "STRUCTURED_QUERY",
    "UNSTRUCTURED_QUERY",
]

RouteEngineType = Literal["semantic_fastpath", "llm_fallback", "heuristic_guardrail"]


class SupervisorDecision(BaseModel):
    """Structured decision output from the supervisor router."""

    intent: IntentType
    confidence: float = Field(default=0.95, ge=0.0, le=1.0, description="Routing confidence score")
    reasoning: str = Field(default="", description="Explanation of routing classification")
    route_engine: RouteEngineType = Field(
        default="semantic_fastpath", description="Which routing tier produced this decision"
    )
    relevant_datasets: List[str] = Field(
        default_factory=list, description="Target dataset names/IDs if identified"
    )
    clarification_question: Optional[str] = Field(
        default=None, description="Proactive clarification question for ambiguous queries"
    )


class TenantCatalog(BaseModel):
    """Per-tenant registry of queryable assets used to synthesize routing anchors."""

    org_id: str
    structured_tables: List[str] = Field(default_factory=list)
    unstructured_docs: List[str] = Field(default_factory=list)


class LLMIntentDecision(BaseModel):
    """
    Flat schema handed to OpenAI structured outputs.

    Deliberately narrower than SupervisorDecision: route_engine and
    relevant_datasets are owned by the router, not the model, and every field
    here is required so the emitted JSON schema satisfies strict mode.
    """

    intent: IntentType
    confidence: float
    reasoning: str
    clarification_question: Optional[str]
