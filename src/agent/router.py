"""
Supervisor Intent Classifier and Router for Multi-Agent Orchestration.
Classifies queries into four core intent routes:
- GREETING_OR_CHITCHAT: Direct conversational reply with zero database lookups or tool executions.
- AMBIGUOUS_QUERY: Proactively asks clarifying questions with candidate dataset/table suggestions.
- STRUCTURED_QUERY: Routes to specialized SQL/DataFrame execution engines.
- UNSTRUCTURED_QUERY: Routes to hybrid dense+sparse document RAG engine.
"""

import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.api.schemas import BaseModel, Field
from src.config import Settings, get_settings
from src.database.connection import DatabaseManager, get_db_manager


class SupervisorDecision(BaseModel):
    """Structured decision output from supervisor router."""

    intent: Literal[
        "GREETING_OR_CHITCHAT", "AMBIGUOUS_QUERY", "STRUCTURED_QUERY", "UNSTRUCTURED_QUERY"
    ]
    confidence: float = Field(default=0.95, ge=0.0, le=1.0, description="Routing confidence score")
    reasoning: str = Field(default="", description="Explanation of routing classification")
    suggested_strategy: Optional[Literal["dedicated_db", "duckdb", "pandas_sandbox"]] = Field(
        default="duckdb", description="Suggested execution strategy for structured queries"
    )
    relevant_datasets: List[str] = Field(
        default_factory=list, description="Target dataset names/IDs if identified"
    )
    clarification_question: Optional[str] = Field(
        default=None, description="Proactive clarification question for ambiguous queries"
    )


# Conversational phrases & greeting patterns
CONVERSATIONAL_PHRASES = [
    "who are you",
    "what are you",
    "what can you do",
    "what are your capabilities",
    "what is your name",
    "how can you help",
    "help me",
    "help",
    "tell me about yourself",
    "introduce yourself",
]

PURE_GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|greetings|howdy|hola|bonjour|sup)(\s+(there|all|team|bot|assistant|everyone|everybody|folks|friend|friends))*[\s!.,?]*$",
    r"^\s*(good\s+(morning|afternoon|evening|day|night))(\s+(there|all|team|bot|assistant|everyone|everybody|folks|friend|friends))*[\s!.,?]*$",
    r"^\s*(thanks|thank\s+you)(\s+(a\s+lot|so\s+much|very\s+much|again|all))?[\s!.,?]*$",
    r"^\s*(bye|goodbye|see\s+you(\s+(later|soon))?|cheers)[\s!.,?]*$",
]

# Ambiguous and underspecified queries
AMBIGUOUS_PATTERNS = [
    r"^\s*(data|show\s+data|show\s+me\s+data|get\s+data|all\s+data)\s*$",
    r"^\s*(tell\s+me|tell\s+me\s+more|what\s+happened|summary|overview|details|information)\s*$",
    r"^\s*(what\s+is\s+it|can\s+you\s+help|analyze|status)\s*$",
    r"^\s*(orders|sales|policy|customers|products|inventory)\s*$",
]

UNSTRUCTURED_KEYWORDS = [
    "policy",
    "policies",
    "handbook",
    "procedure",
    "procedures",
    "guideline",
    "guidelines",
    "protocol",
    "protocols",
    "manual",
    "manuals",
    "document",
    "documents",
    "documentation",
    "incident",
    "incidents",
    "security",
    "hr",
    "vacation",
    "leave",
    "remote",
    "return window",
    "terms",
    "contract",
    "contracts",
    "faq",
    "instructions",
    "sla",
    "slas",
    "compliance",
    "post-mortem",
    "postmortem",
    "on-call",
    "deployment",
    "code review",
    "review",
    "retention",
    "guide",
    "guides",
    "architecture",
    "runbook",
    "specification",
    "sop",
]

STRUCTURED_KEYWORDS = [
    "how many",
    "count",
    "sum",
    "total",
    "average",
    "avg",
    "min",
    "max",
    "highest",
    "lowest",
    "top",
    "bottom",
    "order",
    "orders",
    "revenue",
    "sales",
    "amount",
    "price",
    "cost",
    "customer",
    "customers",
    "product",
    "products",
    "group by",
    "per",
    "by city",
    "by status",
    "completed",
    "pending",
    "cancelled",
    "shipping",
    "quantity",
    "table",
    "rows",
    "sql",
    "dataframe",
    "pandas",
    "duckdb",
    "database",
    "find orders",
    "list orders",
    "filter",
    "aggregate",
    "mean",
    "median",
    "find",
    "list",
    "region",
    "records",
    "metrics",
    "trend",
    "breakdown",
]


class SupervisorRouter:
    """
    Supervisor intent router determining execution dispatch path.
    Uses token-efficient metadata inspection and heuristic/LLM classification.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        settings: Optional[Settings] = None,
    ):
        self.db_manager = db_manager or get_db_manager()
        self.settings = settings or get_settings()
        self._openai_client = None
        if self.settings.openai_api_key:
            try:
                from openai import OpenAI

                self._openai_client = OpenAI(api_key=self.settings.openai_api_key)
            except Exception:
                pass

    def classify_intent(self, query: str, session_id: Optional[str] = None) -> SupervisorDecision:
        """
        Classify user query into one of the 4 intent categories.
        """
        clean_q = query.strip()
        lower_q = clean_q.lower()

        # 1. Check exact pure greeting patterns
        for pattern in PURE_GREETING_PATTERNS:
            if re.match(pattern, lower_q, re.IGNORECASE):
                return SupervisorDecision(
                    intent="GREETING_OR_CHITCHAT",
                    confidence=0.98,
                    reasoning="Query matched conversational greeting.",
                    suggested_strategy=None,
                    relevant_datasets=[],
                    clarification_question=None,
                )

        # Strip conversational prefix if present for further intent analysis
        stripped_q = re.sub(
            r"^\s*(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening|day))(\s+(there|all|team|bot|assistant|everyone|everybody|folks))*\s*[,!.:-]*\s*",
            "",
            lower_q,
            flags=re.IGNORECASE,
        ).strip()

        eval_q = stripped_q if stripped_q else lower_q

        # Pre-compute keyword matching and catalog scores
        unstruct_score = sum(1 for kw in UNSTRUCTURED_KEYWORDS if kw in eval_q or kw in lower_q)
        struct_score = sum(1 for kw in STRUCTURED_KEYWORDS if kw in eval_q or kw in lower_q)

        # Inspect registered datasets in metadata catalog
        dataset_info = self._get_dataset_catalog_hints()
        for doc_title in dataset_info.get("unstructured", []):
            if doc_title.lower() in eval_q or doc_title.lower() in lower_q:
                unstruct_score += 3
        for tbl_title in dataset_info.get("structured", []):
            if tbl_title.lower() in eval_q or tbl_title.lower() in lower_q:
                struct_score += 3

        # Check conversational identity / capability inquiry
        if any(phrase in eval_q or phrase in lower_q for phrase in CONVERSATIONAL_PHRASES):
            if struct_score == 0 and unstruct_score == 0:
                return SupervisorDecision(
                    intent="GREETING_OR_CHITCHAT",
                    confidence=0.98,
                    reasoning="Query matched conversational capability or identity inquiry.",
                    suggested_strategy=None,
                    relevant_datasets=[],
                    clarification_question=None,
                )

        # 2. Check Ambiguous Query (Very short or underspecified queries)
        for pattern in AMBIGUOUS_PATTERNS:
            if re.match(pattern, eval_q, re.IGNORECASE):
                candidate_datasets = self._get_dataset_names()
                dataset_list_str = (
                    ", ".join(candidate_datasets)
                    if candidate_datasets
                    else "structured sales/orders and company policy documents"
                )
                return SupervisorDecision(
                    intent="AMBIGUOUS_QUERY",
                    confidence=0.92,
                    reasoning="Query is underspecified and lacks clear filtering, aggregation, or document context.",
                    suggested_strategy=None,
                    relevant_datasets=candidate_datasets,
                    clarification_question=(
                        f"Your query '{clean_q}' is broad. Are you looking to query structured datasets "
                        f"({dataset_list_str}) or search unstructured documentation (e.g. policy handbook, procedures)?"
                    ),
                )

        # 3. LLM-Based Intent Classification if OpenAI client is active
        if self._openai_client:
            try:
                import json

                llm_prompt = (
                    f"You are the Supervisor Router for an AI Knowledge Base Q&A platform.\n"
                    f"Classify the following user query into one of: STRUCTURED_QUERY, UNSTRUCTURED_QUERY, AMBIGUOUS_QUERY, GREETING_OR_CHITCHAT.\n"
                    f"Available structured tables: {dataset_info.get('structured', [])}\n"
                    f"Available unstructured documents: {dataset_info.get('unstructured', [])}\n"
                    f"User Query: {clean_q}\n\n"
                    f"Provide response strictly in JSON format:\n"
                    f'{{"intent": "...", "confidence": 0.95, "reasoning": "...", "suggested_strategy": "duckdb"}}'
                )
                resp = self._openai_client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[{"role": "user", "content": llm_prompt}],
                    temperature=0.0,
                    response_format=(
                        {"type": "json_object"} if hasattr(self.settings, "openai_model") else None
                    ),
                )
                raw_json = resp.choices[0].message.content or "{}"
                data = json.loads(raw_json)
                intent_val = data.get("intent", "").upper()
                if intent_val in (
                    "STRUCTURED_QUERY",
                    "UNSTRUCTURED_QUERY",
                    "AMBIGUOUS_QUERY",
                    "GREETING_OR_CHITCHAT",
                ):
                    strat = data.get("suggested_strategy")
                    if strat not in ("duckdb", "dedicated_db", "pandas_sandbox"):
                        strat = "duckdb"
                    return SupervisorDecision(
                        intent=intent_val,
                        confidence=float(data.get("confidence", 0.95)),
                        reasoning=str(
                            data.get("reasoning", f"LLM classified query as {intent_val}")
                        ),
                        suggested_strategy=strat if intent_val == "STRUCTURED_QUERY" else None,
                        relevant_datasets=dataset_info.get(
                            "structured" if intent_val == "STRUCTURED_QUERY" else "unstructured", []
                        ),
                        clarification_question=data.get("clarification_question"),
                    )
            except Exception:
                pass

        # 4. Deterministic / Heuristic Classification Fallback
        suggested_strategy: Literal["dedicated_db", "duckdb", "pandas_sandbox"] = "duckdb"
        if "pandas" in eval_q or "python" in eval_q or "dataframe" in eval_q or "sandbox" in eval_q:
            suggested_strategy = "pandas_sandbox"
        elif (
            "postgres" in eval_q
            or "dedicated" in eval_q
            or "postgresql" in eval_q
            or "sql table" in eval_q
        ):
            suggested_strategy = "dedicated_db"

        if unstruct_score > struct_score and unstruct_score > 0:
            return SupervisorDecision(
                intent="UNSTRUCTURED_QUERY",
                confidence=min(0.99, 0.75 + unstruct_score * 0.08),
                reasoning=f"Query references unstructured documentation keywords/titles (score: {unstruct_score}).",
                suggested_strategy=None,
                relevant_datasets=dataset_info.get("unstructured", []),
                clarification_question=None,
            )

        if struct_score > 0:
            return SupervisorDecision(
                intent="STRUCTURED_QUERY",
                confidence=min(0.99, 0.80 + struct_score * 0.05),
                reasoning=f"Query targets structured data aggregation, calculation, or table filtering (score: {struct_score}).",
                suggested_strategy=suggested_strategy,
                relevant_datasets=dataset_info.get("structured", []),
                clarification_question=None,
            )

        return SupervisorDecision(
            intent="UNSTRUCTURED_QUERY",
            confidence=0.70,
            reasoning="Defaulting to hybrid document retrieval for open-ended informational query.",
            suggested_strategy=None,
            relevant_datasets=dataset_info.get("unstructured", []),
            clarification_question=None,
        )

    def _get_dataset_names(self) -> List[str]:
        """Fetch human-readable names of all registered datasets."""
        try:
            datasets = self.db_manager.list_datasets()
            return [d.name for d in datasets]
        except Exception:
            return []

    def _get_dataset_catalog_hints(self) -> Dict[str, List[str]]:
        """Fetch structured table names and unstructured document names for classification."""
        try:
            datasets = self.db_manager.list_datasets()
            structured = [d.name for d in datasets if d.category == "structured"]
            unstructured = [d.name for d in datasets if d.category == "unstructured"]
            return {"structured": structured, "unstructured": unstructured}
        except Exception:
            return {"structured": [], "unstructured": []}
