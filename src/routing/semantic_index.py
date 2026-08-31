"""
Semantic route index backing the supervisor's local fast path.

Two layers, both sub-millisecond once warm:

1. Lexical anchors - regex greetings, catalog-name substrings, and keyword
   margin scoring. Deterministic, provider-independent, and the reason the
   fast path works offline where EMBEDDING_PROVIDER=mock yields hash-projection
   vectors with no usable cosine structure.
2. Dense anchors - global static intent phrases plus per-tenant phrases
   synthesized from registered table and document names, embedded once and
   compared by cosine similarity.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import Settings, get_settings
from src.ingestion.metadata_extractor import EmbeddingService
from src.routing.schemas import IntentType, TenantCatalog

logger = logging.getLogger(__name__)


# =============================================================================
# Global static anchors (shared across all tenants)
# =============================================================================

PURE_GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|greetings|howdy|hola|bonjour|sup)(\s+(there|all|team|bot|assistant|everyone|everybody|folks|friend|friends))*[\s!.,?]*$",
    r"^\s*(good\s+(morning|afternoon|evening|day|night))(\s+(there|all|team|bot|assistant|everyone|everybody|folks|friend|friends))*[\s!.,?]*$",
    r"^\s*(thanks|thank\s+you)(\s+(a\s+lot|so\s+much|very\s+much|again|all))?[\s!.,?]*$",
    r"^\s*(bye|goodbye|see\s+you(\s+(later|soon))?|cheers)[\s!.,?]*$",
]

GREETING_PREFIX_PATTERN = r"^\s*(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening|day))(\s+(there|all|team|bot|assistant|everyone|everybody|folks))*\s*[,!.:-]*\s*"

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

# Natural-language phrasings embedded as dense global anchors. Keyword lists
# above drive the lexical layer; these give the cosine layer real sentences.
GLOBAL_DENSE_ANCHORS: List[Tuple[str, IntentType]] = (
    [
        (phrase, "GREETING_OR_CHITCHAT")
        for phrase in CONVERSATIONAL_PHRASES
        + ["hello there", "good morning", "thanks a lot", "goodbye for now"]
    ]
    + [
        (phrase, "STRUCTURED_QUERY")
        for phrase in [
            "how many orders were completed",
            "total revenue by region",
            "average order amount per customer",
            "count the rows in the table",
            "list the top products by sales",
            "group sales by status and city",
            "run a sql aggregation over the dataset",
            "compute a breakdown of metrics",
        ]
    ]
    + [
        (phrase, "UNSTRUCTURED_QUERY")
        for phrase in [
            "what does the policy say about returns",
            "search the handbook for the procedure",
            "find the guideline in the documentation",
            "summarize the incident post-mortem",
            "what are the compliance requirements",
            "explain the deployment runbook",
            "look up the contract terms",
        ]
    ]
    + [
        (phrase, "AMBIGUOUS_QUERY")
        for phrase in [
            "show me the data",
            "tell me more",
            "give me a summary",
            "what happened",
            "can you help",
            "details please",
        ]
    ]
)


# Names too generic to identify a dataset. A table literally called "orders"
# or "data" would otherwise claim every query containing that word, and its
# synthesized anchors would collide with the global ambiguity anchors.
RESERVED_ANCHOR_TERMS = {
    "all",
    "csv",
    "data",
    "dataset",
    "datasets",
    "details",
    "doc",
    "docs",
    "document",
    "documents",
    "file",
    "files",
    "info",
    "information",
    "inventory",
    "misc",
    "new",
    "other",
    "overview",
    "policy",
    "products",
    "customers",
    "orders",
    "report",
    "reports",
    "sales",
    "sheet",
    "sheet1",
    "status",
    "summary",
    "table",
    "tables",
    "temp",
    "test",
    "untitled",
    "upload",
}

# Shorter than this, a token is an abbreviation or an id fragment, not a name.
MIN_ANCHOR_TOKEN_LEN = 3


def humanize(name: str) -> str:
    """sales_orders.csv -> sales orders"""
    stem = re.sub(r"\.[a-z0-9]{1,6}$", "", name.strip(), flags=re.IGNORECASE)
    return re.sub(r"[_\-\.]+", " ", stem).strip().lower()


def is_distinctive_name(name: str) -> bool:
    """
    Whether a dataset name is specific enough to route on.

    The hijack risk is a name that is one generic word: a table called "data"
    or "orders" would claim every query containing that word. A multi-word name
    is safe because the whole phrase has to appear on word boundaries, so
    "sales orders" and "inventory q3" are both kept.

    A rejected name is still queryable - it just cannot be matched by name, so
    those queries reach the LLM tier instead of being mis-claimed.
    """
    label = humanize(name)
    if not label:
        return False

    tokens = label.split()

    def is_noise(token: str) -> bool:
        """An id fragment or version suffix, carrying no meaning by itself."""
        return len(token) < MIN_ANCHOR_TOKEN_LEN or token.isdigit()

    if len(tokens) == 1:
        return tokens[0] not in RESERVED_ANCHOR_TERMS and not is_noise(tokens[0])
    # Multi-word names survive on the strength of the phrase - "sales orders"
    # is specific even though both words are generic alone. Only a name made
    # entirely of id fragments ("q3 v2") is unroutable.
    return not all(is_noise(t) for t in tokens)


def synthesize_anchors(name: str, kind: str) -> List[str]:
    """Auto-synthesize query phrasings for a registered table or document name."""
    label = humanize(name)
    if not is_distinctive_name(name):
        return []
    if kind == "structured":
        return [
            label,
            f"query {label}",
            f"how many {label}",
            f"total of {label}",
            f"{label} grouped by category",
        ]
    return [
        label,
        f"what does {label} say",
        f"search {label} for details",
        f"{label} policy and procedure",
    ]


@dataclass
class _AnchorMatrix:
    """L2-normalized anchor vectors with parallel intent and dataset labels."""

    vectors: np.ndarray  # (n, dim) float32
    labels: List[IntentType]
    datasets: List[str]  # "" for anchors not tied to a dataset

    def __len__(self) -> int:
        return len(self.labels)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class SemanticRouteIndex:
    """
    Multi-tenant semantic route index.

    Global anchors are embedded at construction. Tenant anchors are synthesized
    and embedded on registration, so no manual tagging is needed.

    Embedding failures are not caught. A half-built index would silently
    misroute, so a cold-start or provider failure surfaces to the caller.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.tenant_matrices: Dict[str, _AnchorMatrix] = {}
        self.tenant_catalogs: Dict[str, TenantCatalog] = {}
        # org_id -> compiled word-boundary pattern -> (intent, original name)
        self._tenant_lexicon: Dict[str, List[Tuple[re.Pattern, IntentType, str]]] = {}

        self.embedding_service = embedding_service or EmbeddingService(self.settings)
        self.global_matrix = self._build_matrix(
            [phrase for phrase, _ in GLOBAL_DENSE_ANCHORS],
            [intent for _, intent in GLOBAL_DENSE_ANCHORS],
            [""] * len(GLOBAL_DENSE_ANCHORS),
        )

    # -------------------------------------------------------------------------
    # Tenant registration
    # -------------------------------------------------------------------------

    def register_or_update_tenant(self, catalog: TenantCatalog) -> None:
        """
        Embed synthesized anchors for a tenant's tables and documents.

        A re-register with an unchanged catalog is a no-op, so callers may sync
        freely without paying the embedding cost again.
        """
        existing = self.tenant_catalogs.get(catalog.org_id)
        if existing is not None and (
            existing.structured_tables == catalog.structured_tables
            and existing.unstructured_docs == catalog.unstructured_docs
        ):
            return

        self.tenant_catalogs[catalog.org_id] = catalog

        lexicon: List[Tuple[re.Pattern, IntentType, str]] = []
        phrases: List[str] = []
        labels: List[IntentType] = []
        datasets: List[str] = []

        for names, intent, kind in (
            (catalog.structured_tables, "STRUCTURED_QUERY", "structured"),
            (catalog.unstructured_docs, "UNSTRUCTURED_QUERY", "unstructured"),
        ):
            for name in names:
                if not is_distinctive_name(name):
                    # Too generic to identify: matching it by name would claim
                    # unrelated queries. Such queries fall through to the LLM.
                    logger.info(
                        "Dataset %r in org %s is too generic to route by name; "
                        "no anchors synthesized.",
                        name,
                        catalog.org_id,
                    )
                    continue
                for variant in {humanize(name), name.lower()}:
                    lexicon.append(
                        (re.compile(rf"\b{re.escape(variant)}\b"), intent, name)
                    )
                for phrase in synthesize_anchors(name, kind):
                    phrases.append(phrase)
                    labels.append(intent)
                    datasets.append(name)

        # Longer names first so "sales orders eu" wins over "sales orders".
        lexicon.sort(key=lambda item: len(item[0].pattern), reverse=True)
        self._tenant_lexicon[catalog.org_id] = lexicon
        self.tenant_matrices.pop(catalog.org_id, None)

        if phrases:
            self.tenant_matrices[catalog.org_id] = self._build_matrix(phrases, labels, datasets)

    def catalog_names(self, org_id: str) -> List[str]:
        """All registered dataset names for a tenant, structured first."""
        catalog = self.tenant_catalogs.get(org_id)
        if catalog is None:
            return []
        return list(catalog.structured_tables) + list(catalog.unstructured_docs)

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    def score_query(self, org_id: str, query: str) -> Tuple[IntentType, float, List[str]]:
        """
        Score a query against global and tenant anchors.

        Returns (intent, confidence, relevant_datasets). Confidence is a
        similarity in [0, 1] directly comparable to the router's tier
        thresholds.
        """
        lexical = self._score_lexical(org_id, query)
        if lexical is not None:
            return lexical
        return self._score_dense(org_id, query)

    def _score_lexical(
        self, org_id: str, query: str
    ) -> Optional[Tuple[IntentType, float, List[str]]]:
        lower_q = query.strip().lower()
        if not lower_q:
            return ("AMBIGUOUS_QUERY", 0.92, self.catalog_names(org_id))

        for pattern in PURE_GREETING_PATTERNS:
            if re.match(pattern, lower_q, re.IGNORECASE):
                return ("GREETING_OR_CHITCHAT", 0.98, [])

        stripped = re.sub(GREETING_PREFIX_PATTERN, "", lower_q, flags=re.IGNORECASE).strip()
        eval_q = stripped or lower_q

        # Catalog-name hits dominate: an explicit dataset mention is unambiguous.
        matched: Dict[IntentType, List[str]] = {}
        for pattern, intent, original in self._tenant_lexicon.get(org_id, []):
            if pattern.search(eval_q):
                matched.setdefault(intent, [])
                if original not in matched[intent]:
                    matched[intent].append(original)
        if len(matched) == 1:
            intent, names = next(iter(matched.items()))
            return (intent, 0.95, names)

        unstruct_score = sum(1 for kw in UNSTRUCTURED_KEYWORDS if kw in eval_q)
        struct_score = sum(1 for kw in STRUCTURED_KEYWORDS if kw in eval_q)
        if matched:
            # Ambiguous catalog match: let keywords break the tie.
            struct_score += 3 * len(matched.get("STRUCTURED_QUERY", []))
            unstruct_score += 3 * len(matched.get("UNSTRUCTURED_QUERY", []))

        if any(phrase in eval_q for phrase in CONVERSATIONAL_PHRASES):
            if struct_score == 0 and unstruct_score == 0:
                return ("GREETING_OR_CHITCHAT", 0.98, [])

        for pattern in AMBIGUOUS_PATTERNS:
            if re.match(pattern, eval_q, re.IGNORECASE):
                return ("AMBIGUOUS_QUERY", 0.92, self.catalog_names(org_id))

        margin = abs(struct_score - unstruct_score)
        if margin >= 1:
            intent: IntentType = (
                "STRUCTURED_QUERY" if struct_score > unstruct_score else "UNSTRUCTURED_QUERY"
            )
            # Scale margin into [0.55, 0.90]: a one-keyword edge lands in the
            # LLM grey zone, a decisive edge takes the fast path.
            confidence = min(0.90, 0.55 + 0.07 * margin)
            datasets = matched.get(intent, []) or self._datasets_for_intent(org_id, intent)
            return (intent, round(confidence, 4), datasets)

        return None

    def _score_dense(self, org_id: str, query: str) -> Tuple[IntentType, float, List[str]]:
        # An embedding failure here is not routed around: a zeroed score would
        # look like a legitimate ambiguity verdict.
        q_vec = np.asarray(self.embedding_service.embed_text(query), dtype=np.float32).reshape(
            1, -1
        )
        q_vec = _l2_normalize(q_vec)

        matrices = [self.global_matrix]
        tenant = self.tenant_matrices.get(org_id)
        if tenant is not None:
            matrices.append(tenant)

        best_intent: IntentType = "AMBIGUOUS_QUERY"
        best_score = -1.0
        best_datasets: List[str] = []

        for matrix in matrices:
            sims = (matrix.vectors @ q_vec.T).ravel()
            top = int(np.argmax(sims))
            score = float(sims[top])
            if score > best_score:
                best_score = score
                best_intent = matrix.labels[top]
                best_datasets = [matrix.datasets[top]] if matrix.datasets[top] else []

        if not best_datasets:
            best_datasets = self._datasets_for_intent(org_id, best_intent)

        # Cosine can be negative; the tier thresholds live in [0, 1].
        return (best_intent, round(max(0.0, best_score), 4), best_datasets)

    def _datasets_for_intent(self, org_id: str, intent: IntentType) -> List[str]:
        catalog = self.tenant_catalogs.get(org_id)
        if catalog is None:
            return []
        if intent == "STRUCTURED_QUERY":
            return list(catalog.structured_tables)
        if intent == "UNSTRUCTURED_QUERY":
            return list(catalog.unstructured_docs)
        return []

    def _build_matrix(
        self, phrases: List[str], labels: List[IntentType], datasets: List[str]
    ) -> _AnchorMatrix:
        vectors = np.asarray(self.embedding_service.embed_texts(phrases), dtype=np.float32)
        return _AnchorMatrix(_l2_normalize(vectors), labels, datasets)
