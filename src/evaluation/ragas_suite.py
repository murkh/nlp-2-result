"""
Ragas Unstructured Evaluation Suite.
Evaluates Hybrid RAG and Unstructured Document QA retrieval and synthesis quality.
Implements the 4 core Ragas metrics:
  1. Faithfulness (factual grounding against retrieved context)
  2. Answer Relevancy (semantic & lexical relevance to user query)
  3. Context Precision (mean reciprocal rank / precision@k of relevant context chunks)
  4. Context Recall (proportion of ground-truth facts retrieved in contexts)

Supports both official Ragas library evaluation and a high-fidelity standalone evaluation engine
for offline, zero-dependency, and deterministic testing.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

from src.evaluation.compat import DataFrame, Series, np, pd

# =============================================================================
# Stopwords and Tokenization Helpers
# =============================================================================

DEFAULT_STOPWORDS: Set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def tokenize_text(text: Optional[str], remove_stopwords: bool = True) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    if not text:
        return []
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", str(text).lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in DEFAULT_STOPWORDS]
    return tokens


def extract_claims(text: Optional[str]) -> List[str]:
    """Extract individual sentences / factual assertions from text."""
    if not text or not str(text).strip():
        return []
    # Split by period, semicolon, exclamation, or question mark followed by space or newline
    raw_sentences = re.split(r"[.;!?\n]+", str(text))
    claims = [
        s.strip()
        for s in raw_sentences
        if s.strip() and len(tokenize_text(s, remove_stopwords=True)) > 0
    ]
    return claims


def extract_facts(text: Optional[str]) -> List[str]:
    """Extract distinct ground-truth factual statements or phrases."""
    if not text or not str(text).strip():
        return []
    # If text is multiline or has bullet points / commas in lists
    lines = [line.strip().lstrip("-*•0123456789. ") for line in str(text).splitlines()]
    facts = []
    for line in lines:
        if not line:
            continue
        sub_sentences = [s.strip() for s in re.split(r"[;!?]+", line) if s.strip()]
        for sub in sub_sentences:
            if tokenize_text(sub, remove_stopwords=True):
                facts.append(sub)
    if not facts:
        # Fallback to sentence extraction
        facts = extract_claims(text)
    return facts if facts else [str(text).strip()]


# =============================================================================
# Standalone Core Metric Calculators
# =============================================================================


def stem_word(w: str) -> str:
    """Lightweight suffix stemmer for robust entity and morphological matching."""
    s = str(w).lower().strip()
    for suffix in ("ing", "ed", "es", "ly", "tion", "ions", "ment", "s"):
        if len(s) > len(suffix) + 2 and s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def tokens_match(t1: str, t2: str) -> bool:
    """Check if two tokens match verbatim, by stem, or by substring containment."""
    t1_s, t2_s = str(t1).lower(), str(t2).lower()
    if t1_s == t2_s:
        return True
    if stem_word(t1_s) == stem_word(t2_s):
        return True
    if len(t1_s) >= 4 and len(t2_s) >= 4 and (t1_s in t2_s or t2_s in t1_s):
        return True
    return False


def calculate_faithfulness(
    question: Optional[str],
    answer: Optional[str],
    contexts: Optional[Union[List[str], str]],
) -> float:
    """
    Measures faithfulness: ratio of answer claims inferable / supported by retrieved context.
    Returns float in [0.0, 1.0].
    Zero context or empty answer results in 0.0.
    """
    if not contexts:
        return 0.0
    if not answer or not str(answer).strip():
        return 0.0

    if isinstance(contexts, str):
        context_list = [contexts]
    else:
        context_list = [c for c in contexts if c and str(c).strip()]

    if not context_list:
        return 0.0

    full_context = " ".join(context_list).lower()
    context_tokens = set(tokenize_text(full_context, remove_stopwords=True))

    claims = extract_claims(answer)
    if not claims:
        return 0.0

    supported_count = 0
    for claim in claims:
        claim_str = claim.lower()
        # Direct substring check
        if claim_str in full_context:
            supported_count += 1
            continue

        claim_tokens = tokenize_text(claim, remove_stopwords=True)
        if not claim_tokens:
            continue

        # Check token containment ratio with morphological matching
        matching_tokens = [
            t
            for t in claim_tokens
            if any(tokens_match(t, ct) for ct in context_tokens) or t in full_context
        ]
        containment = len(matching_tokens) / len(claim_tokens)

        # Check for numbers/identifiers specifically (e.g. 512MB, 30 days)
        claim_nums = re.findall(r"\b\d+[a-zA-Z]*\b", claim_str)
        nums_matched = True
        if claim_nums:
            nums_matched = all(n in full_context for n in claim_nums)

        if containment >= 0.50 and nums_matched:
            supported_count += 1

    score = supported_count / len(claims)
    return max(0.0, min(1.0, float(round(score, 4))))


def calculate_answer_relevancy(
    question: Optional[str],
    answer: Optional[str],
) -> float:
    """
    Measures answer relevancy: semantic and lexical relevance of answer to question.
    Returns float in [0.0, 1.0].
    Identical or directly answering text scores 1.0.
    Completely unrelated text scores 0.0.
    """
    if not question or not str(question).strip():
        return 0.0
    if not answer or not str(answer).strip():
        return 0.0

    q_str = str(question).strip().lower()
    a_str = str(answer).strip().lower()

    if q_str == a_str:
        return 1.0

    q_tokens = tokenize_text(q_str, remove_stopwords=True)
    a_tokens = tokenize_text(a_str, remove_stopwords=True)

    if not q_tokens or not a_tokens:
        # Fallback to non-filtered tokens
        q_tokens = tokenize_text(q_str, remove_stopwords=False)
        a_tokens = tokenize_text(a_str, remove_stopwords=False)

    if not q_tokens or not a_tokens:
        return 0.0

    matched_q = 0
    matched_set = set()
    for q_t in q_tokens:
        for a_t in a_tokens:
            if tokens_match(q_t, a_t):
                matched_q += 1
                matched_set.add(q_t)
                break

    if matched_q == 0:
        return 0.0

    # Recall of question terms in answer (crucial for relevance)
    q_recall = matched_q / len(q_tokens)

    # Jaccard and precision
    intersection_len = len(matched_set)
    union_len = len(q_tokens) + len(a_tokens) - intersection_len
    jaccard = intersection_len / union_len if union_len > 0 else 0.0
    a_precision = intersection_len / len(a_tokens) if a_tokens else 0.0

    score = (0.70 * q_recall) + (0.20 * jaccard) + (0.10 * a_precision)

    # If key question keywords are present in the answer, it strongly addresses the question
    if q_recall >= 0.70:
        score = max(score, 0.85)

    if q_recall >= 0.50 and len(a_tokens) <= 15:
        score = max(score, 0.75)

    return max(0.0, min(1.0, float(round(score, 4))))


def calculate_context_precision(
    question: Optional[str],
    contexts: Optional[Union[List[str], str]],
    ground_truth: Optional[str] = None,
) -> float:
    """
    Measures context precision: Mean reciprocal rank / precision@k of relevant context chunks.
    Evaluates whether relevant contexts are ranked higher in the retrieval results.
    Returns float in [0.0, 1.0]. Empty ground truth or empty contexts return 0.0 (nan-safe).
    """
    if not contexts:
        return 0.0

    target_ref = (
        ground_truth if (ground_truth is not None and str(ground_truth).strip()) else question
    )
    if not target_ref or not str(target_ref).strip():
        return 0.0

    if isinstance(contexts, str):
        context_list = [contexts]
    else:
        context_list = [c for c in contexts if c and str(c).strip()]

    if not context_list:
        return 0.0

    target_tokens = set(tokenize_text(target_ref, remove_stopwords=True))
    if not target_tokens:
        target_tokens = set(tokenize_text(target_ref, remove_stopwords=False))

    relevant_indicator: List[int] = []
    for ctx in context_list:
        ctx_str = str(ctx).strip().lower()
        ref_str = str(target_ref).strip().lower()

        # Exact match or substring
        if ref_str in ctx_str or ctx_str in ref_str:
            relevant_indicator.append(1)
            continue

        ctx_tokens = set(tokenize_text(ctx_str, remove_stopwords=True))
        overlap = target_tokens.intersection(ctx_tokens)

        # Consider relevant if significant keyword overlap
        overlap_ratio = len(overlap) / len(target_tokens) if target_tokens else 0.0
        if overlap_ratio >= 0.40 or len(overlap) >= 2:
            relevant_indicator.append(1)
        else:
            relevant_indicator.append(0)

    total_relevant = sum(relevant_indicator)
    if total_relevant == 0:
        return 0.0

    # Calculate cumulative precision at rank k
    precisions_at_k = []
    running_rel = 0
    for k, is_rel in enumerate(relevant_indicator, start=1):
        if is_rel == 1:
            running_rel += 1
            precision_k = running_rel / k
            precisions_at_k.append(precision_k)

    if not precisions_at_k:
        return 0.0

    context_precision_score = sum(precisions_at_k) / total_relevant
    return max(0.0, min(1.0, float(round(context_precision_score, 4))))


def calculate_context_recall(
    ground_truth: Optional[str],
    contexts: Optional[Union[List[str], str]],
) -> float:
    """
    Measures context recall: proportion of ground-truth facts/sentences present in retrieved contexts.
    Returns float in [0.0, 1.0].
    """
    if not ground_truth or not str(ground_truth).strip():
        return 0.0
    if not contexts:
        return 0.0

    if isinstance(contexts, str):
        context_list = [contexts]
    else:
        context_list = [c for c in contexts if c and str(c).strip()]

    if not context_list:
        return 0.0

    full_context = " ".join(context_list).lower()
    facts = extract_facts(ground_truth)
    if not facts:
        return 0.0

    recalled_count = 0
    for fact in facts:
        fact_str = fact.lower()
        if fact_str in full_context:
            recalled_count += 1
            continue

        fact_tokens = tokenize_text(fact, remove_stopwords=True)
        if not fact_tokens:
            fact_tokens = tokenize_text(fact, remove_stopwords=False)

        if not fact_tokens:
            continue

        matched_tokens = [t for t in fact_tokens if t in full_context]
        containment = len(matched_tokens) / len(fact_tokens)

        # Numbers check
        fact_nums = re.findall(r"\b\d+[a-zA-Z]*\b", fact_str)
        nums_ok = True
        if fact_nums:
            nums_ok = all(n in full_context for n in fact_nums)

        if containment >= 0.60 and nums_ok:
            recalled_count += 1

    score = recalled_count / len(facts)
    return max(0.0, min(1.0, float(round(score, 4))))


# =============================================================================
# Metric Wrapper Classes (Compatible with Ragas API)
# =============================================================================


class RagasMetric:
    """Base class for Ragas evaluation metrics."""

    name: str

    def __call__(self, *args, **kwargs) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<RagasMetric: {self.name}>"


class Faithfulness(RagasMetric):
    name = "faithfulness"

    def score(self, question: str, answer: str, contexts: List[str]) -> float:
        return calculate_faithfulness(question=question, answer=answer, contexts=contexts)

    def __call__(
        self, question: str = "", answer: str = "", contexts: Optional[List[str]] = None, **kwargs
    ) -> float:
        return self.score(question=question, answer=answer, contexts=contexts or [])


class AnswerRelevancy(RagasMetric):
    name = "answer_relevancy"

    def score(self, question: str, answer: str) -> float:
        return calculate_answer_relevancy(question=question, answer=answer)

    def __call__(self, question: str = "", answer: str = "", **kwargs) -> float:
        return self.score(question=question, answer=answer)


class ContextPrecision(RagasMetric):
    name = "context_precision"

    def score(
        self, question: str, contexts: List[str], ground_truth: Optional[str] = None
    ) -> float:
        return calculate_context_precision(
            question=question, contexts=contexts, ground_truth=ground_truth
        )

    def __call__(
        self,
        question: str = "",
        contexts: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> float:
        return self.score(question=question, contexts=contexts or [], ground_truth=ground_truth)


class ContextRecall(RagasMetric):
    name = "context_recall"

    def score(self, ground_truth: str, contexts: List[str]) -> float:
        return calculate_context_recall(ground_truth=ground_truth, contexts=contexts)

    def __call__(
        self, ground_truth: str = "", contexts: Optional[List[str]] = None, **kwargs
    ) -> float:
        return self.score(ground_truth=ground_truth, contexts=contexts or [])


# Singleton Metric Instances for direct import and use
faithfulness = Faithfulness()
answer_relevancy = AnswerRelevancy()
context_precision = ContextPrecision()
context_recall = ContextRecall()

VALID_METRIC_MAP: Dict[str, RagasMetric] = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "relevancy": answer_relevancy,
    "context_precision": context_precision,
    "precision": context_precision,
    "context_recall": context_recall,
    "recall": context_recall,
}


# =============================================================================
# Evaluation Result Container
# =============================================================================


@dataclass
class RagasEvaluationResult:
    """Evaluation result container with summary scores, per-case records, and DataFrame export."""

    summary: Dict[str, float] = field(default_factory=dict)
    details: List[Dict[str, Any]] = field(default_factory=list)

    def __getitem__(self, item: str) -> Any:
        if item in self.summary:
            return self.summary[item]
        if item == "summary":
            return self.summary
        if item == "details" or item == "records":
            return self.details
        raise KeyError(f"Key '{item}' not found in RagasEvaluationResult summary or properties.")

    def get(self, item: str, default: Any = None) -> Any:
        if item in self.summary:
            return self.summary[item]
        if item == "summary":
            return self.summary
        if item == "details" or item == "records":
            return self.details
        return default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "details": self.details,
        }

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.details)

    def to_dataframe(self) -> pd.DataFrame:
        return self.to_pandas()

    def summary_dict(self) -> Dict[str, float]:
        return self.summary


# =============================================================================
# Main Ragas Evaluator
# =============================================================================


class RagasEvaluator:
    """
    Ragas Unstructured Evaluation Suite.
    Computes faithfulness, answer_relevancy, context_precision, and context_recall.
    """

    def __init__(
        self,
        metrics: Optional[List[Union[str, RagasMetric, Any]]] = None,
        use_official_ragas: bool = False,
    ):
        """
        Initialize RagasEvaluator with specified metrics.
        Raises ValueError if an invalid metric name is passed.
        """
        self.use_official_ragas = use_official_ragas
        self.metrics: List[RagasMetric] = []
        self.metric_names: List[str] = []

        if metrics is None:
            self.metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            self.metric_names = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]
        else:
            for m in metrics:
                if isinstance(m, str):
                    name_clean = m.strip().lower()
                    if name_clean not in VALID_METRIC_MAP:
                        raise ValueError(
                            f"Unsupported metric: '{m}'. Valid metrics are: "
                            f"{list(set(VALID_METRIC_MAP.keys()))}"
                        )
                    resolved_metric = VALID_METRIC_MAP[name_clean]
                    if resolved_metric not in self.metrics:
                        self.metrics.append(resolved_metric)
                        self.metric_names.append(resolved_metric.name)
                elif isinstance(m, RagasMetric):
                    if m not in self.metrics:
                        self.metrics.append(m)
                        self.metric_names.append(m.name)
                elif hasattr(m, "name"):
                    name_clean = getattr(m, "name").strip().lower()
                    if name_clean not in VALID_METRIC_MAP:
                        raise ValueError(f"Unsupported metric object: '{m}'")
                    resolved_metric = VALID_METRIC_MAP[name_clean]
                    if resolved_metric not in self.metrics:
                        self.metrics.append(resolved_metric)
                        self.metric_names.append(resolved_metric.name)
                else:
                    raise ValueError(f"Invalid metric specification: {m}")

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate a single test case across all configured metrics."""
        ctx_list = contexts or []
        scores: Dict[str, float] = {}

        for metric in self.metrics:
            if metric.name == "faithfulness":
                scores["faithfulness"] = calculate_faithfulness(
                    question=question, answer=answer, contexts=ctx_list
                )
            elif metric.name == "answer_relevancy":
                scores["answer_relevancy"] = calculate_answer_relevancy(
                    question=question, answer=answer
                )
            elif metric.name == "context_precision":
                scores["context_precision"] = calculate_context_precision(
                    question=question, contexts=ctx_list, ground_truth=ground_truth
                )
            elif metric.name == "context_recall":
                scores["context_recall"] = calculate_context_recall(
                    ground_truth=ground_truth or "", contexts=ctx_list
                )

        return scores

    def evaluate_test_cases(
        self,
        test_cases: Union[List[Dict[str, Any]], pd.DataFrame, Any],
    ) -> RagasEvaluationResult:
        """
        Evaluate a batch of unstructured RAG test cases.
        Supports inputs as:
          - List of dicts: [{"question": ..., "contexts": [...], "answer": ..., "ground_truth": ...}]
          - pandas.DataFrame
          - HuggingFace / Ragas Dataset
        """
        records: List[Dict[str, Any]] = []

        if isinstance(test_cases, pd.DataFrame):
            records = test_cases.to_dict(orient="records")
        elif isinstance(test_cases, list):
            records = test_cases
        elif hasattr(test_cases, "to_pandas"):
            records = test_cases.to_pandas().to_dict(orient="records")
        elif hasattr(test_cases, "to_dict"):
            d = test_cases.to_dict()
            if isinstance(d, dict):
                # check column-oriented vs record-oriented
                keys = list(d.keys())
                if keys and isinstance(d[keys[0]], (list, tuple)):
                    length = len(d[keys[0]])
                    records = [{k: d[k][i] for k in keys} for i in range(length)]
                else:
                    records = [d]
        else:
            records = list(test_cases)

        if not records:
            empty_summary = {name: 0.0 for name in self.metric_names}
            return RagasEvaluationResult(summary=empty_summary, details=[])

        # Optional: Try official ragas library if explicitly requested
        if self.use_official_ragas:
            try:
                from datasets import Dataset
                from ragas import evaluate as ragas_eval

                df = pd.DataFrame(records)
                dataset = Dataset.from_pandas(df)
                official_score = ragas_eval(dataset=dataset, metrics=self.metrics)
                res_df = official_score.to_pandas()
                summary = {
                    col: float(res_df[col].mean())
                    for col in res_df.columns
                    if col in self.metric_names
                }
                details = res_df.to_dict(orient="records")
                return RagasEvaluationResult(summary=summary, details=details)
            except Exception:
                # Seamless fallback to standalone built-in evaluation
                pass

        details: List[Dict[str, Any]] = []
        metric_accumulators: Dict[str, List[float]] = {name: [] for name in self.metric_names}

        for record in records:
            # Normalize field names
            q = record.get("question") or record.get("query") or record.get("user_input") or ""
            ans = (
                record.get("answer")
                or record.get("response")
                or record.get("generated_answer")
                or ""
            )

            raw_ctx = (
                record.get("contexts")
                or record.get("context")
                or record.get("retrieved_contexts")
                or []
            )
            if isinstance(raw_ctx, str):
                ctx_list = [raw_ctx]
            elif isinstance(raw_ctx, list):
                ctx_list = [str(c) for c in raw_ctx]
            else:
                ctx_list = []

            gt = (
                record.get("ground_truth") or record.get("ground_truths") or record.get("reference")
            )
            if isinstance(gt, list):
                gt_str = " ".join(str(g) for g in gt)
            else:
                gt_str = str(gt) if gt is not None else ""

            case_scores = self.evaluate_single(
                question=str(q),
                answer=str(ans),
                contexts=ctx_list,
                ground_truth=gt_str,
            )

            detail_entry = dict(record)
            detail_entry.update(case_scores)
            details.append(detail_entry)

            for m_name, score_val in case_scores.items():
                if m_name in metric_accumulators:
                    metric_accumulators[m_name].append(score_val)

        summary: Dict[str, float] = {}
        for m_name in self.metric_names:
            vals = metric_accumulators.get(m_name, [])
            avg = sum(vals) / len(vals) if vals else 0.0
            summary[m_name] = round(avg, 4)

        return RagasEvaluationResult(summary=summary, details=details)

    def evaluate(
        self, dataset: Union[List[Dict[str, Any]], pd.DataFrame, Any]
    ) -> RagasEvaluationResult:
        """Alias for evaluate_test_cases."""
        return self.evaluate_test_cases(dataset)
