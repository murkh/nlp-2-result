"""
Evaluation Module for Multi-Agent Knowledge Base Q&A.
Provides:
  - Ragas Unstructured Evaluation Suite (faithfulness, answer_relevancy, context_precision, context_recall)
  - Structured Ground-Truth Execution Equivalence Suite (normalize_dataframe, check_execution_equivalence)
  - Benchmark Evaluation Runner (EvaluationRunner, run_eval_cli)
"""

from src.evaluation.ragas_suite import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    RagasEvaluationResult,
    RagasEvaluator,
    RagasMetric,
    answer_relevancy,
    calculate_answer_relevancy,
    calculate_context_precision,
    calculate_context_recall,
    calculate_faithfulness,
    context_precision,
    context_recall,
    faithfulness,
)
from src.evaluation.structured_equivalence import (
    StructuredBenchmarkRecord,
    StructuredBenchmarkResult,
    StructuredEquivalenceEvaluator,
    assert_frame_equivalence,
    calculate_equivalence_rate,
    calculate_syntax_first_pass_rate,
    check_execution_equivalence,
    compute_latency_statistics,
    estimate_token_cost,
    normalize_dataframe,
)


def __getattr__(name: str):
    if name in ("EvaluationRunner", "run_eval_cli", "format_evaluation_tables"):
        from src.evaluation.runner import EvaluationRunner, format_evaluation_tables, run_eval_cli

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Ragas Suite
    "RagasEvaluator",
    "RagasEvaluationResult",
    "RagasMetric",
    "Faithfulness",
    "AnswerRelevancy",
    "ContextPrecision",
    "ContextRecall",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "calculate_faithfulness",
    "calculate_answer_relevancy",
    "calculate_context_precision",
    "calculate_context_recall",
    # Structured Equivalence Suite
    "StructuredEquivalenceEvaluator",
    "StructuredBenchmarkResult",
    "StructuredBenchmarkRecord",
    "normalize_dataframe",
    "check_execution_equivalence",
    "assert_frame_equivalence",
    "calculate_syntax_first_pass_rate",
    "calculate_equivalence_rate",
    "compute_latency_statistics",
    "estimate_token_cost",
    # Runner
    "EvaluationRunner",
    "run_eval_cli",
    "format_evaluation_tables",
]
