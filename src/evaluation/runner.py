"""
CLI and Programmatic Evaluation Runner for Milestone 4.
Executes Structured Execution Equivalence suites and Unstructured Ragas evaluation suites,
renders rich/tabulated summary telemetry, and exports comprehensive JSON evaluation artifacts.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from src.evaluation.compat import pd, np, DataFrame, Series

from src.evaluation.ragas_suite import (
    RagasEvaluator,
    RagasEvaluationResult,
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from src.evaluation.structured_equivalence import (
    StructuredEquivalenceEvaluator,
    StructuredBenchmarkResult,
    check_execution_equivalence,
    estimate_token_cost,
)


# =============================================================================
# Built-in Default Benchmark Datasets for Standalone Evaluation
# =============================================================================

DEFAULT_STRUCTURED_TEST_CASES = [
    {
        "test_id": "struct_01_total_revenue",
        "query": "What is the total revenue from all completed orders?",
        "engine": "dedicated_db",
        "df_golden": pd.DataFrame({"total_revenue": [941.30]}),
        "df_generated": pd.DataFrame({"total_revenue": [941.30]}),
        "latency_ms": 28.5,
        "prompt_tokens": 140,
        "completion_tokens": 35,
        "error": None,
    },
    {
        "test_id": "struct_02_duckdb_region_count",
        "query": "Count the number of orders by shipping city in alphabetical order",
        "engine": "duckdb",
        "df_golden": pd.DataFrame({
            "shipping_city": ["Austin", "Chicago", "New York", "San Francisco"],
            "order_count": [1, 1, 2, 1]
        }),
        "df_generated": pd.DataFrame({
            "shipping_city": ["Austin", "Chicago", "New York", "San Francisco"],
            "order_count": [1, 1, 2, 1]
        }),
        "latency_ms": 14.2,
        "prompt_tokens": 125,
        "completion_tokens": 42,
        "error": None,
    },
    {
        "test_id": "struct_03_pandas_average_ticket",
        "query": "Calculate average order amount for completed orders",
        "engine": "pandas_sandbox",
        "df_golden": pd.DataFrame({"avg_amount": [313.7667]}),
        "df_generated": pd.DataFrame({"avg_amount": [313.7667]}),
        "latency_ms": 45.0,
        "prompt_tokens": 160,
        "completion_tokens": 55,
        "error": None,
    },
    {
        "test_id": "struct_04_floating_point_epsilon",
        "query": "Calculate total tax amount with 8.875% tax rate",
        "engine": "duckdb",
        "df_golden": pd.DataFrame({"total_tax": [83.5404]}),
        "df_generated": pd.DataFrame({"total_tax": [83.54038]}),
        "latency_ms": 16.8,
        "prompt_tokens": 130,
        "completion_tokens": 38,
        "error": None,
    },
    {
        "test_id": "struct_05_column_permuted_order",
        "query": "List customer id, order status, and total amount for order 101",
        "engine": "dedicated_db",
        "df_golden": pd.DataFrame({
            "customer_id": [501],
            "status": ["completed"],
            "total_amount": [150.50]
        }),
        "df_generated": pd.DataFrame({
            "total_amount": [150.50],
            "status": ["completed"],
            "customer_id": [501]
        }),
        "latency_ms": 22.1,
        "prompt_tokens": 135,
        "completion_tokens": 30,
        "error": None,
    },
]

DEFAULT_UNSTRUCTURED_TEST_CASES = [
    {
        "test_id": "unstruct_01_deployment_guideline",
        "question": "What is the mandatory monitoring duration for canary deployments?",
        "contexts": [
            "All deployments must pass continuous integration tests before release. "
            "Canary releases should be monitored for at least 15 minutes before full traffic migration."
        ],
        "answer": "Canary releases must be monitored for at least 15 minutes before full traffic migration.",
        "ground_truth": "Canary releases must be monitored for at least 15 minutes before full traffic migration.",
    },
    {
        "test_id": "unstruct_02_incident_sla",
        "question": "Within how many hours must a post-mortem document be published after resolving a Severity 1 incident?",
        "contexts": [
            "When a Severity 1 incident occurs, page the on-call engineer and open an incident Slack channel. "
            "A post-mortem document must be published within 48 hours of resolution."
        ],
        "answer": "A post-mortem document must be published within 48 hours of incident resolution.",
        "ground_truth": "A post-mortem document must be published within 48 hours of resolution.",
    },
    {
        "test_id": "unstruct_03_code_review_signoff",
        "question": "Who must provide sign-off for security-sensitive pull requests?",
        "contexts": [
            "Every pull request requires two approving reviews and zero unresolved comments. "
            "Security-sensitive modules require an explicit sign-off from the Application Security team."
        ],
        "answer": "The Application Security team must provide an explicit sign-off for security-sensitive modules.",
        "ground_truth": "Security-sensitive modules require an explicit sign-off from the Application Security team.",
    },
    {
        "test_id": "unstruct_04_hallucination_degradation",
        "question": "What is the policy for company holiday bonuses?",
        "contexts": [
            "Engineering on-call rotation is scheduled on a weekly basis with compensatory time off."
        ],
        "answer": "Every employee receives a 20% year-end holiday bonus paid in December.",
        "ground_truth": "Company holiday bonus policies are determined by executive leadership annually.",
    },
]


# =============================================================================
# Evaluation Runner Core
# =============================================================================

class EvaluationRunner:
    """
    Orchestrates end-to-end evaluation runs, formatted reporting, and artifact persistence.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "eval_output",
        tolerance: float = 1e-4,
        default_model: str = "gpt-4o-mini",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.structured_evaluator = StructuredEquivalenceEvaluator(
            tolerance=tolerance, default_model=default_model
        )
        self.ragas_evaluator = RagasEvaluator()

    def run_structured_suite(
        self,
        test_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> StructuredBenchmarkResult:
        """Run Structured Execution Equivalence benchmark suite."""
        cases = test_cases or DEFAULT_STRUCTURED_TEST_CASES
        return self.structured_evaluator.evaluate_benchmark(cases)

    def run_unstructured_suite(
        self,
        test_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> RagasEvaluationResult:
        """Run Ragas Unstructured Evaluation suite."""
        cases = test_cases or DEFAULT_UNSTRUCTURED_TEST_CASES
        return self.ragas_evaluator.evaluate_test_cases(cases)

    def run_all(
        self,
        structured_cases: Optional[List[Dict[str, Any]]] = None,
        unstructured_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run both structured and unstructured evaluation suites."""
        struct_res = self.run_structured_suite(structured_cases)
        unstruct_res = self.run_unstructured_suite(unstructured_cases)

        timestamp = datetime.now(timezone.utc).isoformat()
        combined_summary = {
            "timestamp": timestamp,
            "structured_summary": struct_res.summary_dict(),
            "unstructured_summary": unstruct_res.summary_dict(),
        }

        combined_details = {
            "timestamp": timestamp,
            "structured_details": struct_res.details,
            "unstructured_details": unstruct_res.details,
        }

        # Save artifacts
        summary_path, details_path = self.save_artifacts(combined_summary, combined_details)

        return {
            "timestamp": timestamp,
            "structured": struct_res,
            "unstructured": unstruct_res,
            "summary": combined_summary,
            "artifacts": {
                "summary_path": str(summary_path),
                "details_path": str(details_path),
            },
        }

    def save_artifacts(
        self,
        summary_data: Dict[str, Any],
        details_data: Dict[str, Any],
        prefix: str = "eval",
    ) -> Tuple[Path, Path]:
        """Save evaluation summary and full details as JSON artifacts."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = self.output_dir / f"{prefix}_summary.json"
        details_file = self.output_dir / f"{prefix}_details.json"

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, default=str)

        with open(details_file, "w", encoding="utf-8") as f:
            json.dump(details_data, f, indent=2, default=str)

        return summary_file, details_file


# =============================================================================
# Formatted Output Display Helpers
# =============================================================================

def format_evaluation_tables(
    structured_result: Optional[StructuredBenchmarkResult] = None,
    unstructured_result: Optional[RagasEvaluationResult] = None,
) -> str:
    """Format evaluation results as clean human-readable tables."""
    output_sections: List[str] = []

    # 1. Structured Results Table
    if structured_result:
        output_sections.append("================================================================================")
        output_sections.append("             STRUCTURED EXECUTION EQUIVALENCE BENCHMARK SUMMARY                 ")
        output_sections.append("================================================================================")
        sum_dict = structured_result.summary_dict()
        output_sections.append(f"Total Test Cases:               {sum_dict['total_cases']}")
        output_sections.append(f"Syntax First-Pass Rate:         {sum_dict['syntax_first_pass_rate_pct']:.2f}%")
        output_sections.append(f"Execution Equivalence Rate:     {sum_dict['equivalence_rate_pct']:.2f}%")
        output_sections.append(f"Mean Latency (ms):              {sum_dict['mean_latency_ms']:.2f} ms")
        output_sections.append(f"P95 Latency (ms):               {sum_dict['p95_latency_ms']:.2f} ms")
        output_sections.append(f"Total Tokens Consumed:          {sum_dict['total_tokens']}")
        output_sections.append(f"Estimated Cost (USD):           ${sum_dict['total_cost_usd']:.6f}")
        output_sections.append("")

        if structured_result.per_engine_stats:
            output_sections.append("--- Per-Engine Breakdown ---")
            output_sections.append(f"{'Engine':<20} | {'Cases':<6} | {'Syntax %':<10} | {'Equiv %':<10} | {'Mean ms':<10} | {'Tokens':<8}")
            output_sections.append("-" * 75)
            for eng, stats in structured_result.per_engine_stats.items():
                output_sections.append(
                    f"{eng:<20} | {stats['total_cases']:<6} | "
                    f"{stats['syntax_first_pass_rate']*100:<9.1f}% | "
                    f"{stats['equivalence_rate']*100:<9.1f}% | "
                    f"{stats['mean_latency_ms']:<10.2f} | "
                    f"{stats['total_tokens']:<8}"
                )
            output_sections.append("")

    # 2. Unstructured Ragas Results Table
    if unstructured_result:
        output_sections.append("================================================================================")
        output_sections.append("                  RAGAS UNSTRUCTURED EVALUATION SUMMARY                         ")
        output_sections.append("================================================================================")
        ragas_sum = unstructured_result.summary_dict()
        output_sections.append(f"Faithfulness:                   {ragas_sum.get('faithfulness', 0.0):.4f}")
        output_sections.append(f"Answer Relevancy:               {ragas_sum.get('answer_relevancy', 0.0):.4f}")
        output_sections.append(f"Context Precision:              {ragas_sum.get('context_precision', 0.0):.4f}")
        output_sections.append(f"Context Recall:                 {ragas_sum.get('context_recall', 0.0):.4f}")
        output_sections.append("")

    return "\n".join(output_sections)


# =============================================================================
# CLI Entrypoint
# =============================================================================

def run_eval_cli(args: Optional[List[str]] = None) -> int:
    """
    Main CLI entrypoint for evaluation runner (`nlp-eval`).
    """
    parser = argparse.ArgumentParser(
        description="Multi-Agent Knowledge Base Evaluation Runner (Milestone 4)"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "structured", "unstructured"],
        default="all",
        help="Evaluation suite to execute (default: all)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional path to custom test cases JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_output",
        help="Directory to save evaluation artifacts (default: eval_output)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Floating point equivalence tolerance (default: 1e-4)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "summary"],
        default="table",
        help="Console output format (default: table)",
    )

    parsed = parser.parse_args(args)
    runner = EvaluationRunner(output_dir=parsed.output_dir, tolerance=parsed.tolerance)

    custom_cases = None
    if parsed.dataset:
        dpath = Path(parsed.dataset)
        if dpath.exists():
            with open(dpath, "r", encoding="utf-8") as f:
                custom_cases = json.load(f)

    if parsed.mode == "structured":
        struct_res = runner.run_structured_suite(custom_cases)
        if parsed.format == "json":
            print(json.dumps(struct_res.to_dict(), indent=2, default=str))
        else:
            print(format_evaluation_tables(structured_result=struct_res))
    elif parsed.mode == "unstructured":
        unstruct_res = runner.run_unstructured_suite(custom_cases)
        if parsed.format == "json":
            print(json.dumps(unstruct_res.to_dict(), indent=2, default=str))
        else:
            print(format_evaluation_tables(unstructured_result=unstruct_res))
    else:
        results = runner.run_all(
            structured_cases=custom_cases.get("structured") if isinstance(custom_cases, dict) else None,
            unstructured_cases=custom_cases.get("unstructured") if isinstance(custom_cases, dict) else None,
        )
        if parsed.format == "json":
            print(json.dumps(results["summary"], indent=2, default=str))
        else:
            print(format_evaluation_tables(
                structured_result=results["structured"],
                unstructured_result=results["unstructured"],
            ))
            print(f"Artifacts saved to: {results['artifacts']['summary_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(run_eval_cli())
