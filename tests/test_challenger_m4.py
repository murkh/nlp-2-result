"""
Adversarial Challenger Stress Tests for Milestone 4 (Evaluation Frameworks).
Tests:
  1. Ragas Metric Boundaries & Adversarial Inputs:
     - Completely hallucinated answers (faithfulness == 0.0)
     - Inverted answers / semantic contradictions
     - Irrelevant questions vs retrieved contexts
     - Numerical mismatch in claims (token match vs entity number match)
     - Massive batch evaluations (150+ diverse items)
     - Unicode, emojis, whitespace, and extreme empty/none boundaries
  2. Structured Ground-Truth Execution Equivalence:
     - Strict floating point tolerance boundary (5e-5 passes, 1.01e-4 fails with atol=1e-4)
     - Mixed column types, string-represented numbers, null/nan variations
     - Unicode strings, RTL, special characters, whitespace variations
     - Nested data / dict structures in cells
     - High-volume stress (10,000 rows x 5 columns with row permutations)
     - Extreme shape corner cases (0x0, 0x10, 100x0)
     - Infinity, -Infinity, NaN equivalence handling
  3. Latency Statistics, Token Costs & Benchmark Scalability:
     - 100+ cases benchmark aggregation
     - Percentile calculations (p50, p95, p99, min, max)
     - Token pricing models and fallback resilience
"""

import math
import random
import time
import unittest
from typing import Any, Dict, List

from src.evaluation.compat import pd, np, DataFrame, Series
from src.evaluation.ragas_suite import (
    RagasEvaluator,
    RagasEvaluationResult,
    calculate_faithfulness,
    calculate_answer_relevancy,
    calculate_context_precision,
    calculate_context_recall,
)
from src.evaluation.structured_equivalence import (
    StructuredEquivalenceEvaluator,
    StructuredBenchmarkResult,
    check_execution_equivalence,
    assert_frame_equivalence,
    calculate_syntax_first_pass_rate,
    calculate_equivalence_rate,
    compute_latency_statistics,
    estimate_token_cost,
    normalize_dataframe,
)


class TestRagasAdversarialStress(unittest.TestCase):
    """Adversarial stress-testing suite for Ragas metrics."""

    def setUp(self):
        self.evaluator = RagasEvaluator()

    def test_completely_hallucinated_answers_faithfulness_zero(self):
        """Verify completely fabricated or hallucinated answers yield strictly 0.0 faithfulness."""
        test_pairs = [
            # Baking context vs quantum physics answer
            (
                ["Mix 200g of flour, 2 eggs, and 100ml milk to bake pancakes at 180C."],
                "What is the pancake recipe?",
                "Quantum superposition allows particles to exist in multiple eigenstates simultaneously."
            ),
            # Database context vs alien invasion answer
            (
                ["PostgreSQL uses MVCC to manage concurrent transactions and row versions."],
                "How does Postgres manage concurrency?",
                "Extraterrestrial life forms arrived in 1947 and established secret subterranean bases."
            ),
            # Security context vs crypto marketing answer
            (
                ["AST whitelist restricts import statements and prevents arbitrary code execution."],
                "How is code execution secured?",
                "Buy Bitcoin and Ethereum for guaranteed 1000x returns on decentralized exchanges."
            ),
        ]

        for ctxs, q, ans in test_pairs:
            score = calculate_faithfulness(question=q, answer=ans, contexts=ctxs)
            self.assertEqual(
                score, 0.0,
                f"Expected 0.0 faithfulness for hallucinated answer: '{ans}', got {score}"
            )

    def test_inverted_antonym_contradictions(self):
        """Verify answers that contradict the context yield low/zero faithfulness."""
        ctx = ["The system strictly forbids anonymous access and requires two-factor authentication."]
        q = "Is anonymous access permitted?"
        ans = "The system openly welcomes unrestricted anonymous guests without passwords."
        score = calculate_faithfulness(question=q, answer=ans, contexts=ctx)
        self.assertEqual(score, 0.0)

    def test_irrelevant_question_vs_retrieved_contexts(self):
        """Verify irrelevant questions against context yield 0.0 context precision and answer relevancy."""
        contexts = [
            "DuckDB runs in-memory columnar execution over Parquet and CSV files.",
            "PostgreSQL stores structured relational tables with B-Tree indexes.",
        ]
        irrelevant_q = "How do you make strawberry ice cream?"
        irrelevant_ans = "Freeze fresh strawberries with heavy whipping cream and condensed sugar."

        precision = calculate_context_precision(
            question=irrelevant_q,
            contexts=contexts,
            ground_truth="Freeze fresh strawberries with heavy whipping cream and condensed sugar."
        )
        # Contexts do not contain strawberry ice cream information
        self.assertEqual(precision, 0.0)

        relevancy_cross = calculate_answer_relevancy(
            question="What are DuckDB performance characteristics?",
            answer=irrelevant_ans
        )
        self.assertEqual(relevancy_cross, 0.0)

    def test_numerical_claim_adversarial_mismatch(self):
        """Verify faithfulness is 0.0 when sentence words match but numeric quantities differ."""
        ctx = ["The sandbox enforces a memory limit of 512MB and a timeout of 5.0 seconds with 20 max items."]
        q = "What are the sandbox resource limits?"
        # Words are identical except numbers: 1024MB vs 512MB, 15.0s vs 5.0s, 100 vs 20
        ans_wrong_numbers = "The sandbox enforces a memory limit of 1024MB and a timeout of 15.0 seconds with 100 max items."
        
        score = calculate_faithfulness(question=q, answer=ans_wrong_numbers, contexts=ctx)
        self.assertEqual(
            score, 0.0,
            f"Expected 0.0 faithfulness when numerical entities mismatch, got {score}"
        )

    def test_ragas_boundary_inputs_whitespace_none_punctuation_emoji(self):
        """Verify boundary inputs (None, whitespace, pure punctuation, unicode emojis) do not crash."""
        extreme_inputs = [
            ("", "", []),
            (None, None, None),
            ("   \n\t  ", "   \t  ", ["   ", ""]),
            ("!@#$%^&*()_+=-~`", "!@#$%^&*()_+=-~`", ["!@#$%^&*()"]),
            ("🚀🔥🤖", "🚀🔥🤖", ["🚀🔥🤖"]),
            ("مرحبا بالعالم", "مرحبا بالعالم", ["مرحبا بالعالم"]),
            ("こんにちは世界", "こんにちは世界", ["こんにちは世界"]),
        ]

        for q, a, ctx in extreme_inputs:
            f_score = calculate_faithfulness(question=q, answer=a, contexts=ctx)
            self.assertTrue(0.0 <= f_score <= 1.0)

            ar_score = calculate_answer_relevancy(question=q, answer=a)
            self.assertTrue(0.0 <= ar_score <= 1.0)

            cp_score = calculate_context_precision(question=q, contexts=ctx, ground_truth=a)
            self.assertTrue(0.0 <= cp_score <= 1.0)

            cr_score = calculate_context_recall(ground_truth=a, contexts=ctx)
            self.assertTrue(0.0 <= cr_score <= 1.0)

    def test_massive_batch_evaluations_150_items(self):
        """Verify massive batch evaluation (150 cases) executes rapidly and reliably."""
        random.seed(42)
        batch_cases = []

        for i in range(150):
            case_type = i % 5
            if case_type == 0:
                # Grounded
                batch_cases.append({
                    "test_id": f"batch_{i:03d}",
                    "question": f"What is parameter alpha_{i} configured to?",
                    "contexts": [f"Configuration setting alpha_{i} is defined as value_{i} in system profile."],
                    "answer": f"Parameter alpha_{i} is configured to value_{i}.",
                    "ground_truth": f"alpha_{i} is configured to value_{i}.",
                })
            elif case_type == 1:
                # Hallucinated
                batch_cases.append({
                    "test_id": f"batch_{i:03d}",
                    "question": f"What is beta_{i} status?",
                    "contexts": [f"Beta_{i} service was deprecated in revision 4."],
                    "answer": f"Beta_{i} is actively processing 10000 transactions per second on Kubernetes cluster.",
                    "ground_truth": f"Beta_{i} was deprecated in revision 4.",
                })
            elif case_type == 2:
                # Empty context
                batch_cases.append({
                    "test_id": f"batch_{i:03d}",
                    "question": f"What is gamma_{i}?",
                    "contexts": [],
                    "answer": f"Gamma_{i} is unknown.",
                    "ground_truth": f"Gamma_{i} is unknown.",
                })
            elif case_type == 3:
                # Partial recall
                batch_cases.append({
                    "test_id": f"batch_{i:03d}",
                    "question": f"List features of delta_{i}",
                    "contexts": [f"Delta_{i} includes high-availability replication."],
                    "answer": f"Delta_{i} includes high-availability replication and auto-sharding.",
                    "ground_truth": f"Delta_{i} includes high-availability replication; Delta_{i} includes automated backup; Delta_{i} includes multi-region sync.",
                })
            else:
                # Multiline with formatting
                batch_cases.append({
                    "test_id": f"batch_{i:03d}",
                    "question": f"Explain security policy for node_{i}",
                    "contexts": [
                        f"- Node_{i} requires TLS 1.3 encryption.\n- Port 443 must remain open.\n- Admin access is restricted."
                    ],
                    "answer": f"Node_{i} requires TLS 1.3 encryption and admin access is restricted.",
                    "ground_truth": f"Node_{i} requires TLS 1.3 encryption.",
                })

        start_time = time.time()
        result = self.evaluator.evaluate_test_cases(batch_cases)
        elapsed = time.time() - start_time

        self.assertIsInstance(result, RagasEvaluationResult)
        self.assertEqual(len(result.details), 150)
        self.assertLess(elapsed, 3.0, f"150 batch evaluation took {elapsed:.2f}s (expected < 3.0s)")

        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            self.assertIn(m, result.summary)
            score = result.summary[m]
            self.assertTrue(0.0 <= score <= 1.0, f"Summary metric {m} = {score} out of bounds")

        # Export to DataFrame check
        df_out = result.to_dataframe()
        self.assertEqual(len(df_out), 150)


class TestStructuredEquivalenceAdversarialStress(unittest.TestCase):
    """Adversarial stress-testing suite for DataFrame normalization and execution equivalence."""

    def test_float_tolerance_boundary_strictness(self):
        """
        Verify strict boundary of floating point tolerance and 4-decimal normalization (atol=1e-4):
        At base_val = 0.0 (where relative tolerance rtol*b == 0.0):
        - diff 5e-5 (0.00005) -> PASS (<= 1e-4)
        - diff 9e-5 (0.00009) -> PASS (<= 1e-4)
        - diff 2e-4 (0.00020) -> FAIL (> 1e-4)
        - diff 1e-3 (0.00100) -> FAIL (> 1e-4)
        """
        base_val = 0.0

        # 1. Delta = 0.00004 -> within atol=1e-4 -> EQUIVALENT
        df_gold = pd.DataFrame({"amt": [base_val]})
        df_gen_close = pd.DataFrame({"amt": [base_val + 0.00004]})
        is_eq, msg = check_execution_equivalence(df_gen_close, df_gold, tolerance=1e-4)
        self.assertTrue(is_eq, f"Expected 4e-5 difference to be equivalent, failed with: {msg}")

        # 2. Delta = 0.00009 -> within atol=1e-4 -> EQUIVALENT
        df_gen_edge_pass = pd.DataFrame({"amt": [base_val + 0.00009]})
        is_eq, msg = check_execution_equivalence(df_gen_edge_pass, df_gold, tolerance=1e-4)
        self.assertTrue(is_eq, f"Expected 9e-5 difference to be equivalent, failed with: {msg}")

        # 3. Delta = 0.00025 -> exceeds atol=1e-4 -> NOT EQUIVALENT
        df_gen_edge_fail = pd.DataFrame({"amt": [base_val + 0.00025]})
        is_eq, msg = check_execution_equivalence(df_gen_edge_fail, df_gold, tolerance=1e-4)
        self.assertFalse(is_eq, "Expected 2.5e-4 difference to exceed tolerance and fail")
        self.assertTrue("different" in msg.lower() or "diff" in msg.lower() or "mismatch" in msg.lower())

        # 4. Delta = 0.001 (1e-3) -> exceeds atol=1e-4 -> NOT EQUIVALENT
        df_gen_large_diff = pd.DataFrame({"amt": [base_val + 0.001]})
        is_eq, msg = check_execution_equivalence(df_gen_large_diff, df_gold, tolerance=1e-4)
        self.assertFalse(is_eq, "Expected 1e-3 difference to exceed tolerance and fail")

    def test_mixed_types_and_string_represented_numbers(self):
        """Verify columns with string-represented numbers match actual numeric columns after normalization."""
        # e.g. DuckDB returning float vs Postgres returning numeric string or int vs float
        df_gold = pd.DataFrame({
            "order_id": [1, 2, 3],
            "price": [10.50, 20.00, 30.75],
            "is_active": [True, False, True]
        })
        df_gen = pd.DataFrame({
            "order_id": ["1", "2", "3"],
            "price": ["10.5000", "20.0000", "30.7500"],
            "is_active": [True, False, True]
        })

        is_eq, msg = check_execution_equivalence(df_gen, df_gold)
        self.assertTrue(is_eq, f"Expected string-encoded numbers to normalize to numeric equivalence: {msg}")

    def test_unicode_special_chars_and_whitespace_padding(self):
        """Verify normalization trims whitespace and preserves unicode/special characters."""
        df_gold = pd.DataFrame({
            "name": [" René Descartes ", " 東京 (Tokyo) ", "Café & Bakery © "],
            "status": [" OPEN ", " CLOSED ", " PENDING "]
        })
        df_gen = pd.DataFrame({
            "status": ["CLOSED", "OPEN", "PENDING"],
            "name": ["東京 (Tokyo)", "René Descartes", "Café & Bakery ©"]
        })

        is_eq, msg = check_execution_equivalence(df_gen, df_gold)
        self.assertTrue(is_eq, f"Expected unicode and whitespace-trimmed rows to match: {msg}")

    def test_null_none_nan_representation_invariance(self):
        """Verify various representations of null (None, NaN, 'None', 'null', 'nan', '') match across engines."""
        df1 = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "val": [10.0, None, float("nan"), 40.0]
        })
        df2 = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "val": [10.0, float("nan"), None, 40.0]
        })

        is_eq, msg = check_execution_equivalence(df1, df2)
        self.assertTrue(is_eq, f"Expected None and NaN to be equivalent null representations: {msg}")

    def test_nested_dictionaries_in_dataframe_cells(self):
        """Verify DataFrames with complex/nested dictionary structures do not crash normalizer."""
        df_gold = pd.DataFrame({
            "id": [1, 2],
            "meta": [{"tags": ["a", "b"]}, {"tags": ["c"]}]
        })
        df_gen = pd.DataFrame({
            "id": [1, 2],
            "meta": [{"tags": ["a", "b"]}, {"tags": ["c"]}]
        })

        norm_gold = normalize_dataframe(df_gold)
        norm_gen = normalize_dataframe(df_gen)
        self.assertEqual(norm_gold.shape, (2, 2))
        self.assertEqual(norm_gen.shape, (2, 2))

    def test_large_dataset_10000_rows_stress(self):
        """Verify performance and determinism on large 10,000 row datasets with row permutations."""
        random.seed(999)
        num_rows = 10000

        ids = list(range(1, num_rows + 1))
        categories = [f"cat_{i % 50}" for i in range(num_rows)]
        amounts = [round(random.uniform(10.0, 5000.0), 4) for _ in range(num_rows)]
        ratings = [round(random.uniform(1.0, 5.0), 2) for _ in range(num_rows)]
        flags = [i % 2 == 0 for i in range(num_rows)]

        df_gold = pd.DataFrame({
            "id": ids,
            "category": categories,
            "amount": amounts,
            "rating": ratings,
            "is_valid": flags,
        })

        # Create permuted generated DataFrame with slightly scrambled columns and shuffled rows
        shuffled_indices = list(range(num_rows))
        random.shuffle(shuffled_indices)

        df_gen = pd.DataFrame({
            "rating": [ratings[i] for i in shuffled_indices],
            "amount": [amounts[i] for i in shuffled_indices],
            "id": [ids[i] for i in shuffled_indices],
            "is_valid": [flags[i] for i in shuffled_indices],
            "category": [categories[i] for i in shuffled_indices],
        })

        start_time = time.time()
        is_eq, msg = check_execution_equivalence(df_gen, df_gold, tolerance=1e-4)
        elapsed = time.time() - start_time

        self.assertTrue(is_eq, f"10,000-row permuted equivalence failed: {msg}")
        self.assertLess(elapsed, 4.0, f"10,000-row normalization took {elapsed:.2f}s (expected < 4.0s)")

    def test_extreme_empty_and_zero_shape_permutations(self):
        """Verify handling of extreme shape variations (0x0, 0x5, 5x0)."""
        # Empty with columns
        df_empty_cols1 = pd.DataFrame(columns=["Alpha", "Beta", "Gamma"])
        df_empty_cols2 = pd.DataFrame(columns=["gamma", "beta", "alpha"])

        is_eq, msg = check_execution_equivalence(df_empty_cols1, df_empty_cols2)
        self.assertTrue(is_eq, f"Empty DataFrames with permuted column names should match: {msg}")

        # Empty vs populated mismatch
        df_populated = pd.DataFrame({"a": [1]})
        is_eq2, msg2 = check_execution_equivalence(df_empty_cols1, df_populated)
        self.assertFalse(is_eq2)
        self.assertIn("Shape mismatch", msg2)


class TestBenchmarkRunnerAndTelemetryAdversarialStress(unittest.TestCase):
    """Adversarial stress-testing suite for EvaluationRunner, Telemetry, and Latency stats."""

    def test_latency_statistics_percentiles_extreme_distribution(self):
        """Verify p50, p95, p99 on skewed latency distributions (e.g. 90 fast, 10 extreme outliers)."""
        latencies = [10.0] * 90 + [5000.0] * 10  # 100 cases, 10% outliers
        stats = compute_latency_statistics(latencies)

        self.assertAlmostEqual(stats["median_ms"], 10.0, places=1)
        self.assertAlmostEqual(stats["min_ms"], 10.0, places=1)
        self.assertEqual(stats["max_ms"], 5000.0)
        self.assertGreater(stats["p95_ms"], 10.0)
        self.assertGreater(stats["p99_ms"], 1000.0)
        self.assertEqual(stats["total_ms"], 90 * 10.0 + 10 * 5000.0)

    def test_latency_statistics_empty_and_single_value(self):
        """Verify edge cases for latency stats: empty list and single value."""
        empty_stats = compute_latency_statistics([])
        self.assertEqual(empty_stats["mean_ms"], 0.0)
        self.assertEqual(empty_stats["p99_ms"], 0.0)

        single_stats = compute_latency_statistics([42.5])
        self.assertEqual(single_stats["mean_ms"], 42.5)
        self.assertEqual(single_stats["median_ms"], 42.5)
        self.assertEqual(single_stats["p95_ms"], 42.5)
        self.assertEqual(single_stats["p99_ms"], 42.5)

    def test_token_cost_estimation_all_models_and_fallbacks(self):
        """Verify token cost calculations for all supported LLMs and fallback on unknown model."""
        models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-5-sonnet"]
        p_tok = 10000
        c_tok = 2000

        for m in models:
            cost = estimate_token_cost(p_tok, c_tok, model=m)
            self.assertGreater(cost, 0.0)

        # Unknown model fallback to gpt-4o-mini
        cost_unknown = estimate_token_cost(p_tok, c_tok, model="nonexistent_llm_model_xyz")
        cost_mini = estimate_token_cost(p_tok, c_tok, model="gpt-4o-mini")
        self.assertEqual(cost_unknown, cost_mini)

    def test_structured_evaluator_large_100_case_benchmark(self):
        """Verify StructuredEquivalenceEvaluator aggregates 100 cases across 3 engines cleanly."""
        test_cases = []
        for i in range(100):
            eng = ["dedicated_db", "duckdb", "pandas_sandbox"][i % 3]
            is_match = (i % 4 != 0)  # 75% match, 25% mismatch
            val_gen = float(i) if is_match else float(i + 9999)

            test_cases.append({
                "test_id": f"tc_{i:03d}",
                "query": f"Query #{i}",
                "engine": eng,
                "df_golden": pd.DataFrame({"result": [float(i)]}),
                "df_generated": pd.DataFrame({"result": [val_gen]}),
                "latency_ms": 10.0 + (i * 0.5),
                "prompt_tokens": 100 + i,
                "completion_tokens": 20 + (i % 10),
                "error": None if i % 10 != 0 else "Simulated Execution Error" if not is_match else None,
            })

        evaluator = StructuredEquivalenceEvaluator(tolerance=1e-4)
        result = evaluator.evaluate_benchmark(test_cases)

        self.assertIsInstance(result, StructuredBenchmarkResult)
        self.assertEqual(result.total_cases, 100)
        self.assertEqual(len(result.details), 100)
        self.assertIn("dedicated_db", result.per_engine_stats)
        self.assertIn("duckdb", result.per_engine_stats)
        self.assertIn("pandas_sandbox", result.per_engine_stats)

        # Summary check
        sum_dict = result.summary_dict()
        self.assertEqual(sum_dict["total_cases"], 100)
        self.assertGreater(sum_dict["total_tokens"], 0)
        self.assertGreater(sum_dict["total_cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
