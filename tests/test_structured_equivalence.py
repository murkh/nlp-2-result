"""
Unit and Integration Tests for Structured Ground-Truth Execution Equivalence Suite (Milestone 4).
Verifies:
  - DataFrame normalization (column casing, column sorting, float rounding, row sorting, null handling)
  - Execution equivalence checking (exact, column-permuted, row-permuted, float-tolerance, shape/value mismatch detection)
  - Syntax first-pass success rate calculation
  - Equivalence rate calculation
  - Latency statistics breakdown (mean, p50, p95, p99)
  - Token cost estimation
  - StructuredEquivalenceEvaluator benchmark harness
"""

import unittest
from src.evaluation.compat import pd, np

from src.evaluation.structured_equivalence import (
    StructuredEquivalenceEvaluator,
    StructuredBenchmarkResult,
    assert_frame_equivalence,
    calculate_equivalence_rate,
    calculate_syntax_first_pass_rate,
    check_execution_equivalence,
    compute_latency_statistics,
    estimate_token_cost,
    normalize_dataframe,
)


class TestStructuredEquivalence(unittest.TestCase):
    """Test suite for DataFrame normalization and execution equivalence evaluation."""

    def test_normalize_dataframe_column_casing_and_order(self):
        """Verify column names are stripped, lowercased, and sorted alphabetically."""
        df = pd.DataFrame({
            " Total_Amount ": [100.5, 200.0],
            "Customer_ID": [501, 502],
            " STATUS ": [" completed ", "shipped "]
        })
        norm = normalize_dataframe(df)
        expected_cols = ["customer_id", "status", "total_amount"]
        self.assertEqual(list(norm.columns), expected_cols)
        self.assertEqual(norm["status"].tolist(), ["completed", "shipped"])

    def test_normalize_dataframe_float_rounding(self):
        """Verify float numbers are rounded to 4 decimal places."""
        df = pd.DataFrame({
            "val": [3.14159265, 2.71828182, 1.0]
        })
        norm = normalize_dataframe(df)
        # Note: row sorting orders rows ascending across columns
        self.assertEqual(norm["val"].tolist(), [1.0, 2.7183, 3.1416])

    def test_normalize_dataframe_row_sorting_order_independence(self):
        """Verify rows are sorted across columns for order-independent comparison."""
        df1 = pd.DataFrame({"id": [2, 1, 3], "val": ["b", "a", "c"]})
        df2 = pd.DataFrame({"id": [1, 3, 2], "val": ["a", "c", "b"]})

        norm1 = normalize_dataframe(df1)
        norm2 = normalize_dataframe(df2)

        pd.testing.assert_frame_equal(norm1, norm2)

    def test_normalize_dataframe_empty_and_none_handling(self):
        """Verify None or empty DataFrames return valid empty DataFrames without error."""
        norm_none = normalize_dataframe(None)
        self.assertTrue(norm_none.empty)

        norm_empty = normalize_dataframe(pd.DataFrame())
        self.assertTrue(norm_empty.empty)

    def test_execution_equivalence_exact_match(self):
        """Verify identical DataFrames are evaluated as equivalent."""
        df_gold = pd.DataFrame({"region": ["North", "South"], "sales": [560.0, 1070.5]})
        df_gen = pd.DataFrame({"region": ["North", "South"], "sales": [560.0, 1070.5]})

        is_equiv, msg = check_execution_equivalence(df_gen, df_gold)
        self.assertTrue(is_equiv)
        self.assertEqual(msg, "Execution Equivalent")

    def test_execution_equivalence_column_permuted_order(self):
        """Verify equivalence when columns are in different orders."""
        df_gold = pd.DataFrame({"a": [1, 2], "b": [10.0, 20.0]})
        df_gen = pd.DataFrame({"b": [10.0, 20.0], "a": [1, 2]})

        is_equiv, _ = check_execution_equivalence(df_gen, df_gold)
        self.assertTrue(is_equiv)

    def test_execution_equivalence_float_tolerance_within_1e4(self):
        """Verify equivalence tolerates floating-point variations within 1e-4."""
        df_gold = pd.DataFrame({"amount": [83.5404]})
        df_gen = pd.DataFrame({"amount": [83.54038]})

        is_equiv, _ = check_execution_equivalence(df_gen, df_gold, tolerance=1e-4)
        self.assertTrue(is_equiv)

    def test_execution_equivalence_shape_mismatch_detected(self):
        """Verify shape mismatch is caught and reported."""
        df_gold = pd.DataFrame({"val": [1, 2, 3]})
        df_gen = pd.DataFrame({"val": [1, 2]})

        is_equiv, msg = check_execution_equivalence(df_gen, df_gold)
        self.assertFalse(is_equiv)
        self.assertIn("Shape mismatch", msg)

    def test_execution_equivalence_column_mismatch_detected(self):
        """Verify missing/different column schemas are detected."""
        df_gold = pd.DataFrame({"total": [100.0]})
        df_gen = pd.DataFrame({"sum": [100.0]})

        is_equiv, msg = check_execution_equivalence(df_gen, df_gold)
        self.assertFalse(is_equiv)
        self.assertIn("Column mismatch", msg)

    def test_execution_equivalence_value_mismatch_detected(self):
        """Verify value discrepancies beyond tolerance are detected."""
        df_gold = pd.DataFrame({"val": [1.0, 2.0]})
        df_gen = pd.DataFrame({"val": [1.0, 3.0]})

        is_equiv, msg = check_execution_equivalence(df_gen, df_gold)
        self.assertFalse(is_equiv)
        self.assertTrue("different" in msg.lower())

    def test_assert_frame_equivalence_raises_assertion_error(self):
        """Verify assert_frame_equivalence raises AssertionError on mismatch."""
        df_gold = pd.DataFrame({"val": [1, 2]})
        df_gen = pd.DataFrame({"val": [1, 5]})

        with self.assertRaises(AssertionError):
            assert_frame_equivalence(df_gen, df_gold)

    def test_syntax_first_pass_rate_metric(self):
        """Verify first-pass syntax rate calculation."""
        attempts = [True, True, True, False, True]  # 4 out of 5
        rate = calculate_syntax_first_pass_rate(attempts)
        self.assertEqual(rate, 0.8)

        # Using result dicts
        dict_attempts = [
            {"status": "SUCCESS", "error": None},
            {"status": "SUCCESS", "error": None},
            {"status": "FAILED", "error": "SyntaxError: near SELECT"},
            {"status": "SUCCESS", "error": None},
        ]
        self.assertEqual(calculate_syntax_first_pass_rate(dict_attempts), 0.75)

    def test_equivalence_rate_metric(self):
        """Verify equivalence rate calculation."""
        results = [True, True, False, True]
        self.assertEqual(calculate_equivalence_rate(results), 0.75)

    def test_latency_statistics_breakdown(self):
        """Verify latency statistics computation."""
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_latency_statistics(latencies)
        self.assertEqual(stats["mean_ms"], 30.0)
        self.assertEqual(stats["median_ms"], 30.0)
        self.assertEqual(stats["min_ms"], 10.0)
        self.assertEqual(stats["max_ms"], 50.0)
        self.assertEqual(stats["total_ms"], 150.0)

    def test_token_cost_estimation(self):
        """Verify token cost calculation across models."""
        cost_mini = estimate_token_cost(prompt_tokens=1000, completion_tokens=500, model="gpt-4o-mini")
        self.assertGreater(cost_mini, 0.0)

        cost_4o = estimate_token_cost(prompt_tokens=1000, completion_tokens=500, model="gpt-4o")
        self.assertGreater(cost_4o, cost_mini)

    def test_structured_evaluator_benchmark_suite(self):
        """Verify StructuredEquivalenceEvaluator benchmark aggregation."""
        test_cases = [
            {
                "test_id": "tc1",
                "query": "Select all orders",
                "engine": "dedicated_db",
                "df_golden": pd.DataFrame({"id": [101, 102], "amount": [150.5, 280.0]}),
                "df_generated": pd.DataFrame({"id": [101, 102], "amount": [150.5, 280.0]}),
                "latency_ms": 25.0,
                "prompt_tokens": 100,
                "completion_tokens": 30,
            },
            {
                "test_id": "tc2",
                "query": "Count completed orders",
                "engine": "duckdb",
                "df_golden": pd.DataFrame({"count": [2]}),
                "df_generated": pd.DataFrame({"count": [2]}),
                "latency_ms": 15.0,
                "prompt_tokens": 80,
                "completion_tokens": 20,
            },
            {
                "test_id": "tc3",
                "query": "Average order amount",
                "engine": "pandas_sandbox",
                "df_golden": pd.DataFrame({"avg": [215.25]}),
                "df_generated": pd.DataFrame({"avg": [999.99]}),  # Mismatch
                "latency_ms": 40.0,
                "prompt_tokens": 120,
                "completion_tokens": 40,
            },
        ]

        evaluator = StructuredEquivalenceEvaluator(tolerance=1e-4)
        result = evaluator.evaluate_benchmark(test_cases)

        self.assertIsInstance(result, StructuredBenchmarkResult)
        self.assertEqual(result.total_cases, 3)
        self.assertEqual(result.syntax_first_pass_rate, 1.0)
        self.assertAlmostEqual(result.equivalence_rate, 2 / 3, places=2)
        self.assertIn("dedicated_db", result.per_engine_stats)
        self.assertIn("duckdb", result.per_engine_stats)
        self.assertIn("pandas_sandbox", result.per_engine_stats)

        # Check export
        df_report = result.to_dataframe()
        self.assertEqual(len(df_report), 3)


if __name__ == "__main__":
    unittest.main()
