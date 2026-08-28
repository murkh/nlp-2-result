"""
Unit and Integration Tests for Benchmark Arena Engine.
Verifies parallel 3-way execution (PostgreSQL, DuckDB, Pandas Sandbox),
ground-truth execution equivalence verification, and telemetry comparison.
"""

from pathlib import Path
import unittest

from src.api.schemas import QueryBenchmarkRequest, TabularResult
from src.engines.benchmark_arena import (
    BenchmarkArenaEngine,
    are_values_equivalent,
    compare_tabular_results,
)
from tests.conftest import create_test_fixtures


class TestBenchmarkArena(unittest.TestCase):
    """Test suite for BenchmarkArenaEngine."""

    def setUp(self):
        fixtures = create_test_fixtures()
        self.temp_dir = fixtures["temp_dir"]
        self.db_manager = fixtures["test_db"]
        self.blob_manager = fixtures["blob_manager"]
        self.structured_engine = fixtures["structured_engine"]
        self.schema_pruner = fixtures["schema_pruner"]

        sample_csv = (
            "order_id,customer_id,order_date,status,total_amount,shipping_city\n"
            "101,501,2024-01-10 10:00:00,completed,150.50,New York\n"
            "102,502,2024-01-11 11:30:00,completed,280.00,San Francisco\n"
            "103,501,2024-01-12 14:15:00,shipped,75.25,New York\n"
            "104,503,2024-01-13 09:45:00,cancelled,45.00,Chicago\n"
            "105,504,2024-01-14 16:20:00,completed,510.80,Austin\n"
        )
        self.dataset_rec = self.structured_engine.ingest_file(
            file_input=sample_csv,
            filename="orders.csv",
            display_name="Orders",
            description="E-commerce orders dataset for 3-way benchmarking.",
        )

        self.settings = fixtures.get("settings")
        self.arena = BenchmarkArenaEngine(
            db_manager=self.db_manager,
            blob_manager=self.blob_manager,
            schema_pruner=self.schema_pruner,
            settings=self.settings,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_value_and_tabular_equivalence(self):
        """Verify mathematical and tabular equivalence functions."""
        # Float tolerance
        self.assertTrue(are_values_equivalent(12.340001, 12.340002, tolerance=1e-4))
        self.assertTrue(are_values_equivalent("completed", "COMPLETED"))
        self.assertTrue(are_values_equivalent(None, None))
        self.assertFalse(are_values_equivalent(12.34, 15.00))

        # Tabular result comparison
        res1 = TabularResult(columns=["status", "cnt"], rows=[{"status": "completed", "cnt": 3}])
        res2 = TabularResult(columns=["status", "cnt"], rows=[{"status": "completed", "cnt": 3.0}])
        res3 = TabularResult(columns=["status", "cnt"], rows=[{"status": "completed", "cnt": 4}])

        self.assertTrue(compare_tabular_results(res1, res2))
        self.assertFalse(compare_tabular_results(res1, res3))

    def test_parallel_benchmark_count_query(self):
        """Verify parallel execution of Strategy A, B, and C on count query."""
        req = QueryBenchmarkRequest(query="How many total orders are in the database?")
        resp = self.arena.execute_benchmark(req)

        self.assertEqual(resp.strategy_a.status, "SUCCESS")
        self.assertEqual(resp.strategy_b.status, "SUCCESS")
        self.assertEqual(resp.strategy_c.status, "SUCCESS")

        self.assertTrue(resp.benchmark_summary.consensus_reached)
        self.assertIn("Benchmark Arena Analysis", resp.benchmark_summary.summary_analysis)
        self.assertGreater(resp.total_arena_latency_ms, 0)
        self.assertIsNotNone(resp.benchmark_summary.fastest_strategy)
        self.assertIsNotNone(resp.benchmark_summary.most_token_efficient_strategy)

    def test_parallel_benchmark_sum_query(self):
        """Verify parallel execution on revenue sum query with equivalence check."""
        req = QueryBenchmarkRequest(query="What is the total sales revenue?")
        resp = self.arena.execute_benchmark(req)

        self.assertEqual(resp.strategy_a.status, "SUCCESS")
        self.assertEqual(resp.strategy_b.status, "SUCCESS")
        self.assertEqual(resp.strategy_c.status, "SUCCESS")

        val_a = float(resp.strategy_a.tabular_result.rows[0]["total_revenue"])
        val_b = float(resp.strategy_b.tabular_result.rows[0]["total_revenue"])
        val_c = float(resp.strategy_c.tabular_result.rows[0]["total_revenue"])

        self.assertAlmostEqual(val_a, 1061.55, places=2)
        self.assertAlmostEqual(val_b, 1061.55, places=2)
        self.assertAlmostEqual(val_c, 1061.55, places=2)
        self.assertTrue(resp.benchmark_summary.consensus_reached)


if __name__ == "__main__":
    unittest.main()
