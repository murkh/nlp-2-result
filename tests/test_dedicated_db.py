"""
Unit and Integration Tests for Strategy A: Dedicated PostgreSQL Text2SQL Engine.
Verifies query generation, read-only execution, LIMIT 20 enforcement,
destructive query rejection, and response synthesis.
"""

from pathlib import Path
import unittest

from src.api.schemas import QueryDedicatedDBRequest
from tests.conftest import create_test_fixtures


class TestDedicatedDBEngine(unittest.TestCase):
    """Test suite for DedicatedDBEngine."""

    def setUp(self):
        fixtures = create_test_fixtures()
        self.temp_dir = fixtures["temp_dir"]
        self.db_manager = fixtures["test_db"]
        self.blob_manager = fixtures["blob_manager"]
        self.structured_engine = fixtures["structured_engine"]
        self.schema_pruner = fixtures["schema_pruner"]

        # Ingest sample CSV datasets
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
            display_name="Customer Orders",
            description="Table containing e-commerce customer orders and totals.",
        )

        from src.engines.dedicated_db import DedicatedDBEngine
        self.settings = fixtures.get("settings")
        self.engine = DedicatedDBEngine(
            db_manager=self.db_manager,
            schema_pruner=self.schema_pruner,
            settings=self.settings,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_count_query_execution(self):
        """Verify counting orders executes correctly and returns scalar count."""
        req = QueryDedicatedDBRequest(query="How many total orders are in the database?")
        resp = self.engine.execute_query(req)

        self.assertIsNone(resp.error)
        self.assertEqual(resp.tabular_result.row_count, 1)
        self.assertIn("total_records", resp.tabular_result.columns)
        self.assertEqual(resp.tabular_result.rows[0]["total_records"], 5)
        self.assertIn("5", resp.answer)
        self.assertGreater(resp.metrics.total_latency_ms, 0)
        self.assertGreater(resp.token_usage.total_tokens, 0)

    def test_filtered_aggregation_query(self):
        """Verify filtered query for completed orders."""
        req = QueryDedicatedDBRequest(query="How many orders are completed?")
        resp = self.engine.execute_query(req)

        self.assertIsNone(resp.error)
        self.assertEqual(resp.tabular_result.row_count, 1)
        self.assertEqual(resp.tabular_result.rows[0]["completed_count"], 3)
        self.assertIn("3", resp.answer)

    def test_sum_revenue_query(self):
        """Verify total sales / revenue aggregation query."""
        req = QueryDedicatedDBRequest(query="What is the total sales revenue?")
        resp = self.engine.execute_query(req)

        self.assertIsNone(resp.error)
        self.assertEqual(resp.tabular_result.row_count, 1)
        total_val = float(resp.tabular_result.rows[0]["total_revenue"])
        self.assertAlmostEqual(total_val, 1061.55, places=2)
        self.assertIn("1,061.55", resp.answer)

    def test_limit_20_enforcement(self):
        """Verify that queries automatically enforce LIMIT 20."""
        req = QueryDedicatedDBRequest(query="Show me all order records")
        resp = self.engine.execute_query(req)

        self.assertIsNone(resp.error)
        self.assertIn("LIMIT 20", resp.sql_query.upper())
        self.assertLessEqual(resp.tabular_result.row_count, 20)

    def test_security_rejection_for_destructive_sql(self):
        """Verify that destructive SQL statements (DROP, DELETE, UPDATE) are blocked."""
        from src.engines.dedicated_db import validate_sql_security
        self.assertFalse(validate_sql_security("DROP TABLE tbl_orders;")[0])
        self.assertFalse(validate_sql_security("DELETE FROM tbl_orders WHERE id = 1;")[0])
        self.assertFalse(validate_sql_security("UPDATE tbl_orders SET status = 'hack';")[0])
        self.assertFalse(validate_sql_security("TRUNCATE tbl_orders;")[0])
        self.assertTrue(validate_sql_security("SELECT * FROM tbl_orders LIMIT 20;")[0])


if __name__ == "__main__":
    unittest.main()
