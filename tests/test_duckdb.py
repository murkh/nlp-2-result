"""
Unit and Integration Tests for Strategy B: DuckDB In-Memory Engine over Blob Files.
Verifies view registration, analytical query execution, LIMIT 20 enforcement,
and security PRAGMA validation.
"""

import unittest
from pathlib import Path

from src.api.schemas import QueryDuckDBRequest
from tests.conftest import create_test_fixtures


class TestDuckDBQueryEngine(unittest.TestCase):
    """Test suite for DuckDBQueryEngine."""

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
            filename="sales_orders.csv",
            display_name="Sales Orders",
            description="Blob storage dataset for DuckDB queries.",
        )

        from src.engines.duckdb_engine import DuckDBQueryEngine

        self.settings = fixtures.get("settings")
        self.engine = DuckDBQueryEngine(
            db_manager=self.db_manager,
            schema_pruner=self.schema_pruner,
            settings=self.settings,
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_duckdb_count_query(self):
        """Verify DuckDB counts rows directly over blob storage files."""
        req = QueryDuckDBRequest(query="How many total sales orders exist?")
        resp = self.engine.execute_query(req)

        self.assertIsNone(resp.error)
        self.assertEqual(resp.tabular_result.row_count, 1)
        self.assertEqual(int(resp.tabular_result.rows[0]["total_records"]), 5)
        self.assertIn("5", resp.answer)
        self.assertGreater(resp.metrics.total_latency_ms, 0)

    def test_duckdb_sum_query(self):
        """Verify DuckDB calculates total revenue over blob files."""
        req = QueryDuckDBRequest(query="What is the total sales revenue?")
        resp = self.engine.execute_query(req)

        self.assertIsNone(resp.error)
        self.assertEqual(resp.tabular_result.row_count, 1)
        total_val = float(resp.tabular_result.rows[0]["total_revenue"])
        self.assertAlmostEqual(total_val, 1061.55, places=2)

    def test_duckdb_limit_enforcement(self):
        """Verify DuckDB automatically caps queries to LIMIT 20."""
        req = QueryDuckDBRequest(query="Select all records from sales orders")
        resp = self.engine.execute_query(req)

        self.assertIsNone(resp.error)
        self.assertIn("LIMIT 20", resp.sql_query.upper())
        self.assertLessEqual(resp.tabular_result.row_count, 20)

    def test_duckdb_security_rejection(self):
        """Verify DuckDB rejects administrative and destructive SQL statements."""
        from src.engines.duckdb_engine import validate_duckdb_security

        self.assertFalse(validate_duckdb_security("ATTACH '/etc/passwd' AS shadow;")[0])
        self.assertFalse(validate_duckdb_security("COPY tbl_orders TO '/tmp/leak.csv';")[0])
        self.assertFalse(validate_duckdb_security("EXPORT DATABASE '/tmp/db';")[0])
        self.assertFalse(validate_duckdb_security("DROP TABLE sales;")[0])
        self.assertTrue(validate_duckdb_security("SELECT * FROM sales_orders LIMIT 20;")[0])


if __name__ == "__main__":
    unittest.main()
