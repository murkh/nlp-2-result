"""
Verifies that the structured engine hard-fails visibly when no LLM is configured.
There is no deterministic fallback generator: a missing or broken LLM must
surface as error= on the response and an ERROR row in query_logs.
"""

import unittest

from src.api.schemas import QueryPandasSandboxRequest
from src.config import Settings
from tests.conftest import SAMPLE_CSV_TEXT, create_test_fixtures


class TestLLMUnavailableSurfacing(unittest.TestCase):
    """The structured engine fails loudly without an LLM client."""

    def setUp(self):
        fixtures = create_test_fixtures()
        self.temp_dir = fixtures["temp_dir"]
        self.db_manager = fixtures["test_db"]
        self.schema_pruner = fixtures["schema_pruner"]
        self.no_llm_settings = Settings(openai_api_key="", embedding_provider="mock")

        fixtures["structured_engine"].ingest_file(
            file_input=SAMPLE_CSV_TEXT,
            filename="orders.csv",
            display_name="Customer Orders",
            description="Table containing e-commerce customer orders and totals.",
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _error_logs(self):
        cols, rows = self.db_manager.execute_sql_query(
            "SELECT engine, status, error_message FROM query_logs WHERE status = 'ERROR'"
        )
        return [dict(zip(cols, r)) for r in rows]

    def _assert_failed_visibly(self, resp, engine_name: str):
        self.assertIsNotNone(resp.error)
        self.assertIn("LLM", resp.error)
        self.assertEqual(resp.tabular_result.row_count, 0)

        logged = [row for row in self._error_logs() if row["engine"] == engine_name]
        self.assertEqual(len(logged), 1)
        self.assertIn("LLM", logged[0]["error_message"])

    def test_pandas_sandbox_reports_missing_llm(self):
        from src.engines.pandas_sandbox.engine import PandasSandboxEngine

        engine = PandasSandboxEngine(
            db_manager=self.db_manager,
            schema_pruner=self.schema_pruner,
            settings=self.no_llm_settings,
        )
        resp = engine.execute_query(
            QueryPandasSandboxRequest(query="How many orders are completed?")
        )
        self._assert_failed_visibly(resp, "pandas_sandbox")
        self.assertEqual(resp.python_code, "")


if __name__ == "__main__":
    unittest.main()
