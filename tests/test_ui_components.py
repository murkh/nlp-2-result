"""
Unit and Integration Tests for Frontend UI Components and Backend API Client.
Verifies BackendClient API methods, payload serialization, error handling,
tabular dataframe converters, and telemetry formatting.
"""

import unittest
from unittest.mock import MagicMock, patch

from frontend.client import BackendClient
from src.evaluation.compat import pd


class TestBackendClient(unittest.TestCase):
    """Test suite for Frontend BackendClient."""

    def setUp(self):
        self.base_url = "http://testserver:8000"
        self.client = BackendClient(base_url=self.base_url, timeout=5.0)

    def test_client_initialization_and_url_normalization(self):
        """Verify client strips trailing slashes and sets timeout."""
        c = BackendClient(base_url="http://localhost:8000///", timeout=12.5)
        self.assertEqual(c.base_url, "http://localhost:8000")
        self.assertEqual(c.timeout, 12.5)

    def test_health_success(self):
        """Verify health check returns parsed JSON on 200 OK."""
        mock_data = {"status": "healthy", "service": "multiagent-knowledge-qa", "version": "0.1.0"}
        with patch.object(self.client, "_get", return_value=mock_data):
            res = self.client.health()
            self.assertEqual(res.get("status"), "healthy")
            self.assertEqual(res.get("version"), "0.1.0")

    def test_health_unreachable_handling(self):
        """Verify health check returns error dict when backend connection fails."""
        with patch.object(self.client, "_get", side_effect=Exception("Connection refused")):
            res = self.client.health()
            self.assertEqual(res.get("status"), "unreachable")
            self.assertIn("Connection refused", res.get("error", ""))

    def test_list_datasets_with_and_without_category(self):
        """Verify list_datasets queries /datasets endpoint with proper query params."""
        mock_data = {
            "datasets": [
                {"id": "ds-1", "name": "sales.csv", "category": "structured"},
                {"id": "ds-2", "name": "handbook.pdf", "category": "unstructured"},
            ]
        }

        with patch.object(self.client, "_get", return_value=mock_data) as mock_get:
            # Without filter
            datasets = self.client.list_datasets()
            self.assertEqual(len(datasets), 2)
            mock_get.assert_called_with("/datasets", params={})

            # With category filter
            struct_datasets = self.client.list_datasets(category="structured")
            mock_get.assert_called_with("/datasets", params={"category": "structured"})

    def test_get_dataset_success_and_not_found(self):
        """Verify get_dataset retrieves dataset by ID and handles 404."""
        mock_ok = {"id": "ds-123", "name": "orders.parquet"}

        with patch.object(self.client, "_get", return_value=mock_ok):
            res = self.client.get_dataset("ds-123")
            self.assertEqual(res.get("id"), "ds-123")
            self.assertEqual(res.get("name"), "orders.parquet")

        mock_err = {"error": "Dataset not found", "status_code": 404}

        with patch.object(self.client, "_get", return_value=mock_err):
            res = self.client.get_dataset("non-existent")
            self.assertIn("error", res)
            self.assertEqual(res.get("status_code"), 404)

    def test_ingest_file_multipart_post(self):
        """Verify ingest_file sends multipart payload."""
        mock_response = {
            "dataset_id": "new-uuid",
            "name": "sales_q3.csv",
            "file_type": ".csv",
            "category": "structured",
            "row_count": 150,
            "message": "Ingestion successful",
        }

        with patch.object(self.client, "_post", return_value=mock_response) as mock_post:
            res = self.client.ingest_file(
                file_bytes=b"col1,col2\n1,2",
                filename="sales_q3.csv",
                display_name="Q3 Sales",
                description="Quarterly sales data",
            )
            self.assertEqual(res.get("dataset_id"), "new-uuid")
            self.assertEqual(res.get("category"), "structured")
            self.assertEqual(res.get("row_count"), 150)
            mock_post.assert_called_once()

    def test_query_agent_request_payload_construction(self):
        """Verify query_agent serializes session_id, strategy, and dataset_ids."""
        mock_response = {
            "query": "What are total sales?",
            "session_id": "sess-42",
            "intent": "STRUCTURED_QUERY",
            "confidence": 0.98,
            "answer": "Total sales are $1,250,000.",
            "suggested_strategy": "duckdb",
            "tabular_result": {"columns": ["total_sales"], "rows": [{"total_sales": 1250000}]},
            "metrics": {"total_latency_ms": 120.5},
            "token_usage": {"prompt_tokens": 150, "completion_tokens": 45, "total_tokens": 195},
        }

        with patch.object(self.client, "_post", return_value=mock_response) as mock_post:
            res = self.client.query_agent(
                query="What are total sales?",
                session_id="sess-42",
                suggested_strategy="duckdb",
                dataset_ids=["ds-1"],
                temperature=0.2,
            )
            self.assertEqual(res.get("intent"), "STRUCTURED_QUERY")
            self.assertEqual(res.get("answer"), "Total sales are $1,250,000.")
            mock_post.assert_called_with(
                "/query/agent",
                json_data={
                    "query": "What are total sales?",
                    "temperature": 0.2,
                    "session_id": "sess-42",
                    "suggested_strategy": "duckdb",
                    "dataset_ids": ["ds-1"],
                },
            )

    def test_query_dedicated_db_and_duckdb_and_pandas(self):
        """Verify individual structured engine API query methods."""
        mock_db_res = {
            "query": "Count users",
            "answer": "There are 50 users.",
            "sql_query": "SELECT count(*) FROM tbl_users LIMIT 20;",
            "tabular_result": {"columns": ["count"], "rows": [{"count": 50}]},
        }

        with patch.object(self.client, "_post", return_value=mock_db_res):
            res_a = self.client.query_dedicated_db("Count users")
            self.assertEqual(res_a.get("answer"), "There are 50 users.")
            self.assertEqual(res_a.get("sql_query"), "SELECT count(*) FROM tbl_users LIMIT 20;")

            res_b = self.client.query_duckdb("Count users")
            self.assertEqual(res_b.get("answer"), "There are 50 users.")

        mock_pandas_res = {
            "query": "Count users",
            "answer": "There are 50 users.",
            "python_code": "df.shape[0]",
            "security_report": {"ast_passed": True, "violations": []},
        }

        with patch.object(self.client, "_post", return_value=mock_pandas_res):
            res_c = self.client.query_pandas_sandbox("Count users")
            self.assertEqual(res_c.get("python_code"), "df.shape[0]")
            self.assertTrue(res_c.get("security_report", {}).get("ast_passed"))

    def test_query_unstructured_rag_and_citations(self):
        """Verify unstructured hybrid RAG method and citation extraction."""
        mock_rag_res = {
            "query": "What is company policy?",
            "answer": "Vacation policy allows 20 days [Doc: handbook.pdf, Page 3].",
            "citations": [
                {
                    "document_name": "handbook.pdf",
                    "page_number": 3,
                    "chunk_index": 0,
                    "similarity_score": 0.895,
                    "snippet": "Full-time employees receive 20 days vacation.",
                }
            ],
        }

        with patch.object(self.client, "_post", return_value=mock_rag_res):
            res = self.client.query_unstructured_rag("What is company policy?", top_k=3)
            self.assertEqual(len(res.get("citations", [])), 1)
            cit = res["citations"][0]
            self.assertEqual(cit.get("document_name"), "handbook.pdf")
            self.assertEqual(cit.get("page_number"), 3)

    def test_query_benchmark_response_structure(self):
        """Verify 3-way benchmark query method returns complete arena structure."""
        mock_bench_res = {
            "query": "Total revenue",
            "strategy_a": {"strategy_name": "Strategy A", "status": "SUCCESS", "answer": "1000"},
            "strategy_b": {"strategy_name": "Strategy B", "status": "SUCCESS", "answer": "1000"},
            "strategy_c": {"strategy_name": "Strategy C", "status": "SUCCESS", "answer": "1000"},
            "benchmark_summary": {
                "fastest_strategy": "Strategy B",
                "most_token_efficient_strategy": "Strategy B",
                "consensus_reached": True,
            },
            "total_arena_latency_ms": 145.2,
        }

        with patch.object(self.client, "_post", return_value=mock_bench_res):
            res = self.client.query_benchmark("Total revenue")
            self.assertTrue(res.get("benchmark_summary", {}).get("consensus_reached"))
            self.assertEqual(res.get("benchmark_summary", {}).get("fastest_strategy"), "Strategy B")
            self.assertEqual(res.get("total_arena_latency_ms"), 145.2)

    def test_tabular_result_to_dataframe_conversion(self):
        """Verify tabular_result_to_dataframe correctly converts dicts to DataFrames."""
        # Empty inputs
        self.assertTrue(self.client.tabular_result_to_dataframe(None).empty)
        self.assertTrue(self.client.tabular_result_to_dataframe({}).empty)

        # Columns with rows
        raw_tab = {
            "columns": ["id", "amount", "status"],
            "rows": [
                {"id": 1, "amount": 100.5, "status": "COMPLETED"},
                {"id": 2, "amount": 250.0, "status": "PENDING"},
            ],
            "row_count": 2,
            "truncated": False,
        }
        df = self.client.tabular_result_to_dataframe(raw_tab)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["id", "amount", "status"])

    def test_cost_calculation_and_latency_formatting(self):
        """Verify cost and latency helper computations."""
        # Token cost calculation ($0.15/1M prompt, $0.60/1M completion)
        cost = self.client.calculate_cost(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        self.assertAlmostEqual(cost, 0.75, places=5)

        cost_zero = self.client.calculate_cost(0, 0)
        self.assertEqual(cost_zero, 0.0)

        # Latency formatting
        self.assertEqual(self.client.format_latency(50.4), "50.4 ms")
        self.assertEqual(self.client.format_latency(1250.0), "1.25 s")
        self.assertEqual(self.client.format_latency(3500.0), "3.50 s")

    def test_client_retry_mechanism(self):
        """Verify client retries on transient connection failures."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        # Simulate 2 failures then success
        call_count = 0

        def fake_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary glitch")
            return mock_response

        with patch("httpx.Client.get", side_effect=fake_get):
            client = BackendClient(base_url="http://localhost:8000", max_retries=3)
            res = client.health()
            self.assertEqual(res.get("status"), "healthy")
            self.assertEqual(call_count, 3)

    def test_tabular_result_dataframe_column_only_and_object_rows(self):
        """Verify tabular_result_to_dataframe handles column-only and object rows."""
        # Only columns
        df_cols = self.client.tabular_result_to_dataframe({"columns": ["a", "b"]})
        self.assertTrue(df_cols.empty)
        self.assertEqual(list(df_cols.columns), ["a", "b"])

        # Object with rows attribute
        class MockTabular:
            rows = [{"col1": 10, "col2": 20}]

        df_obj = self.client.tabular_result_to_dataframe(MockTabular())
        self.assertFalse(df_obj.empty)
        self.assertEqual(len(df_obj), 1)


class TestUIHelpers(unittest.TestCase):
    """Test suite for UI rendering functions."""

    def test_render_model_thinking_execution(self):
        """Verify render_model_thinking executes cleanly with mock Streamlit."""
        from frontend.ui import render_model_thinking

        sample_thinking = {
            "summary": "Processed with DuckDB",
            "steps": [
                {
                    "step_number": 1,
                    "title": "Intent Classification",
                    "choice": "STRUCTURED_QUERY",
                    "reasoning": "Detected SQL keywords",
                    "details": {"confidence": 0.98},
                },
                {
                    "step_number": 2,
                    "title": "Schema Pruning",
                    "choice": "Selected orders table",
                    "reasoning": "Filtered irrelevant columns",
                    "details": {"retained_columns": {"orders": ["order_id", "total_amount"]}},
                },
            ],
        }

        with (
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.info") as mock_info,
            patch("streamlit.write") as mock_write,
            patch("streamlit.divider") as mock_div,
        ):
            mock_expander.return_value.__enter__.return_value = MagicMock()

            # Should run without error
            render_model_thinking(sample_thinking, default_expanded=True)
            mock_expander.assert_called_once()

            # None / Empty should safely no-op
            render_model_thinking(None)
            render_model_thinking({})


if __name__ == "__main__":
    unittest.main()
