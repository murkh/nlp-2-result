"""
Unit and Integration Tests for FastAPI API Routes.
Tests /health, /ingest, /datasets, and the execution query endpoints:
- POST /query/pandas-sandbox
- POST /query/unstructured-rag
- POST /query/agent
"""

import asyncio
import tempfile
import unittest
from pathlib import Path

from src.api.routes import agent, ingest, query
from src.api.schemas import (
    QueryAgentRequest,
    QueryPandasSandboxRequest,
    QueryUnstructuredRAGRequest,
)
from src.database.connection import get_db_manager
from src.ingestion.metadata_extractor import EmbeddingService, MetadataExtractor
from src.ingestion.structured import StructuredIngestionEngine
from src.ingestion.unstructured import UnstructuredIngestionEngine
from src.storage.blob_store import get_blob_manager
from tests.conftest import requires_llm


class TestAPIRoutes(unittest.TestCase):
    """Test suite for API Routes and Endpoints."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_api_blobs_"))
        self.db_manager = get_db_manager(in_memory=True)
        self.blob_manager = get_blob_manager(base_path=self.temp_dir)
        self.embedding_service = EmbeddingService()
        self.meta_extractor = MetadataExtractor(embedding_service=self.embedding_service)

        self.structured_engine = StructuredIngestionEngine(
            db_manager=self.db_manager,
            blob_manager=self.blob_manager,
            metadata_extractor=self.meta_extractor,
        )
        self.unstructured_engine = UnstructuredIngestionEngine(
            db_manager=self.db_manager,
            blob_manager=self.blob_manager,
            embedding_service=self.embedding_service,
        )

        sample_csv = (
            "order_id,customer_id,order_date,status,total_amount,shipping_city\n"
            "101,501,2024-01-10 10:00:00,completed,150.50,New York\n"
            "102,502,2024-01-11 11:30:00,completed,280.00,San Francisco\n"
        )
        self.struct_ds = self.structured_engine.ingest_file(
            file_input=sample_csv,
            filename="api_orders.csv",
            display_name="API Orders",
        )

        sample_md = (
            "# API Policy Handbook\n\n"
            "## Rate Limits\n"
            "Authenticated clients have a rate limit of 1000 requests per minute.\n"
        )
        self.unstruct_ds = self.unstructured_engine.ingest_file(
            file_input=sample_md,
            filename="api_policy.md",
            display_name="API Policy",
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_datasets_list_and_get_endpoints(self):
        """Verify GET /datasets and GET /datasets/{id}."""
        datasets_resp = asyncio.run(ingest.list_datasets_endpoint())
        self.assertGreaterEqual(len(datasets_resp.datasets), 2)

        ds_detail = asyncio.run(ingest.get_dataset_endpoint(self.struct_ds.id))
        self.assertEqual(ds_detail["id"], self.struct_ds.id)
        self.assertEqual(ds_detail["name"], "API Orders")

    def test_ingest_endpoint(self):
        """Verify POST /ingest uploads and parses files."""
        csv_payload = ("test_upload.csv", b"col_a,col_b\n1,foo\n2,bar\n")
        resp = asyncio.run(
            ingest.ingest_file_endpoint(
                file=csv_payload,
                display_name="Test Upload",
            )
        )
        self.assertIsNotNone(resp.dataset_id)
        self.assertEqual(resp.name, "Test Upload")
        self.assertEqual(resp.category, "structured")

    @requires_llm
    def test_query_pandas_sandbox_endpoint(self):
        """Verify POST /query/pandas-sandbox."""
        req = QueryPandasSandboxRequest(query="How many total orders are there?")
        resp = asyncio.run(query.query_pandas_sandbox_endpoint(req))
        self.assertIsNone(resp.error)
        self.assertTrue(resp.security_report.ast_passed)
        self.assertEqual(resp.tabular_result.row_count, 1)
        self.assertIsNotNone(resp.thinking_process)
        self.assertGreaterEqual(len(resp.thinking_process.steps), 3)

    @requires_llm
    def test_query_unstructured_rag_endpoint(self):
        """Verify POST /query/unstructured-rag."""
        req = QueryUnstructuredRAGRequest(query="What is the rate limit for authenticated clients?")
        resp = asyncio.run(query.query_unstructured_rag_endpoint(req))
        self.assertIsNone(resp.error)
        self.assertGreater(resp.retrieved_chunks_count, 0)
        self.assertIn("1000 requests", resp.answer)
        self.assertIsNotNone(resp.thinking_process)
        self.assertGreaterEqual(len(resp.thinking_process.steps), 3)

    @requires_llm
    def test_query_agent_endpoint(self):
        """Verify POST /query/agent conversational endpoint."""
        req = QueryAgentRequest(
            query="How many total orders are there?", session_id="test_sess_api"
        )
        resp = asyncio.run(agent.query_agent_endpoint(req))
        self.assertIsNone(resp.error)
        self.assertEqual(resp.intent, "STRUCTURED_QUERY")
        self.assertEqual(resp.session_id, "test_sess_api")
        self.assertGreater(resp.token_usage.total_tokens, 0)
        self.assertIn("2", resp.answer)
        self.assertIsNotNone(resp.thinking_process)
        self.assertGreaterEqual(len(resp.thinking_process.steps), 3)


if __name__ == "__main__":
    unittest.main()
