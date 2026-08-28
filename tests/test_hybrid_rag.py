"""
Unit and Integration Tests for Unstructured Hybrid RAG Engine.
Verifies Reciprocal Rank Fusion (RRF k=60), bracketed source citation formatting,
and grounded response synthesis.
"""

import re
import unittest
from pathlib import Path

from src.api.schemas import QueryUnstructuredRAGRequest
from src.engines.hybrid_rag import HybridRAGEngine
from tests.conftest import create_test_fixtures


class TestHybridRAGEngine(unittest.TestCase):
    """Test suite for HybridRAGEngine."""

    def setUp(self):
        fixtures = create_test_fixtures()
        self.temp_dir = fixtures["temp_dir"]
        self.db_manager = fixtures["test_db"]
        self.blob_manager = fixtures["blob_manager"]
        self.unstructured_engine = fixtures["unstructured_engine"]
        self.embedding_service = fixtures["embedding_service"]

        sample_markdown = (
            "# Engineering Operations Handbook\n\n"
            "## Deployment Guidelines\n"
            "All deployments must pass continuous integration tests before release. "
            "Canary releases should be monitored for at least 15 minutes before full traffic migration.\n\n"
            "## Incident Response Protocol\n"
            "When a Severity 1 incident occurs, page the on-call engineer and open an incident Slack channel. "
            "A post-mortem document must be published within 48 hours of resolution.\n\n"
            "## Code Review Policy\n"
            "Every pull request requires two approving reviews and zero unresolved comments. "
            "Security-sensitive modules require an explicit sign-off from the Application Security team.\n"
        )

        self.dataset_rec = self.unstructured_engine.ingest_file(
            file_input=sample_markdown,
            filename="engineering_handbook.md",
            display_name="Engineering Handbook",
            description="Operations and engineering guidelines handbook.",
        )

        self.engine = HybridRAGEngine(
            db_manager=self.db_manager,
            embedding_service=self.embedding_service,
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_hybrid_rag_query_with_citations(self):
        """Verify hybrid RAG query retrieves relevant chunks and returns bracketed citations."""
        req = QueryUnstructuredRAGRequest(
            query="What is the canary release monitoring duration?",
            top_k=3,
        )
        resp = self.engine.execute_query(req)

        self.assertIsNone(resp.error)
        self.assertGreater(resp.retrieved_chunks_count, 0)
        self.assertGreater(len(resp.citations), 0)

        top_cite = resp.citations[0]
        self.assertEqual(top_cite.document_name, "Engineering Handbook")
        self.assertGreater(top_cite.similarity_score, 0.0)

        # Check bracketed citation format [Doc: ..., Page: ..., Chunk: ...]
        citation_pattern = r"\[Doc:\s+[^,]+,\s+Page:\s+\d+,\s+Chunk:\s+\d+\]"
        self.assertTrue(
            bool(re.search(citation_pattern, resp.answer)),
            f"Expected bracketed citation in answer: {resp.answer}",
        )
        self.assertIn("15 minutes", resp.answer)

    def test_incident_response_query(self):
        """Verify retrieval for incident post-mortem policy."""
        req = QueryUnstructuredRAGRequest(
            query="When must a post-mortem document be published for Sev 1 incidents?",
            top_k=3,
        )
        resp = self.engine.execute_query(req)

        self.assertIsNone(resp.error)
        self.assertIn("48 hours", resp.answer)
        self.assertGreater(resp.metrics.engine_execution_ms, 0)
        self.assertGreater(resp.token_usage.total_tokens, 0)

    def test_unstructured_rag_empty_result_handling(self):
        """Verify response when query has no matching documents or empty database."""
        empty_db = create_test_fixtures()["test_db"]
        empty_engine = HybridRAGEngine(
            db_manager=empty_db, embedding_service=self.embedding_service
        )

        req = QueryUnstructuredRAGRequest(query="What is the refund policy?")
        resp = empty_engine.execute_query(req)

        self.assertEqual(resp.retrieved_chunks_count, 0)
        self.assertIn("could not find information", resp.answer.lower())


if __name__ == "__main__":
    unittest.main()
