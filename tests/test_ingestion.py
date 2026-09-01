"""
Unit and Integration Tests for Structured and Unstructured Ingestion Pipelines.
Verifies Blob storage persistence, dedicated table creation, metadata profiling,
chunking, embeddings, and hybrid RRF search.
"""

import shutil
import unittest
from pathlib import Path

from src.database.connection import DatabaseManager
from src.ingestion.metadata_extractor import EmbeddingService, MetadataExtractor
from src.ingestion.structured import (
    StructuredIngestionEngine,
    sanitize_identifier,
    sanitize_table_name,
)
from src.ingestion.unstructured import RecursiveCharacterChunker, UnstructuredIngestionEngine
from src.storage.blob_store import BlobStorageManager
from tests.conftest import (
    SAMPLE_CSV_TEXT,
    SAMPLE_CUSTOMERS_CSV_TEXT,
    SAMPLE_MARKDOWN_TEXT,
    create_test_fixtures,
    read_blob_dataframe,
)


class TestIngestion(unittest.TestCase):
    """Test suite for ingestion pipelines and blob storage."""

    def setUp(self):
        self.fixtures = create_test_fixtures()
        self.temp_dir = self.fixtures["temp_dir"]
        self.test_db = self.fixtures["test_db"]
        self.blob_manager = self.fixtures["blob_manager"]
        self.embedding_service = self.fixtures["embedding_service"]
        self.metadata_extractor = self.fixtures["metadata_extractor"]
        self.structured_engine = self.fixtures["structured_engine"]
        self.unstructured_engine = self.fixtures["unstructured_engine"]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sanitize_identifier(self):
        """Verify identifier sanitization produces valid PostgreSQL identifiers."""
        self.assertEqual(sanitize_identifier("Order ID #"), "order_id")
        self.assertEqual(sanitize_identifier("123_abc"), "col_123_abc")
        self.assertEqual(sanitize_identifier("Customer-Name / Title"), "customer_name_title")
        self.assertEqual(sanitize_identifier("   "), "col")

    def test_sanitize_table_name(self):
        """Verify table name format tbl_{prefix}_{name} within 63 characters."""
        tname = sanitize_table_name(
            "123e4567-e89b-12d3-a456-426614174000", "e_commerce_transactions_2024"
        )
        self.assertTrue(tname.startswith("tbl_123e4567_"))
        self.assertLessEqual(len(tname), 63)

    def test_blob_store_save_and_retrieve(self):
        """Verify blob storage file persistence, hash calculation, and path security."""
        content = b"Sample raw dataset content for testing."
        d_id, blob_path, size_bytes, hash_str = self.blob_manager.save_file(
            file_input=content, filename="test_file.csv"
        )

        self.assertTrue(self.blob_manager.exists(blob_path))
        self.assertEqual(size_bytes, len(content))
        self.assertEqual(len(hash_str), 64)
        self.assertEqual(self.blob_manager.read_bytes(blob_path), content)
        self.assertEqual(
            self.blob_manager.read_text(blob_path), "Sample raw dataset content for testing."
        )

        # Test path traversal prevention
        with self.assertRaises(ValueError):
            self.blob_manager.get_absolute_path("../../etc/passwd")

    def test_metadata_extractor_profiling(self):
        """Verify statistical profiling and SQL type deduction."""
        cols = ["order_id", "customer_id", "amount", "is_paid", "notes"]
        rows = [
            [1, 501, 99.95, True, "Express shipping"],
            [2, 502, 149.00, True, "Standard"],
            [3, 501, 25.50, False, None],
        ]

        profile = self.metadata_extractor.profile_table("orders", cols, rows)
        self.assertEqual(profile.table_name, "orders")
        self.assertEqual(profile.row_count, 3)
        self.assertEqual(profile.column_count, 5)

        col_map = {c.column_name: c for c in profile.columns}
        self.assertTrue(col_map["order_id"].is_primary_key)
        self.assertIn(col_map["order_id"].data_type, ("INTEGER", "BIGINT"))
        self.assertTrue(col_map["customer_id"].is_foreign_key)
        self.assertEqual(col_map["amount"].data_type, "DOUBLE PRECISION")
        self.assertEqual(col_map["is_paid"].data_type, "BOOLEAN")
        self.assertGreater(col_map["notes"].null_percentage, 0.0)

    def test_structured_csv_ingestion(self):
        """Verify full CSV ingestion pipeline: blob save and metadata catalog."""
        dataset = self.structured_engine.ingest_file(
            file_input=SAMPLE_CSV_TEXT,
            filename="orders.csv",
            display_name="Customer Orders",
            description="E-commerce customer orders table",
        )

        # 1. Check dataset record
        self.assertEqual(dataset.category, "structured")
        self.assertEqual(dataset.file_type, "csv")
        self.assertEqual(dataset.row_count, 5)
        self.assertEqual(dataset.name, "Customer Orders")

        # 2. Check table metadata in DB
        tables = self.test_db.list_tables()
        self.assertEqual(len(tables), 1)
        table_meta = tables[0]
        self.assertEqual(table_meta.display_name, "Customer Orders")
        self.assertEqual(table_meta.row_count, 5)
        self.assertEqual(table_meta.column_count, 6)

        # 3. Check column metadata in DB
        columns = self.test_db.get_columns_for_table(table_meta.id)
        self.assertEqual(len(columns), 6)
        col_names = [c.column_name for c in columns]
        self.assertIn("order_id", col_names)
        self.assertIn("customer_id", col_names)
        self.assertIn("total_amount", col_names)

        # 4. Check the stored blob -- the only copy of the rows
        df = read_blob_dataframe(self.blob_manager, dataset)
        self.assertEqual(len(df), 5)
        self.assertIn("order_id", df.columns)
        self.assertIn("total_amount", df.columns)

    def test_recursive_character_chunker(self):
        """Verify recursive character text chunker boundaries and overlap."""
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
        text = (
            "First section paragraph here. This explains the initial design.\n\n"
            "Second section paragraph. It details the second phase of operations.\n\n"
            "Third section paragraph. It covers final deployment and validation."
        )
        chunks = chunker.split_text(text)
        self.assertGreaterEqual(len(chunks), 3)
        for c in chunks:
            self.assertLessEqual(len(c), 120)

    def test_unstructured_markdown_ingestion(self):
        """Verify Markdown document chunking, metadata preservation, and embedding."""
        dataset = self.unstructured_engine.ingest_file(
            file_input=SAMPLE_MARKDOWN_TEXT,
            filename="handbook.md",
            display_name="Engineering Handbook",
        )

        self.assertEqual(dataset.category, "unstructured")
        self.assertEqual(dataset.file_type, "md")
        self.assertGreater(dataset.row_count, 0)

        # Verify chunks stored in database
        retrieved_dataset = self.test_db.get_dataset(dataset.id)
        self.assertIsNotNone(retrieved_dataset)
        self.assertEqual(retrieved_dataset.name, "Engineering Handbook")

    def test_unstructured_hybrid_search(self):
        """Verify Reciprocal Rank Fusion (RRF) hybrid dense + sparse retrieval."""
        self.unstructured_engine.ingest_file(
            file_input=SAMPLE_MARKDOWN_TEXT,
            filename="operations.md",
            display_name="Operations Guide",
        )

        # Search for incident response post-mortem
        results = self.unstructured_engine.search_hybrid(
            query="Incident Response post-mortem 48 hours",
            top_k=3,
        )

        self.assertGreater(len(results), 0)
        top_result = results[0]
        self.assertTrue(
            "Incident" in top_result["content"] or "post-mortem" in top_result["content"]
        )
        self.assertGreater(top_result["rrf_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
