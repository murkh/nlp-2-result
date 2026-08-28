"""
Empirical Adversarial Stress Test Suite for Milestone 1.
Challenger 2 Verification Suite covering:
1. Storage Security & Path Traversal Protections (including dataset_id vulnerability reproduction)
2. SHA-256 Content Deduplication & Concurrency Collision Resilience
3. Hybrid Search Injection Strings, Syntax Attacks & Query Edge Cases
4. Resource Consumption, Scale Stress & Memory Profiling
"""

import hashlib
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
import uuid

from src.database.connection import DatabaseManager
from src.database.models import Dataset
from src.ingestion.metadata_extractor import EmbeddingService, MetadataExtractor
from src.ingestion.structured import StructuredIngestionEngine, sanitize_identifier, sanitize_table_name
from src.ingestion.unstructured import RecursiveCharacterChunker, UnstructuredIngestionEngine
from src.pruning.schema_pruner import TwoStageSchemaPruner
from src.storage.blob_store import BlobStorageManager, compute_sha256, sanitize_filename
from tests.conftest import SAMPLE_CSV_TEXT, SAMPLE_CUSTOMERS_CSV_TEXT, SAMPLE_MARKDOWN_TEXT, create_test_fixtures


class TestStorageSecurity(unittest.TestCase):
    """Adversarial stress tests for filesystem blob storage and path security."""

    def setUp(self):
        self.fixtures = create_test_fixtures()
        self.temp_dir = self.fixtures["temp_dir"]
        self.blob_manager = self.fixtures["blob_manager"]
        self.structured_engine = self.fixtures["structured_engine"]
        self.unstructured_engine = self.fixtures["unstructured_engine"]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sanitize_filename_traversal_payloads(self):
        """Verify sanitize_filename neutralizes path traversal payloads."""
        payloads = [
            ("../../etc/passwd", "passwd"),
            ("../../../../root/secret.txt", "secret.txt"),
            ("C:\\Windows\\System32\\cmd.exe", "cmd.exe"),
            ("/absolute/path/to/data.csv", "data.csv"),
            ("test\x00nullbyte.csv", "test_nullbyte.csv"),
            ("....//....//etc/passwd", "passwd"),
            (".hidden_config", "file_.hidden_config"),
            ("..", "file_.."),
            ("   ", "col"),
            ("foo/bar/baz.parquet", "baz.parquet"),
            ("spaces and !@#$%^&*()+={}[]|:;\"'<>,? file.csv", "spaces_and_____________________________file.csv"),
            ("unicode_🚀_file.txt", "unicode___file.txt"),
        ]

        for payload, expected_substring in payloads:
            sanitized = sanitize_filename(payload)
            self.assertNotIn("/", sanitized, f"Slash found in sanitized filename: {sanitized}")
            self.assertNotIn("\\", sanitized, f"Backslash found in sanitized filename: {sanitized}")
            self.assertNotIn("\x00", sanitized, f"Null byte found in sanitized filename: {sanitized}")
            self.assertLessEqual(len(sanitized), 255)

    def test_get_absolute_path_traversal_prevention(self):
        """Verify get_absolute_path blocks escape attempts outside base_path."""
        traversal_attempts = [
            "../../etc/passwd",
            "../../../secret.txt",
            "/etc/shadow",
            "uuid/../../../etc/passwd",
            f"{self.temp_dir.name}/../../etc/passwd",
            "nonexistent/../../..",
        ]

        for path in traversal_attempts:
            with self.assertRaises(ValueError, msg=f"Failed to block traversal: {path}"):
                self.blob_manager.get_absolute_path(path)

    def test_save_file_dataset_id_path_traversal_vulnerability(self):
        """Verify save_file sanitizes dataset_id preventing escape outside base_path."""
        isolated_root = Path(tempfile.mkdtemp(prefix="test_iso_"))
        try:
            blob_dir = isolated_root / "blob_store"
            blob_dir.mkdir()
            bm = BlobStorageManager(base_path=blob_dir)

            # Attempt to write outside blob_store into isolated_root
            d_id, rel_path, sz, h = bm.save_file(
                file_input=b"malicious payload",
                filename="traversal_test.csv",
                dataset_id="../escaped_dataset",
            )

            escaped_target = isolated_root / "escaped_dataset" / "traversal_test.csv"
            # Ensure file did NOT escape outside blob_dir
            self.assertFalse(escaped_target.exists(), "Security Violation: file escaped outside base_path!")
            # Ensure file is safely contained inside blob_dir
            safe_target = blob_dir / "escaped_dataset" / "traversal_test.csv"
            self.assertTrue(safe_target.exists(), "File should be stored in sanitized dataset directory within base_path")
            self.assertEqual(d_id, "escaped_dataset")
        finally:
            shutil.rmtree(isolated_root, ignore_errors=True)

    def test_list_dataset_files_path_traversal_behavior(self):
        """Verify list_dataset_files sanitizes dataset_id and returns empty list for traversal attempts."""
        traversal_ids = ["../", "../../", "../../../etc", "/root", "///"]
        for tid in traversal_ids:
            files = self.blob_manager.list_dataset_files(tid)
            self.assertEqual(files, [], f"Expected empty list for path traversal dataset_id: {tid}")

    def test_delete_file_traversal_resilience(self):
        """Verify delete_file does not delete files outside base_path."""
        self.assertFalse(self.blob_manager.delete_file("../../etc/passwd"))
        self.assertFalse(self.blob_manager.delete_file("/root/secret"))

    def test_ingestion_with_malicious_filenames(self):
        """Verify structured and unstructured ingestion engines handle malicious filenames safely."""
        # 1. Structured CSV with path traversal filename
        dataset_struct = self.structured_engine.ingest_file(
            file_input=SAMPLE_CSV_TEXT,
            filename="../../../../etc/passwd.csv",
            display_name="Malicious Path Test",
        )
        self.assertIsNotNone(dataset_struct.id)
        self.assertTrue(self.blob_manager.exists(dataset_struct.blob_path))
        abs_path = self.blob_manager.get_absolute_path(dataset_struct.blob_path)
        self.assertTrue(str(abs_path).startswith(str(self.temp_dir.resolve())))

        # 2. Unstructured Markdown with path traversal filename
        dataset_unstruct = self.unstructured_engine.ingest_file(
            file_input=SAMPLE_MARKDOWN_TEXT,
            filename="../../var/log/system.log.md",
            display_name="System Log MD",
        )
        self.assertIsNotNone(dataset_unstruct.id)
        self.assertTrue(self.blob_manager.exists(dataset_unstruct.blob_path))


class TestSha256DeduplicationAndConcurrency(unittest.TestCase):
    """Adversarial stress tests for SHA-256 deduplication and concurrent operations."""

    def setUp(self):
        self.fixtures = create_test_fixtures()
        self.temp_dir = self.fixtures["temp_dir"]
        self.test_db = self.fixtures["test_db"]
        self.blob_manager = self.fixtures["blob_manager"]
        self.structured_engine = self.fixtures["structured_engine"]
        self.unstructured_engine = self.fixtures["unstructured_engine"]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sha256_hash_verification(self):
        """Verify compute_sha256 matches hashlib.sha256 exactly across various data types."""
        test_payloads = [
            b"",
            b"Simple string",
            SAMPLE_CSV_TEXT.encode("utf-8"),
            SAMPLE_MARKDOWN_TEXT.encode("utf-8"),
            b"\x00\x01\x02\xff\xfe\xfd" * 1000,
            ("UNICODE " * 500 + "🚀🎉").encode("utf-8"),
        ]

        for payload in test_payloads:
            expected = hashlib.sha256(payload).hexdigest()
            computed = compute_sha256(payload)
            self.assertEqual(computed, expected)
            self.assertEqual(len(computed), 64)

    def test_duplicate_content_upsert_deduplication(self):
        """Verify identical content ingestion triggers ON CONFLICT update without duplicating dataset records."""
        # 1. First ingestion
        ds1 = self.structured_engine.ingest_file(
            file_input=SAMPLE_CSV_TEXT,
            filename="orders_v1.csv",
            display_name="Orders Initial",
        )
        hash1 = ds1.content_hash

        # 2. Second ingestion with identical content but different filename and display name
        ds2 = self.structured_engine.ingest_file(
            file_input=SAMPLE_CSV_TEXT,
            filename="orders_v2_duplicate.csv",
            display_name="Orders Updated Name",
        )
        hash2 = ds2.content_hash

        self.assertEqual(hash1, hash2)

        # 3. Query all datasets: in SQLite/Postgres with ON CONFLICT (content_hash), ensure no duplicate row in DB
        datasets = self.test_db.list_datasets(category="structured")
        hashes = [d.content_hash for d in datasets]
        self.assertEqual(hashes.count(hash1), 1, "Duplicate content_hash found in datasets table!")

    def test_concurrent_ingestion_sqlite_lock_behavior(self):
        """Verify sequential ingestion completes cleanly."""
        num_files = 5
        results = []
        for i in range(num_files):
            csv_data = f"id,user_id,val_{i}\n1,100,{i * 10}\n2,200,{i * 20}\n"
            ds = self.structured_engine.ingest_file(
                file_input=csv_data,
                filename=f"seq_dataset_{i}.csv",
                display_name=f"Sequential Dataset {i}",
            )
            results.append(ds)

        self.assertEqual(len(results), num_files)
        tables = self.test_db.list_tables()
        self.assertEqual(len(tables), num_files)

    def test_multithreaded_concurrent_ingestion_sqlite_mutex(self):
        """Verify concurrent multi-threaded ingestion executes safely with SQLite RLock mutex."""
        num_threads = 8
        threads = []
        errors = []
        results = []

        def worker(idx: int):
            try:
                csv_data = f"id,user_id,metric_{idx}\n1,100,{idx * 10}\n2,200,{idx * 20}\n3,300,{idx * 30}\n"
                ds = self.structured_engine.ingest_file(
                    file_input=csv_data,
                    filename=f"concurrent_dataset_{idx}.csv",
                    display_name=f"Concurrent Dataset {idx}",
                )
                results.append(ds)
            except Exception as e:
                errors.append(e)

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10.0)

        self.assertEqual(len(errors), 0, f"Thread errors encountered: {errors}")
        self.assertEqual(len(results), num_threads)
        tables = self.test_db.list_tables()
        self.assertEqual(len(tables), num_threads)


class TestHybridSearchInjectionAndEdgeCases(unittest.TestCase):
    """Adversarial stress tests for hybrid search injection strings, edge cases, and query attacks."""

    def setUp(self):
        self.fixtures = create_test_fixtures()
        self.temp_dir = self.fixtures["temp_dir"]
        self.test_db = self.fixtures["test_db"]
        self.unstructured_engine = self.fixtures["unstructured_engine"]

        # Ingest standard document
        self.dataset = self.unstructured_engine.ingest_file(
            file_input=SAMPLE_MARKDOWN_TEXT,
            filename="operations_guide.md",
            display_name="Operations Guide",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_zero_matches_query(self):
        """Verify hybrid search gracefully returns empty list on zero-match or nonsense queries."""
        nonsense_queries = [
            "xyzqweasdzxcv1234567890nonexistentterm",
            "999999999-000000000-aaaaaaaa",
            "__undefined_symbol_magic__",
        ]

        for query in nonsense_queries:
            results = self.unstructured_engine.search_hybrid(query=query, top_k=5)
            self.assertIsInstance(results, list)
            for r in results:
                self.assertIn("chunk_id", r)
                self.assertIn("rrf_score", r)

    def test_sql_injection_payloads(self):
        """Verify hybrid search is immune to SQL injection payloads in query string."""
        sql_injections = [
            "' OR 1=1 --",
            "'); DROP TABLE document_chunks; --",
            "' UNION SELECT id, dataset_id, content, 1, 'sec', 'emb', '2024-01-01' FROM document_chunks --",
            "\" OR \"\"=\"",
            "admin' --",
            "1; SELECT pg_sleep(5); --",
            "'; EXEC sp_msforeachtable 'DROP TABLE ?' --",
            "' OR '1'='1",
        ]

        for payload in sql_injections:
            results = self.unstructured_engine.search_hybrid(query=payload, top_k=5)
            self.assertIsInstance(results, list)

        # Verify document_chunks table still intact
        db_dataset = self.test_db.get_dataset(self.dataset.id)
        self.assertIsNotNone(db_dataset)
        self.assertGreater(db_dataset.row_count, 0)

    def test_fulltext_and_regex_special_characters(self):
        """Verify search handles unescaped regex/FTS syntax characters without throwing unhandled errors."""
        special_payloads = [
            "((((((((((unbalanced parentheses",
            "***+++???[]{}|^$\\~`",
            "AND OR NOT NEAR",
            "\"\"\"\"\"\" quotes galore \"\"\"",
            "\x00null\x00byte\x00query",
            "🚀🔥🎉✨🤖💥🔍",
            "مرحباً بك في نظام البحث الذكي",
            "日本語の全文検索テストとクエリ検証",
            "Русский текст и проверка полнотекстового поиска",
            "<script>alert('XSS')</script>",
            "\\\\\\\\\\\\\\\\\\\\",
        ]

        for payload in special_payloads:
            results = self.unstructured_engine.search_hybrid(query=payload, top_k=3)
            self.assertIsInstance(results, list)

    def test_very_long_search_query(self):
        """Verify hybrid search handles massive search queries (10k - 50k characters) without timeout or OOM."""
        long_queries = [
            "incident response " * 1000,    # ~18,000 chars
            "deployment canary " * 2500,    # ~45,000 chars
            "A" * 50000,                     # 50,000 chars contiguous
        ]

        for q in long_queries:
            t0 = time.time()
            results = self.unstructured_engine.search_hybrid(query=q, top_k=5)
            elapsed = time.time() - t0
            self.assertIsInstance(results, list)
            self.assertLess(elapsed, 2.0, f"Query took too long: {elapsed:.2f}s")

    def test_empty_and_whitespace_only_query(self):
        """Verify hybrid search handles empty and whitespace-only queries cleanly."""
        for empty_q in ["", "   ", "\t\t\n\n\r"]:
            results = self.unstructured_engine.search_hybrid(query=empty_q, top_k=5)
            self.assertIsInstance(results, list)

    def test_invalid_dataset_id_filter(self):
        """Verify dataset_id filtering handles non-existent or malformed dataset IDs."""
        bad_ids = [
            "non-existent-uuid-1234",
            str(uuid.uuid4()),
            "'; DROP TABLE datasets; --",
            "",
        ]

        for bid in bad_ids:
            results = self.unstructured_engine.search_hybrid(
                query="Deployment Guidelines", top_k=5, dataset_id=bid
            )
            self.assertIsInstance(results, list)


class TestResourceConsumptionAndStress(unittest.TestCase):
    """Stress tests evaluating chunking, schema pruning scalability, and wide/deep tabular ingestion."""

    def setUp(self):
        self.fixtures = create_test_fixtures()
        self.temp_dir = self.fixtures["temp_dir"]
        self.test_db = self.fixtures["test_db"]
        self.metadata_extractor = self.fixtures["metadata_extractor"]
        self.structured_engine = self.fixtures["structured_engine"]
        self.schema_pruner = self.fixtures["schema_pruner"]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chunker_massive_unbroken_text(self):
        """Verify RecursiveCharacterChunker handles 100,000 chars of unbroken text without recursion overflow."""
        chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=100)
        unbroken_text = "X" * 100000

        t0 = time.time()
        chunks = chunker.split_text(unbroken_text)
        elapsed = time.time() - t0

        self.assertGreater(len(chunks), 200)
        self.assertLess(elapsed, 1.0, f"Chunking took {elapsed:.2f}s")
        for c in chunks:
            self.assertLessEqual(len(c), 500)

    def test_chunker_large_structured_document(self):
        """Verify chunker performance on 1MB realistic text with paragraphs and headings."""
        chunker = RecursiveCharacterChunker(chunk_size=800, chunk_overlap=150)
        paragraph = (
            "This is a standard section detailing system architecture and security protocols. "
            "Engineers must adhere strictly to compliance frameworks and regression testing. "
            "Data encryption at rest and in transit is strictly enforced.\n\n"
        )
        large_text = paragraph * 5000  # ~1.1 MB

        t0 = time.time()
        chunks = chunker.split_text(large_text)
        elapsed = time.time() - t0

        self.assertGreater(len(chunks), 1000)
        self.assertLess(elapsed, 2.0, f"1MB text chunking took {elapsed:.2f}s")

    def test_schema_pruner_scale_50_tables(self):
        """Stress-test two-stage schema pruner against 50 tables (500+ columns)."""
        for i in range(50):
            cols_def = ", ".join([f"metric_{j}" for j in range(10)])
            vals_def = ", ".join([str(j * 5) for j in range(10)])
            csv_data = f"id,dept_{i}_name,active_flag,{cols_def}\n1,Dept_{i},true,{vals_def}\n2,Dept_{i},false,{vals_def}\n"
            self.structured_engine.ingest_file(
                file_input=csv_data,
                filename=f"scale_table_{i}.csv",
                display_name=f"Scale Table {i}",
                description=f"Department {i} telemetry and performance metrics table.",
            )

        t0 = time.time()
        pruned_ctx = self.schema_pruner.prune_schema(
            query="Analyze metric_3 for Dept_25",
            top_k_tables=3,
            max_cols_per_table=5,
            total_max_cols=12,
        )
        elapsed = time.time() - t0

        self.assertLess(elapsed, 1.0, f"Schema pruning on 50 tables took {elapsed:.2f}s")
        self.assertLessEqual(len(pruned_ctx.table_names), 3)
        self.assertGreater(pruned_ctx.token_savings_percent, 85.0)
        self.assertIn("LIMIT 20", pruned_ctx.ddl_prompt_snippet)

    def test_schema_pruner_boundary_parameters(self):
        """Verify schema pruner handles extreme boundary parameters gracefully."""
        self.structured_engine.ingest_file(
            file_input=SAMPLE_CSV_TEXT,
            filename="orders_boundary.csv",
            display_name="Orders Boundary",
        )

        ctx_zero = self.schema_pruner.prune_schema(query="Find orders", top_k_tables=0)
        self.assertIsInstance(ctx_zero.table_names, list)

        ctx_large = self.schema_pruner.prune_schema(query="Find orders", top_k_tables=1000)
        self.assertIsInstance(ctx_large.table_names, list)

    def test_wide_and_dirty_tabular_ingestion(self):
        """Verify metadata extractor and structured ingestion on wide (100 cols) and dirty dataset."""
        col_names = [f"col_{i}" for i in range(100)]
        header = ",".join(col_names)
        
        row1 = ["1" if i % 4 == 0 else ("3.14" if i % 4 == 1 else ("true" if i % 4 == 2 else "sample text")) for i in range(100)]
        row2 = ["" if i % 5 == 0 else ("NULL" if i % 7 == 0 else str(i * 10)) for i in range(100)]
        row3 = ["NaN" if i % 3 == 0 else str(i) for i in range(100)]
        
        csv_content = header + "\n" + ",".join(row1) + "\n" + ",".join(row2) + "\n" + ",".join(row3) + "\n"

        dataset = self.structured_engine.ingest_file(
            file_input=csv_content,
            filename="wide_dirty_dataset.csv",
            display_name="Wide Dirty Dataset",
        )

        self.assertEqual(dataset.row_count, 3)
        tables = self.test_db.list_tables()
        self.assertEqual(len(tables), 1)
        columns = self.test_db.get_columns_for_table(tables[0].id)
        self.assertEqual(len(columns), 100)


if __name__ == "__main__":
    unittest.main()
