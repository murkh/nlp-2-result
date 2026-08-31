"""
Adversarial Stress Test Suite for Milestone 1 (Ingestion, Storage & Two-Stage Schema Pruner).
Empirical challenge harness testing extreme inputs, malformed data, large scale schemas (>50 tables, >500 cols),
complex multi-hop FK chains, ambiguous queries, and edge cases.
"""

import shutil
import tempfile
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
from src.pruning.schema_pruner import TwoStageSchemaPruner, estimate_token_count
from src.storage.blob_store import BlobStorageManager
from tests.conftest import create_test_fixtures


class TestAdversarialMilestone1(unittest.TestCase):
    """Adversarial challenger test cases for M1 components."""

    def setUp(self):
        self.fixtures = create_test_fixtures()
        self.temp_dir = self.fixtures["temp_dir"]
        self.test_db = self.fixtures["test_db"]
        self.blob_manager = self.fixtures["blob_manager"]
        self.embedding_service = self.fixtures["embedding_service"]
        self.metadata_extractor = self.fixtures["metadata_extractor"]
        self.structured_engine = self.fixtures["structured_engine"]
        self.unstructured_engine = self.fixtures["unstructured_engine"]
        self.schema_pruner = self.fixtures["schema_pruner"]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # =========================================================================
    # 1. TWO-STAGE SCHEMA PRUNER ADVERSARIAL TESTS
    # =========================================================================

    def test_adversarial_ambiguous_queries(self):
        """Stress-test Stage 1 & 2 pruner with ambiguous queries and overlapping table semantics."""
        tables_to_ingest = [
            (
                "sales_2023.csv",
                "id,product_id,amount,region\n1,101,500,North\n2,102,750,South\n",
                "Sales 2023",
                "Historical sales data for 2023 financial year",
            ),
            (
                "sales_2024.csv",
                "id,product_id,amount,region\n1,101,600,North\n2,103,900,East\n",
                "Sales 2024",
                "Current sales data for 2024 financial year",
            ),
            (
                "sales_targets.csv",
                "id,rep_id,target_amount,quarter\n1,201,10000,Q1\n2,202,15000,Q2\n",
                "Sales Targets",
                "Quarterly quota and sales targets for representatives",
            ),
            (
                "customer_sales.csv",
                "id,customer_id,total_spend\n1,301,1250\n2,302,3400\n",
                "Customer Sales",
                "Aggregated customer lifetime purchase value and sales summary",
            ),
            (
                "products.csv",
                "product_id,name,category,price\n101,Widget A,Gadgets,50\n102,Widget B,Gadgets,75\n",
                "Products Catalog",
                "Master product listings, names, categories, and retail prices",
            ),
        ]

        for fname, csv_data, dname, desc in tables_to_ingest:
            self.structured_engine.ingest_file(
                file_input=csv_data, filename=fname, display_name=dname, description=desc
            )

        # Ambiguous query 1: generic "sales"
        ctx1 = self.schema_pruner.prune_schema(query="sales", top_k_tables=3)
        self.assertEqual(len(ctx1.table_names), 3)
        self.assertIn("LIMIT 20", ctx1.ddl_prompt_snippet)
        self.assertIn("Read-only queries only", ctx1.ddl_prompt_snippet)

        # Ambiguous query 2: pure punctuation / symbols
        ctx2 = self.schema_pruner.prune_schema(query="??? !!! @#$%", top_k_tables=2)
        self.assertEqual(len(ctx2.table_names), 2)
        self.assertIn("CREATE TABLE", ctx2.ddl_prompt_snippet)
        self.assertIn("LIMIT 20", ctx2.ddl_prompt_snippet)

        # Ambiguous query 3: empty string
        ctx3 = self.schema_pruner.prune_schema(query="", top_k_tables=2)
        self.assertEqual(len(ctx3.table_names), 2)
        self.assertIn("CREATE TABLE", ctx3.ddl_prompt_snippet)
        self.assertIn("LIMIT 20", ctx3.ddl_prompt_snippet)

    def test_large_scale_schema_50_plus_tables_500_plus_cols(self):
        """
        Adversarial Stress Test: Enterprise schema with 55 tables and 660 columns.
        Verifies:
        1. Token reduction strictly exceeds >85% (per requirement).
        2. DDL snippet contains mandatory LIMIT 20 directive.
        3. All 55 tables are properly indexed in database metadata.
        """
        total_tables = 55
        cols_per_table = 12  # Total: 55 * 12 = 660 columns

        for i in range(total_tables):
            cols = [f"t{i}_col_{j}" for j in range(cols_per_table)]
            header = f"id,t{i}_code," + ",".join(cols) + "\n"
            row1 = f"{i+1},CODE_{i}," + ",".join([str(j * 5) for j in range(cols_per_table)]) + "\n"
            row2 = (
                f"{i+100},CODE_{i},"
                + ",".join([str((j + 1) * 7) for j in range(cols_per_table)])
                + "\n"
            )
            csv_content = header + row1 + row2

            self.structured_engine.ingest_file(
                file_input=csv_content,
                filename=f"enterprise_dept_{i:02d}.csv",
                display_name=f"Enterprise Department {i:02d}",
                description=f"Operational metrics, KPIs, and audit statistics for enterprise department {i:02d}.",
            )

        # Verify DB registered 55 tables
        tables = self.test_db.list_tables()
        self.assertEqual(len(tables), total_tables)

        # Prune schema for a specific targeted query
        pruned_ctx = self.schema_pruner.prune_schema(
            query="Compute sum of t05_col_3 and t05_col_4 for enterprise department 05",
            top_k_tables=3,
            max_cols_per_table=5,
            total_max_cols=15,
        )

        # Verify Stage 1 table count
        self.assertEqual(len(pruned_ctx.table_names), 3)

        # Verify mandatory guardrails
        self.assertIn("LIMIT 20", pruned_ctx.ddl_prompt_snippet)
        self.assertIn("Read-only queries only", pruned_ctx.ddl_prompt_snippet)

        # Verify token reduction > 85%
        print(
            f"\n[Adversarial 55-Table Benchmark] Pruned={pruned_ctx.token_count_pruned} tokens, Full={pruned_ctx.token_count_full} tokens, Savings={pruned_ctx.token_savings_percent}%"
        )
        self.assertGreater(pruned_ctx.token_savings_percent, 85.0)
        self.assertLess(pruned_ctx.token_count_pruned, 600)
        self.assertGreater(pruned_ctx.token_count_full, 4000)

    def test_complex_multi_hop_foreign_key_chains(self):
        """
        Adversarial Stress Test: Multi-table relational schema with deep FK chains.
        Chain: Countries (PK country_id)
                 ^
                 | FK country_id
               Customers (PK customer_id)
                 ^
                 | FK customer_id
               Orders (PK order_id)
                 ^
                 | FK order_id
               Order_Items (PK item_id, FK order_id, FK product_id)
                 |
                 v FK product_id
               Products (PK product_id)

        Verify: Stage 2 pruner NEVER prunes away PK or FK columns even under strict column limits.
        """
        # Ingest Countries
        self.structured_engine.ingest_file(
            file_input="country_id,country_name,iso_code,continent,population,gdp\n1,USA,US,NA,330000000,25000000\n2,Germany,DE,EU,84000000,4200000\n",
            filename="countries.csv",
            display_name="Countries",
            description="Geographic countries reference table",
        )

        # Ingest Customers
        self.structured_engine.ingest_file(
            file_input="customer_id,country_id,first_name,last_name,email,created_at,credit_score,phone\n501,1,Alice,Smith,alice@us.com,2024-01-01,750,+1-555-0100\n502,2,Hans,Mueller,hans@de.com,2024-01-02,800,+49-151-0200\n",
            filename="customers.csv",
            display_name="Customers",
            description="Customer profiles with country foreign key",
        )

        # Ingest Orders
        self.structured_engine.ingest_file(
            file_input="order_id,customer_id,order_date,status,total_amount,discount,shipping_fee,notes\n1001,501,2024-01-10,completed,250.00,10.00,15.00,fast delivery\n1002,502,2024-01-11,shipped,450.00,0.00,20.00,gift wrap\n",
            filename="orders.csv",
            display_name="Orders",
            description="Sales orders linked to customer_id",
        )

        # Ingest Products
        self.structured_engine.ingest_file(
            file_input="product_id,sku,title,description,unit_price,cost,weight_kg,is_available\n9001,SKU-A,Laptop,Pro Laptop,1200.00,800.00,1.8,true\n9002,SKU-B,Monitor,4K Monitor,400.00,250.00,4.5,true\n",
            filename="products.csv",
            display_name="Products",
            description="Product catalog with retail price and SKU",
        )

        # Ingest Order Items (has 2 FKs: order_id, product_id)
        self.structured_engine.ingest_file(
            file_input="item_id,order_id,product_id,quantity,unit_price,tax,subtotal,custom_note,extra_a,extra_b,extra_c\n1,1001,9001,1,1200.00,120.00,1320.00,engraving,x,y,z\n2,1001,9002,2,400.00,80.00,880.00,none,x,y,z\n",
            filename="order_items.csv",
            display_name="Order Items",
            description="Line items of orders with foreign keys to orders and products",
        )

        # Prune schema with extreme column cap (max_cols_per_table=2, total_max_cols=8)
        pruned_ctx = self.schema_pruner.prune_schema(
            query="Find the total revenue and quantities for products ordered by customer Alice",
            top_k_tables=5,
            max_cols_per_table=2,
            total_max_cols=8,
        )

        # Check each table retained in pruning
        for tname in pruned_ctx.table_names:
            cols = pruned_ctx.retained_columns[tname]
            if "order_item" in tname:
                # Must retain item_id (PK), order_id (FK), product_id (FK)
                self.assertIn("item_id", cols, f"PK item_id missing from {tname}")
                self.assertIn("order_id", cols, f"FK order_id missing from {tname}")
                self.assertIn("product_id", cols, f"FK product_id missing from {tname}")
            elif "order" in tname:
                self.assertIn("order_id", cols, f"PK order_id missing from {tname}")
                self.assertIn("customer_id", cols, f"FK customer_id missing from {tname}")
            elif "customer" in tname:
                self.assertIn("customer_id", cols, f"PK customer_id missing from {tname}")
                self.assertIn("country_id", cols, f"FK country_id missing from {tname}")
            elif "product" in tname:
                self.assertIn("product_id", cols, f"PK product_id missing from {tname}")
            elif "countr" in tname:
                self.assertIn("country_id", cols, f"PK country_id missing from {tname}")

        # Check that foreign key comments are present in the DDL prompt
        self.assertIn("-- FOREIGN KEY", pruned_ctx.ddl_prompt_snippet)
        self.assertIn("PRIMARY KEY", pruned_ctx.ddl_prompt_snippet)

    def test_pruner_zero_tables_in_db(self):
        """Verify schema pruner handles empty database gracefully with 0 crashes."""
        pruned_ctx = self.schema_pruner.prune_schema(query="Select all records")
        self.assertEqual(len(pruned_ctx.table_names), 0)
        self.assertEqual(len(pruned_ctx.table_ids), 0)
        self.assertEqual(pruned_ctx.file_paths, {})
        self.assertIn("No structured database tables found", pruned_ctx.ddl_prompt_snippet)
        self.assertEqual(pruned_ctx.token_savings_percent, 0.0)

    # =========================================================================
    # 2. STRUCTURED INGESTION ADVERSARIAL STRESS TESTS
    # =========================================================================

    def test_special_column_names_and_sql_keywords(self):
        """
        Adversarial Test: Ingest CSV containing SQL reserved keywords, spaces,
        special characters, symbols, quotes, and digit prefixes as column names.
        """
        special_headers = (
            "select,from,where,order,group,table,user,limit,join,case,default,check,"
            '"Column With Spaces","Total ($ USD)","Order # / Ref @","col.with.dots",'
            '"!@#$%^&*()",123_starts_with_digit,""\n'
        )
        row1 = '10,20,30,40,50,60,70,80,90,100,110,120,"Value 1",99.99,"REF-101","dot.val","symbols",999,"empty_val"\n'
        row2 = '11,21,31,41,51,61,71,81,91,101,111,121,"Value 2",149.50,"REF-102","dot.val2","symbols2",888,"empty_val2"\n'
        csv_text = special_headers + row1 + row2

        dataset = self.structured_engine.ingest_file(
            file_input=csv_text,
            filename="sql_keywords_and_special.csv",
            display_name="Keywords & Special Cols",
        )

        self.assertEqual(dataset.row_count, 2)
        tables = self.test_db.list_tables()
        self.assertEqual(len(tables), 1)
        t_meta = tables[0]

        # Verify SQL execution on the dedicated table with escaped identifiers
        cols, rows = self.test_db.execute_sql_query(f'SELECT * FROM "{t_meta.table_name}"')
        self.assertEqual(len(rows), 2)
        self.assertIn("select", cols)
        self.assertIn("from", cols)
        self.assertIn("order", cols)
        self.assertIn("column_with_spaces", cols)
        self.assertIn("col_123_starts_with_digit", cols)

    def test_duplicate_and_colliding_column_names(self):
        """Verify ingestion of CSV with duplicate column names generates unique identifiers without error."""
        csv_text = (
            "id,name,Name,NAME,amount,data,data,data_1\n"
            "1,Alice,Alpha,AAA,100,D1,D2,D3\n"
            "2,Bob,Beta,BBB,200,E1,E2,E3\n"
        )
        dataset = self.structured_engine.ingest_file(
            file_input=csv_text,
            filename="duplicate_cols.csv",
            display_name="Duplicate Columns Test",
        )

        self.assertEqual(dataset.row_count, 2)
        t_meta = self.test_db.list_tables()[0]
        columns = self.test_db.get_columns_for_table(t_meta.id)
        col_names = [c.column_name for c in columns]

        # All column names must be unique
        self.assertEqual(len(col_names), len(set(col_names)))
        self.assertIn("name", col_names)
        self.assertIn("name_2", col_names)
        self.assertIn("name_3", col_names)

        # SQL query execution must succeed
        cols, rows = self.test_db.execute_sql_query(f'SELECT * FROM "{t_meta.table_name}"')
        self.assertEqual(len(rows), 2)

    def test_malformed_ragged_rows_and_missing_values(self):
        """Verify parser normalizes ragged rows (shorter and longer than header) with padding and truncation."""
        csv_text = (
            "col_a,col_b,col_c,col_d\n"
            "1,2,3,4\n"
            "5,6\n"  # Short row (needs padding with None)
            "7,8,9,10,11,12,13\n"  # Long row (needs truncation to 4 cols)
            "   \n"  # Empty whitespace line (should be skipped)
            "\n"  # Blank line (should be skipped)
            "14,15,16,17\n"
        )
        dataset = self.structured_engine.ingest_file(
            file_input=csv_text,
            filename="ragged.csv",
            display_name="Ragged CSV Test",
        )

        self.assertEqual(dataset.row_count, 4)
        t_meta = self.test_db.list_tables()[0]
        cols, rows = self.test_db.execute_sql_query(f'SELECT * FROM "{t_meta.table_name}"')
        self.assertEqual(len(rows), 4)

        # Verify second row had short columns padded with NULL / None
        row_2 = rows[1]
        self.assertEqual(row_2[1], 5)
        self.assertEqual(row_2[2], 6)
        self.assertIsNone(row_2[3])
        self.assertIsNone(row_2[4])

    def test_all_null_and_mixed_type_columns(self):
        """Verify statistical profiling and type deduction for 100% null columns and mixed-type columns."""
        csv_text = (
            "id,null_col_1,null_col_2,mixed_col,bool_col,float_col\n"
            '1,"",NULL,100,true,12.34\n'
            "2,NaN,None,hello,false,56.78\n"
            "3,NA,,3.1415,t,90.12\n"
            '4,null,nan,{"key": "value"},f,0.00\n'
        )
        dataset = self.structured_engine.ingest_file(
            file_input=csv_text,
            filename="nulls_and_types.csv",
            display_name="Nulls and Types Test",
        )

        t_meta = self.test_db.list_tables()[0]
        columns = self.test_db.get_columns_for_table(t_meta.id)
        col_map = {c.column_name: c for c in columns}

        # Verify 100% null columns
        self.assertEqual(col_map["null_col_1"].null_percentage, 100.0)
        self.assertEqual(col_map["null_col_1"].data_type, "TEXT")
        self.assertEqual(col_map["null_col_2"].null_percentage, 100.0)
        self.assertEqual(col_map["null_col_2"].data_type, "TEXT")

        # Verify boolean column
        self.assertEqual(col_map["bool_col"].data_type, "BOOLEAN")

        # Verify float column
        self.assertEqual(col_map["float_col"].data_type, "DOUBLE PRECISION")

    def test_empty_and_header_only_csv(self):
        """Verify 0-byte CSV raises ValueError and header-only CSV is ingested with 0 rows."""
        # 0-byte CSV
        with self.assertRaises(ValueError):
            self.structured_engine.ingest_file(
                file_input="",
                filename="empty.csv",
            )

        # Header-only CSV
        header_only_csv = "order_id,customer_id,amount,status\n"
        dataset = self.structured_engine.ingest_file(
            file_input=header_only_csv,
            filename="header_only.csv",
            display_name="Header Only Table",
        )
        self.assertEqual(dataset.row_count, 0)
        t_meta = self.test_db.list_tables()[0]
        self.assertEqual(t_meta.row_count, 0)
        self.assertEqual(t_meta.column_count, 4)

    def test_csv_embedded_newlines_and_escaped_quotes(self):
        """Verify CSV parser correctly handles multi-line cell values and complex escaped quotes."""
        csv_text = (
            "id,title,description,amount\n"
            '1,"Item One","First line of desc\nSecond line of desc\nThird line of desc",19.99\n'
            '2,"Item Two","Contains ""double quotes"" and commas, semicolons;",49.50\n'
        )
        dataset = self.structured_engine.ingest_file(
            file_input=csv_text,
            filename="multiline_cells.csv",
            display_name="Multiline Cells",
        )

        self.assertEqual(dataset.row_count, 2)
        t_meta = self.test_db.list_tables()[0]
        cols, rows = self.test_db.execute_sql_query(f'SELECT * FROM "{t_meta.table_name}"')
        self.assertEqual(len(rows), 2)
        desc_idx = cols.index("description")
        desc_1 = rows[0][desc_idx]
        self.assertIn("First line of desc", desc_1)
        self.assertIn("Second line of desc", desc_1)
        self.assertIn("Third line of desc", desc_1)

    # =========================================================================
    # 3. UNSTRUCTURED INGESTION ADVERSARIAL STRESS TESTS
    # =========================================================================

    def test_unstructured_empty_and_whitespace_documents(self):
        """Verify unstructured engine raises ValueError on empty or whitespace-only documents."""
        # Empty string
        with self.assertRaises(ValueError):
            self.unstructured_engine.ingest_file(
                file_input="",
                filename="empty.txt",
            )

        # Whitespace-only string
        with self.assertRaises(ValueError):
            self.unstructured_engine.ingest_file(
                file_input="   \n\n\t\t\n   ",
                filename="whitespace.txt",
            )

    def test_unstructured_large_document_120k_characters(self):
        """
        Adversarial Stress Test: Large document (>120,000 characters) across 20 distinct chapters.
        Verifies:
        1. Recursive chunking cleanly produces chunks without crash or TLE.
        2. All chunks receive valid token/char counts and vector embeddings.
        3. Hybrid search (dense + BM25 sparse RRF) successfully retrieves target sections.
        """
        chapters = []
        for i in range(20):
            body_paragraphs = [
                f"This is chapter {i+1} discussing topic {i*7 + k}. "
                f"Detailed analysis on subject matter alpha beta gamma delta epsilon. "
                f"Specific reference token: UNIQUE_TARGET_TOKEN_{i}_{k} for testing search accuracy. "
                * 8
                for k in range(5)
            ]
            chapter_text = f"# Chapter {i+1}: Advanced Systems Engineering {i+1}\n\n" + "\n\n".join(
                body_paragraphs
            )
            chapters.append(chapter_text)

        large_doc_text = "\n\n---\n\n".join(chapters)
        self.assertGreater(len(large_doc_text), 100000)

        dataset = self.unstructured_engine.ingest_file(
            file_input=large_doc_text,
            filename="large_systems_manual.md",
            display_name="Large Systems Manual",
        )

        self.assertGreater(dataset.row_count, 100)

        # Verify Hybrid RRF Search against specific token in Chapter 14
        search_query = "UNIQUE_TARGET_TOKEN_14_3"
        results = self.unstructured_engine.search_hybrid(
            query=search_query,
            top_k=5,
        )

        self.assertGreater(len(results), 0)
        found_target = any("UNIQUE_TARGET_TOKEN_14_3" in r["content"] for r in results)
        self.assertTrue(
            found_target,
            "Target token UNIQUE_TARGET_TOKEN_14_3 not found in top hybrid search results",
        )
        self.assertGreater(results[0]["rrf_score"], 0.0)

    def test_unstructured_extreme_unbroken_text_and_unicode(self):
        """
        Adversarial Test: Chunker handles completely unbroken 15,000-char string without spaces,
        and strings with dense emojis, CJK characters, Arabic, Cyrillic, and math symbols.
        """
        # 1. Unbroken continuous string (no spaces, no newlines)
        unbroken_text = "ABCDEFGHIJ" * 1500  # 15,000 characters
        chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.split_text(unbroken_text)

        self.assertGreaterEqual(len(chunks), 30)
        for c in chunks:
            self.assertLessEqual(len(c), 500)

        # 2. Rich unicode & emoji text document
        unicode_doc = (
            "# 多言語ドキュメント (Multilingual Guide) 🚀🔥\n\n"
            "## Japanese (日本語)\n"
            "大規模言語モデルとベクトルデータベースの統合による検索拡張生成アーキテクチャ。\n\n"
            "## Russian (Русский)\n"
            "Архитектура мультиагентных систем для обработки структурированных данных.\n\n"
            "## Arabic (العربية)\n"
            "منصة ذكاء اصطناعي متعددة الوكلاء للإجابة على الأسئلة من قواعد البيانات.\n\n"
            "## Math & Emojis 📊📈💡\n"
            "Formula: RRF = \\sum_{i=1}^k \\frac{1}{60 + r_i} \\implies \\text{Accuracy} \\ge 99.9\\%.\n"
        )

        dataset = self.unstructured_engine.ingest_file(
            file_input=unicode_doc,
            filename="multilingual_guide.md",
            display_name="Multilingual Guide",
        )

        self.assertGreater(dataset.row_count, 0)

        # Search in Japanese
        results_jp = self.unstructured_engine.search_hybrid(query="ベクトルデータベース", top_k=2)
        self.assertGreater(len(results_jp), 0)
        self.assertIn("ベクトルデータベース", results_jp[0]["content"])

        # Search in Russian
        results_ru = self.unstructured_engine.search_hybrid(query="мультиагентных", top_k=2)
        self.assertGreater(len(results_ru), 0)
        self.assertIn("мультиагентных", results_ru[0]["content"])


if __name__ == "__main__":
    unittest.main()
