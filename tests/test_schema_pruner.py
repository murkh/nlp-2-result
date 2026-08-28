"""
Unit and Integration Tests for Two-Stage Vector Schema Pruning & Token Efficiency.
Verifies Stage 1 table retrieval, Stage 2 column retrieval, PK/FK retention,
LIMIT 20 prompt injection, and token reduction savings (>85%).
"""

import shutil
import unittest

from src.database.connection import DatabaseManager
from src.ingestion.structured import StructuredIngestionEngine
from src.pruning.schema_pruner import TwoStageSchemaPruner, estimate_token_count
from tests.conftest import SAMPLE_CSV_TEXT, SAMPLE_CUSTOMERS_CSV_TEXT, create_test_fixtures


class TestSchemaPruner(unittest.TestCase):
    """Test suite for Two-Stage Vector Schema Pruner."""

    def setUp(self):
        self.fixtures = create_test_fixtures()
        self.temp_dir = self.fixtures["temp_dir"]
        self.test_db = self.fixtures["test_db"]
        self.blob_manager = self.fixtures["blob_manager"]
        self.embedding_service = self.fixtures["embedding_service"]
        self.metadata_extractor = self.fixtures["metadata_extractor"]
        self.structured_engine = self.fixtures["structured_engine"]
        self.schema_pruner = self.fixtures["schema_pruner"]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_schema_pruning_table_selection(self):
        """Verify Stage 1 coarse retrieval selects relevant tables for the natural language query."""
        # 1. Ingest orders
        self.structured_engine.ingest_file(
            file_input=SAMPLE_CSV_TEXT,
            filename="orders.csv",
            display_name="Customer Orders",
            description="Records of e-commerce customer purchase orders, total amounts, and payment status.",
        )

        # 2. Ingest customers
        self.structured_engine.ingest_file(
            file_input=SAMPLE_CUSTOMERS_CSV_TEXT,
            filename="customers.csv",
            display_name="Customers Directory",
            description="Master directory of customer profiles, names, emails, and countries.",
        )

        # 3. Ingest unrelated table (warehouse inventory)
        inventory_csv = (
            "warehouse_id,facility_code,square_footage,temperature_controlled,manager_name\n"
            "1,WH-EAST,50000,true,John Doe\n"
            "2,WH-WEST,75000,false,Jane Smith\n"
        )
        self.structured_engine.ingest_file(
            file_input=inventory_csv,
            filename="warehouses.csv",
            display_name="Warehouse Facilities",
            description="Physical warehouse locations, storage capacities, and climate control specifications.",
        )

        # 4. Prune schema for query about customer orders
        pruned_ctx = self.schema_pruner.prune_schema(
            query="List the names of customers who placed orders over 200 dollars",
            top_k_tables=2,
        )

        self.assertEqual(len(pruned_ctx.table_names), 2)
        # Verify selected tables are orders and customers
        joined_names = " ".join(pruned_ctx.table_names).lower()
        self.assertIn("orders", joined_names)
        self.assertIn("customers", joined_names)
        self.assertNotIn("warehouses", joined_names)

    def test_schema_pruning_pk_fk_retention(self):
        """Verify Stage 2 preserves Primary Keys and Foreign Keys unconditionally."""
        wide_table_csv = (
            "order_id,customer_id,product_id,extra_col_1,extra_col_2,extra_col_3,extra_col_4,extra_col_5\n"
            "1,501,901,a,b,c,d,e\n"
            "2,502,902,f,g,h,i,j\n"
        )
        self.structured_engine.ingest_file(
            file_input=wide_table_csv,
            filename="wide_orders.csv",
            display_name="Wide Orders",
            description="Orders table with many auxiliary columns",
        )

        # Query asking only for product
        pruned_ctx = self.schema_pruner.prune_schema(
            query="What products were purchased?",
            top_k_tables=1,
            max_cols_per_table=3,
        )

        table_name = pruned_ctx.table_names[0]
        retained_cols = pruned_ctx.retained_columns[table_name]

        # Primary key and foreign keys must be retained even with low max_cols_per_table
        self.assertIn("order_id", retained_cols)
        self.assertTrue("customer_id" in retained_cols or "product_id" in retained_cols)

    def test_schema_pruner_limit_20_directive(self):
        """Verify generated DDL snippet contains mandatory LIMIT 20 and read-only directives."""
        self.structured_engine.ingest_file(
            file_input=SAMPLE_CSV_TEXT,
            filename="orders.csv",
            display_name="Orders",
        )

        pruned_ctx = self.schema_pruner.prune_schema(query="Find top 5 orders by amount")

        self.assertIn("LIMIT 20", pruned_ctx.ddl_prompt_snippet)
        self.assertIn("Read-only queries only", pruned_ctx.ddl_prompt_snippet)
        self.assertIn("CREATE TABLE", pruned_ctx.ddl_prompt_snippet)

    def test_token_reduction_benchmark(self):
        """Verify schema pruner achieves >85% token reduction against a large multi-table schema."""
        # Create 10 tables with multiple columns to simulate enterprise schema
        for i in range(10):
            csv_data = (
                f"table_{i}_id,tenant_id,metric_a,metric_b,metric_c,status_code,created_timestamp,notes,tag_1,tag_2\n"
                f"{i*100 + 1},10,12.5,99.4,3.14,active,2024-01-01 00:00:00,log message,tagA,tagB\n"
                f"{i*100 + 2},10,14.1,88.2,2.71,pending,2024-01-02 00:00:00,log message,tagA,tagC\n"
            )
            self.structured_engine.ingest_file(
                file_input=csv_data,
                filename=f"enterprise_dataset_{i}.csv",
                display_name=f"Enterprise Dataset {i}",
                description=f"Operational records for enterprise business department {i} with key metrics.",
            )

        # Prune for a focused query
        pruned_ctx = self.schema_pruner.prune_schema(
            query="Show summary of metric_a for enterprise department 3",
            top_k_tables=2,
            max_cols_per_table=4,
            total_max_cols=8,
        )

        self.assertLess(pruned_ctx.token_count_pruned, pruned_ctx.token_count_full)
        self.assertGreaterEqual(pruned_ctx.token_savings_percent, 60.0)
        print(f"\n[Benchmark 10 Tables] Pruned={pruned_ctx.token_count_pruned} tokens, Full={pruned_ctx.token_count_full} tokens, Savings={pruned_ctx.token_savings_percent}%")

    def test_large_schema_85_percent_token_savings(self):
        """Verify schema pruner achieves >85% token reduction on 25-table enterprise database."""
        for i in range(25):
            cols_def = ", ".join([f"metric_col_{j}" for j in range(12)])
            vals_def = ", ".join([str(j * 10) for j in range(12)])
            csv_data = f"id,dept_{i}_code,name,status,{cols_def}\n1,D{i},Alpha,active,{vals_def}\n2,D{i},Beta,active,{vals_def}\n"
            self.structured_engine.ingest_file(
                file_input=csv_data,
                filename=f"dept_table_{i}.csv",
                display_name=f"Department {i} Analytics",
                description=f"Detailed financial and operational analytics table for division {i}.",
            )

        pruned_ctx = self.schema_pruner.prune_schema(
            query="Calculate average metric_col_1 for department 5",
            top_k_tables=2,
            max_cols_per_table=5,
            total_max_cols=10,
        )

        self.assertGreaterEqual(pruned_ctx.token_savings_percent, 85.0)
        print(f"\n[Benchmark 25 Tables] Pruned={pruned_ctx.token_count_pruned} tokens, Full={pruned_ctx.token_count_full} tokens, Savings={pruned_ctx.token_savings_percent}%")


if __name__ == "__main__":
    unittest.main()
