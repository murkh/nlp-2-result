"""
Unit and Security Tests for Strategy C: Sandboxed Python DataFrame Execution.
Verifies AST security validation against adversarial payloads,
subprocess timeout isolation, DataFrame result serialization, and response synthesis.
"""

from pathlib import Path
import unittest

from src.api.schemas import QueryPandasSandboxRequest
from src.engines.pandas_sandbox.ast_validator import validate_python_code
from src.engines.pandas_sandbox.engine import PandasSandboxEngine
from src.engines.pandas_sandbox.runner import execute_sandboxed_code
from tests.conftest import create_test_fixtures


class TestPandasSandboxEngine(unittest.TestCase):
    """Test suite for Strategy C Pandas Sandbox."""

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
            filename="orders.csv",
            display_name="Orders Data",
            description="Blob storage dataset for Pandas sandbox.",
        )

        self.settings = fixtures.get("settings")
        self.engine = PandasSandboxEngine(
            db_manager=self.db_manager,
            schema_pruner=self.schema_pruner,
            settings=self.settings,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # AST Security Attack Payloads
    # -------------------------------------------------------------------------
    def test_ast_blocks_unauthorized_imports(self):
        """Verify AST rejects forbidden module imports."""
        bad_scripts = [
            "import os",
            "import sys",
            "import subprocess",
            "import shutil",
            "import socket",
            "import urllib.request",
            "import pickle",
            "import ctypes",
            "from os import system",
            "from subprocess import Popen, PIPE",
            "from shutil import rmtree",
        ]
        for script in bad_scripts:
            ok, errors = validate_python_code(script)
            self.assertFalse(ok, f"Expected AST rejection for: {script}")
            self.assertGreater(len(errors), 0)

    def test_ast_blocks_dangerous_builtins(self):
        """Verify AST rejects dangerous built-in function calls."""
        bad_calls = [
            "x = eval('1 + 1')",
            "exec('import os')",
            "f = open('/etc/passwd', 'r')",
            "mod = __import__('os')",
            "val = getattr(dict, 'keys')",
            "setattr(dict, 'x', 1)",
            "g = globals()",
            "l = locals()",
            "exit(0)",
            "quit()",
        ]
        for script in bad_calls:
            ok, errors = validate_python_code(script)
            self.assertFalse(ok, f"Expected AST rejection for: {script}")
            self.assertGreater(len(errors), 0)

    def test_ast_blocks_dunder_escapes(self):
        """Verify AST blocks Python object model traversal and sandbox escapes."""
        bad_dunders = [
            "().__class__.__bases__[0].__subclasses__()",
            "[].__class__.__mro__",
            "x = ().__class__.__globals__",
            "x = ().__builtins__",
            "x = ().__dict__",
        ]
        for script in bad_dunders:
            ok, errors = validate_python_code(script)
            self.assertFalse(ok, f"Expected AST rejection for: {script}")
            self.assertGreater(len(errors), 0)

    def test_ast_permits_safe_modules(self):
        """Verify AST permits whitelisted data science modules."""
        safe_scripts = [
            "import pandas as pd\nimport numpy as np\nresult = {'sum': 10}",
            "import math\nimport datetime\nimport json\nimport re\nresult = [1, 2, 3]",
            "import polars as pl\nresult = {'status': 'ok'}",
        ]
        for script in safe_scripts:
            ok, errors = validate_python_code(script)
            self.assertTrue(ok, f"Expected AST approval for: {script}, got errors: {errors}")
            self.assertEqual(len(errors), 0)

    # -------------------------------------------------------------------------
    # Subprocess Timeout Watchdog
    # -------------------------------------------------------------------------
    def test_subprocess_timeout_watchdog(self):
        """Verify subprocess execution watchdog terminates infinite loops."""
        infinite_loop = "while True: pass"
        ok, res, err, rc = execute_sandboxed_code(infinite_loop, timeout_seconds=0.5)
        self.assertFalse(ok)
        self.assertTrue("timed out" in err.lower() or rc == 124)

    # -------------------------------------------------------------------------
    # End-to-End Query Execution
    # -------------------------------------------------------------------------
    def test_pandas_sandbox_count_query(self):
        """Verify Strategy C executes count aggregation and returns structured telemetry."""
        req = QueryPandasSandboxRequest(query="How many total orders exist?")
        resp = self.engine.execute_query(req)

        self.assertIsNone(resp.error)
        self.assertTrue(resp.security_report.ast_passed)
        self.assertFalse(resp.security_report.timeout_occurred)
        self.assertEqual(resp.tabular_result.row_count, 1)
        self.assertEqual(int(resp.tabular_result.rows[0]["total_records"]), 5)
        self.assertIn("5", resp.answer)

    def test_pandas_sandbox_sum_query(self):
        """Verify Strategy C calculates total revenue via sandboxed Python."""
        req = QueryPandasSandboxRequest(query="What is the total sales revenue?")
        resp = self.engine.execute_query(req)

        self.assertIsNone(resp.error)
        self.assertTrue(resp.security_report.ast_passed)
        self.assertEqual(resp.tabular_result.row_count, 1)
        val = float(resp.tabular_result.rows[0]["total_revenue"])
        self.assertAlmostEqual(val, 1061.55, places=2)


if __name__ == "__main__":
    unittest.main()
