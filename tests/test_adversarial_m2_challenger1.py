"""
Adversarial Stress Test Suite for Milestone 2 (Challenger 1).
Exhaustively stress-tests:
1. AST Security Validator: Whitelists, forbidden calls, dunder escapes, and AST bypass vulnerabilities.
2. Subprocess Watchdog & Sandbox Isolation: CPU spin timeouts, environment sanitation, error isolation.
3. Benchmark Arena Equivalence: Float/Int tolerances, NaN/Inf, string normalization, schema variances, concurrent thread safety.
"""

import math
import os
import tempfile
import time
import unittest

from src.api.schemas import (
    QueryBenchmarkRequest,
    QueryPandasSandboxRequest,
    TabularResult,
)
from src.engines.benchmark_arena import (
    BenchmarkArenaEngine,
    are_values_equivalent,
    compare_tabular_results,
)
from src.engines.pandas_sandbox.ast_validator import (
    ASTSecurityValidator,
    validate_python_code,
)
from src.engines.pandas_sandbox.engine import PandasSandboxEngine
from src.engines.pandas_sandbox.runner import execute_sandboxed_code
from tests.conftest import create_test_fixtures


class TestASTSecurityAdversarial(unittest.TestCase):
    """Adversarial stress testing of the AST Security Validator."""

    # -------------------------------------------------------------------------
    # 1. Module Import Boundary Tests
    # -------------------------------------------------------------------------
    def test_ast_blocks_dangerous_direct_imports(self):
        """Verify AST validator rejects all dangerous standard library and system modules."""
        forbidden_modules = [
            "os",
            "sys",
            "subprocess",
            "shutil",
            "socket",
            "ctypes",
            "pathlib",
            "http",
            "urllib",
            "io",
            "importlib",
            "inspect",
            "builtins",
            "pickle",
            "shelve",
            "pty",
            "commands",
            "multiprocessing",
            "threading",
            "signal",
        ]
        for mod in forbidden_modules:
            code = f"import {mod}"
            valid, errs = validate_python_code(code)
            self.assertFalse(valid, f"Expected rejection for: {code}")
            self.assertTrue(any("Forbidden module" in e for e in errs), f"Unexpected err msg for {mod}: {errs}")

    def test_ast_blocks_from_imports_and_submodules(self):
        """Verify from-import and nested submodule import variations are rejected."""
        bad_from_imports = [
            "from os import system",
            "from os.path import exists",
            "from subprocess import Popen, PIPE, run",
            "from sys import modules, exit",
            "from urllib.request import urlopen",
            "from http.client import HTTPSConnection",
            "from pathlib import Path",
            "from io import StringIO, BytesIO",
            "from importlib import import_module",
        ]
        for code in bad_from_imports:
            valid, errs = validate_python_code(code)
            self.assertFalse(valid, f"Expected rejection for: {code}")
            self.assertGreater(len(errs), 0)

    def test_ast_blocks_aliased_and_multi_imports(self):
        """Verify aliased imports (as alias) and multi-imports in single line are rejected."""
        multi_imports = [
            "import os as safe_os",
            "import sys as safe_sys",
            "import pandas as pd, os as evil_os",
            "import math, numpy, subprocess",
            "from os import system as safe_system",
        ]
        for code in multi_imports:
            valid, errs = validate_python_code(code)
            self.assertFalse(valid, f"Expected rejection for: {code}")

    def test_ast_permits_whitelisted_data_science_modules(self):
        """Verify permitted modules (pandas, polars, numpy, math, datetime, json, re) pass AST validation."""
        valid_scripts = [
            "import pandas as pd\nresult = pd.DataFrame([{'a': 1}])",
            "import polars as pl\nresult = {'status': 1}",
            "import numpy as np\nresult = {'mean': 42.0}",
            "import math\nresult = {'pi': math.pi}",
            "import datetime\nresult = {'now': str(datetime.date.today())}",
            "import json\nresult = json.loads('{\"x\": 1}')",
            "import re\nresult = {'match': bool(re.match(r'\\d+', '123'))}",
        ]
        for code in valid_scripts:
            valid, errs = validate_python_code(code)
            self.assertTrue(valid, f"Expected approval for: {code}, got errors: {errs}")
            self.assertEqual(len(errs), 0)

    # -------------------------------------------------------------------------
    # 2. Builtin & Dunder Attack Payloads
    # -------------------------------------------------------------------------
    def test_ast_blocks_direct_forbidden_calls(self):
        """Verify direct calls to blacklisted built-in functions are blocked."""
        forbidden_calls = [
            "eval('1+1')",
            "exec('x = 1')",
            "open('/etc/passwd', 'r')",
            "compile('1+1', '<string>', 'eval')",
            "__import__('os')",
            "getattr(dict, 'items')",
            "setattr(dict, 'x', 1)",
            "delattr(dict, 'x')",
            "globals()",
            "locals()",
            "vars()",
            "breakpoint()",
            "exit(0)",
            "quit()",
            "input()",
            "help(str)",
        ]
        for call_expr in forbidden_calls:
            code = f"result = {call_expr}"
            valid, errs = validate_python_code(code)
            self.assertFalse(valid, f"Expected rejection for call: {call_expr}")
            self.assertTrue(any("Forbidden" in e for e in errs), f"Errors for {call_expr}: {errs}")

    def test_ast_blocks_dunder_attributes(self):
        """Verify AST blocks object traversal and reflection dunder attributes."""
        dunder_payloads = [
            "().__class__",
            "().__class__.__bases__",
            "().__class__.__subclasses__()",
            "[].__class__.__mro__",
            "(1).__class__.__globals__",
            "(lambda: None).__code__",
            "().__builtins__",
            "().__dict__",
            "().__loader__",
            "().__spec__",
            "().__package__",
        ]
        for dunder in dunder_payloads:
            code = f"x = {dunder}"
            valid, errs = validate_python_code(code)
            self.assertFalse(valid, f"Expected rejection for dunder: {dunder}")
            self.assertTrue(any("Forbidden dunder" in e for e in errs), f"Errors for {dunder}: {errs}")

    # -------------------------------------------------------------------------
    # 3. Empirical Demonstration of AST Bypass Vulnerabilities
    # -------------------------------------------------------------------------
    def test_vulnerability_forbidden_call_name_aliasing(self):
        """
        EMPIRICAL CHALLENGE: Proves AST bypass via variable aliasing of forbidden built-ins.
        Because ASTSecurityValidator does not implement visit_Name, assigning a forbidden
        built-in to a variable name (e.g. `f = eval`) allows arbitrary execution.
        """
        exploit_eval = "f = eval\nresult = f('10 * 10')"
        # 1. AST Validation test (Bypasses AST check)
        valid, errs = validate_python_code(exploit_eval)
        self.assertTrue(valid, "Vulnerability confirmed: AST validator permits 'f = eval' assignment without error")
        self.assertEqual(len(errs), 0)

        # 2. Execution test (Runs successfully in sandbox)
        ok, res, err, rc = execute_sandboxed_code(exploit_eval)
        self.assertTrue(ok)
        self.assertEqual(res["data"], 100)

    def test_vulnerability_import_aliasing_os_execution(self):
        """
        EMPIRICAL CHALLENGE: Proves arbitrary OS execution via `im = __import__`.
        """
        exploit_import = "im = __import__\nos_mod = im('os')\nresult = {'cwd': os_mod.getcwd()}"
        valid, errs = validate_python_code(exploit_import)
        self.assertTrue(valid, "Vulnerability confirmed: AST validator permits 'im = __import__'")
        ok, res, err, rc = execute_sandboxed_code(exploit_import)
        self.assertTrue(ok)
        self.assertIn("cwd", res["data"])

    def test_vulnerability_walrus_operator_bypass(self):
        """
        EMPIRICAL CHALLENGE: Proves AST bypass using Python 3.8+ walrus operator (NamedExpr).
        `visit_Call` checks `isinstance(func, ast.Name)` and `isinstance(func, ast.Attribute)`.
        When func is `ast.NamedExpr` like `(f := eval)`, the check is skipped.
        """
        walrus_eval = "result = (f := eval)('50 + 50')"
        valid, errs = validate_python_code(walrus_eval)
        self.assertTrue(valid, "Vulnerability confirmed: Walrus operator (NamedExpr) bypasses visit_Call")

        ok, res, err, rc = execute_sandboxed_code(walrus_eval)
        self.assertTrue(ok)
        self.assertEqual(res["data"], 100)

    def test_vulnerability_object_getattribute_traversal(self):
        """
        EMPIRICAL CHALLENGE: Proves object traversal sandbox escape via `object.__getattribute__`.
        `__getattribute__` is not listed in `FORBIDDEN_ATTRS` or `FORBIDDEN_CALLS`.
        """
        exploit_getattribute = (
            "ga = object.__getattribute__\n"
            "cls = ga('hello', '__class__')\n"
            "base = ga(cls, '__base__')\n"
            "subs = ga(base, '__subclasses__')()\n"
            "result = {'subclasses_count': len(subs)}"
        )
        valid, errs = validate_python_code(exploit_getattribute)
        self.assertTrue(valid, "Vulnerability confirmed: object.__getattribute__ bypasses AST validator")

        ok, res, err, rc = execute_sandboxed_code(exploit_getattribute)
        self.assertTrue(ok)
        self.assertGreater(res["data"]["subclasses_count"], 50)


class TestSubprocessWatchdogAdversarial(unittest.TestCase):
    """Adversarial stress testing of Subprocess Runner watchdog & resource isolation."""

    def test_watchdog_terminates_infinite_cpu_loop(self):
        """Verify watchdog terminates CPU spinning loop strictly within configured timeout."""
        t0 = time.perf_counter()
        ok, res, err, rc = execute_sandboxed_code("while True: pass", timeout_seconds=0.5)
        elapsed = time.perf_counter() - t0

        self.assertFalse(ok)
        self.assertEqual(rc, 124)
        self.assertTrue("timed out" in err.lower())
        self.assertLess(elapsed, 2.0, "Watchdog failed to terminate process within reasonable buffer")

    def test_watchdog_terminates_large_iteration_loop(self):
        """Verify watchdog terminates huge range loop."""
        code = "for i in range(10**9): pass"
        ok, res, err, rc = execute_sandboxed_code(code, timeout_seconds=0.5)
        self.assertFalse(ok)
        self.assertEqual(rc, 124)

    def test_watchdog_isolates_syntax_error(self):
        """Verify invalid Python syntax fails cleanly without crashing parent process."""
        bad_syntax = "def invalid_syntax(:"
        ok, res, err, rc = execute_sandboxed_code(bad_syntax, timeout_seconds=1.0)
        self.assertFalse(ok)
        self.assertEqual(rc, 1)
        self.assertTrue("syntax" in err.lower() or "runtimeerror" in err.lower())

    def test_watchdog_isolates_runtime_exception(self):
        """Verify runtime exceptions (ZeroDivisionError) are captured in stderr without parent failure."""
        runtime_err = "x = 10 / 0"
        ok, res, err, rc = execute_sandboxed_code(runtime_err, timeout_seconds=1.0)
        self.assertFalse(ok)
        self.assertEqual(rc, 1)
        self.assertTrue("division by zero" in err.lower() or "runtimeerror" in err.lower())

    def test_environment_sanitation_no_sensitive_env_leakage(self):
        """Verify parent environment secrets are not passed to sandbox subprocess."""
        os.environ["SECRET_API_TOKEN_XYZ"] = "top_secret_token_12345"
        probe_code = "im = __import__\nos_mod = im('os')\nresult = {'has_secret': 'SECRET_API_TOKEN_XYZ' in os_mod.environ}"
        ok, res, err, rc = execute_sandboxed_code(probe_code, timeout_seconds=2.0)
        self.assertTrue(ok)
        self.assertFalse(res["data"]["has_secret"], "Security violation: Sensitive env var leaked into sandbox!")


class TestBenchmarkArenaAdversarial(unittest.TestCase):
    """Adversarial stress testing of Benchmark Arena equivalence and parallel execution."""

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
            display_name="Orders",
            description="E-commerce orders dataset for 3-way benchmarking.",
        )
        self.arena = BenchmarkArenaEngine(
            db_manager=self.db_manager,
            blob_manager=self.blob_manager,
            schema_pruner=self.schema_pruner,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. are_values_equivalent Edge Cases
    # -------------------------------------------------------------------------
    def test_value_equivalence_float_and_int_tolerances(self):
        """Verify numeric equivalence handles float/int mix, epsilon tolerance, and signed zeros."""
        self.assertTrue(are_values_equivalent(100, 100.0))
        self.assertTrue(are_values_equivalent(0, -0.0))
        self.assertTrue(are_values_equivalent(10.00001, 10.00002, tolerance=1e-4))
        self.assertFalse(are_values_equivalent(10.0, 11.0, tolerance=1e-4))

    def test_value_equivalence_string_case_and_whitespace(self):
        """Verify string equivalence normalizes whitespace and casing."""
        self.assertTrue(are_values_equivalent("COMPLETED", "completed"))
        self.assertTrue(are_values_equivalent(" New York ", "new york"))
        self.assertFalse(are_values_equivalent("New York", "San Francisco"))

    def test_value_equivalence_none_and_null(self):
        """Verify None handling in values equivalence."""
        self.assertTrue(are_values_equivalent(None, None))
        self.assertFalse(are_values_equivalent(None, 0))
        self.assertFalse(are_values_equivalent(None, ""))

    def test_value_equivalence_inf_and_nan(self):
        """Verify infinity and NaN behavior in are_values_equivalent."""
        self.assertTrue(are_values_equivalent(float("inf"), float("inf")))
        self.assertTrue(are_values_equivalent(float("-inf"), float("-inf")))
        self.assertFalse(are_values_equivalent(float("inf"), float("-inf")))
        self.assertFalse(are_values_equivalent(float("nan"), float("nan")))

    # -------------------------------------------------------------------------
    # 2. compare_tabular_results Edge Cases
    # -------------------------------------------------------------------------
    def test_tabular_equivalence_scalar_column_alias_invariance(self):
        """Verify single-value scalar results match even if column alias differs."""
        res_a = TabularResult(columns=["total_records"], rows=[{"total_records": 5}], row_count=1)
        res_b = TabularResult(columns=["count"], rows=[{"count": 5.0}], row_count=1)
        self.assertTrue(compare_tabular_results(res_a, res_b))

    def test_tabular_equivalence_column_order_invariance(self):
        """Verify records with different key ordering in dict are equivalent."""
        res_a = TabularResult(
            columns=["city", "revenue"],
            rows=[{"city": "NYC", "revenue": 100.0}, {"city": "SFO", "revenue": 200.0}],
            row_count=2,
        )
        res_b = TabularResult(
            columns=["revenue", "city"],
            rows=[{"revenue": 100.0, "city": "nyc"}, {"revenue": 200.0, "city": "sfo"}],
            row_count=2,
        )
        self.assertTrue(compare_tabular_results(res_a, res_b))

    def test_tabular_equivalence_row_count_mismatch(self):
        """Verify row count mismatch immediately fails equivalence."""
        res_a = TabularResult(columns=["status"], rows=[{"status": "completed"}], row_count=1)
        res_b = TabularResult(columns=["status"], rows=[{"status": "completed"}, {"status": "pending"}], row_count=2)
        self.assertFalse(compare_tabular_results(res_a, res_b))

    def test_tabular_equivalence_empty_tables(self):
        """Verify two empty results (0 rows) evaluate as equivalent."""
        res_a = TabularResult(columns=["a", "b"], rows=[], row_count=0)
        res_b = TabularResult(columns=["x", "y"], rows=[], row_count=0)
        self.assertTrue(compare_tabular_results(res_a, res_b))

    def test_tabular_equivalence_schema_column_subset_anomaly(self):
        """
        EMPIRICAL OBSERVATION: Tests partial key intersection behavior in compare_tabular_results.
        """
        res_3cols = TabularResult(
            columns=["city", "revenue", "extra_metric"],
            rows=[{"city": "NYC", "revenue": 100.0, "extra_metric": 99.9}],
            row_count=1,
        )
        res_2cols = TabularResult(
            columns=["city", "revenue"],
            rows=[{"city": "NYC", "revenue": 100.0}],
            row_count=1,
        )
        self.assertTrue(compare_tabular_results(res_3cols, res_2cols))

    # -------------------------------------------------------------------------
    # 3. Concurrent Multi-Threaded Arena Execution
    # -------------------------------------------------------------------------
    def test_arena_concurrent_execution_thread_safety(self):
        """Verify concurrent execution of Strategy A, B, and C in parallel ThreadPool."""
        req = QueryBenchmarkRequest(query="What is the total sales revenue?")
        resp = self.arena.execute_benchmark(req)

        self.assertEqual(resp.strategy_a.status, "SUCCESS")
        self.assertEqual(resp.strategy_b.status, "SUCCESS")
        self.assertEqual(resp.strategy_c.status, "SUCCESS")
        self.assertTrue(resp.benchmark_summary.consensus_reached)

        val_a = float(resp.strategy_a.tabular_result.rows[0]["total_revenue"])
        val_b = float(resp.strategy_b.tabular_result.rows[0]["total_revenue"])
        val_c = float(resp.strategy_c.tabular_result.rows[0]["total_revenue"])
        self.assertAlmostEqual(val_a, 1061.55, places=2)
        self.assertAlmostEqual(val_b, 1061.55, places=2)
        self.assertAlmostEqual(val_c, 1061.55, places=2)


if __name__ == "__main__":
    unittest.main()
