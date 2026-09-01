"""
Adversarial Stress Test Suite for Milestone 2 (Challenger 1).
Exhaustively stress-tests:
1. AST Security Validator: Whitelists, forbidden calls, dunder escapes, and AST bypass vulnerabilities.
2. Subprocess Watchdog & Sandbox Isolation: CPU spin timeouts, environment sanitation, error isolation.
"""

import math
import os
import tempfile
import time
import unittest

from src.api.schemas import QueryPandasSandboxRequest
from src.engines.pandas_sandbox.ast_validator import (
    ASTSecurityValidator,
    validate_python_code,
)
from src.engines.pandas_sandbox.engine import PandasSandboxEngine
from src.engines.pandas_sandbox.runner import execute_sandboxed_code


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
            self.assertTrue(
                any("Forbidden module" in e for e in errs), f"Unexpected err msg for {mod}: {errs}"
            )

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
            self.assertTrue(
                any("Forbidden dunder" in e for e in errs), f"Errors for {dunder}: {errs}"
            )

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
        self.assertTrue(
            valid,
            "Vulnerability confirmed: AST validator permits 'f = eval' assignment without error",
        )
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
        self.assertTrue(
            valid, "Vulnerability confirmed: Walrus operator (NamedExpr) bypasses visit_Call"
        )

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
        self.assertTrue(
            valid, "Vulnerability confirmed: object.__getattribute__ bypasses AST validator"
        )

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
        self.assertLess(
            elapsed, 2.0, "Watchdog failed to terminate process within reasonable buffer"
        )

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
        self.assertFalse(
            res["data"]["has_secret"], "Security violation: Sensitive env var leaked into sandbox!"
        )


if __name__ == "__main__":
    unittest.main()
