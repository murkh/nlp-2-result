"""
AST Security Validator for Sandboxed Python / Pandas DataFrame Execution.
Enforces multi-layer defense-in-depth:
1. Whitelist of permitted data science modules (pandas, polars, numpy, math, datetime, json, re)
2. Blacklist of dangerous built-in functions (eval, exec, open, compile, __import__, getattr, setattr, etc.)
3. Blacklist of dunder attribute escapes (__class__, __bases__, __subclasses__, __globals__, etc.)
"""

import ast
from typing import List, Set, Tuple


class ASTSecurityValidator(ast.NodeVisitor):
    """
    Traverses the Abstract Syntax Tree (AST) of user/LLM-generated Python code
    and flags any violations of security constraints.
    """

    ALLOWED_MODULES: Set[str] = {
        "pandas",
        "polars",
        "numpy",
        "math",
        "datetime",
        "json",
        "re",
    }

    FORBIDDEN_CALLS: Set[str] = {
        "eval",
        "exec",
        "open",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
        "help",
        "exit",
        "quit",
        "super",
        "memoryview",
    }

    FORBIDDEN_ATTRS: Set[str] = {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__dict__",
        "__import__",
        "__loader__",
        "__spec__",
        "__package__",
    }

    def __init__(self):
        self.errors: List[str] = []

    def visit_Import(self, node: ast.Import):
        """Verify that imported modules belong to the whitelist."""
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod not in self.ALLOWED_MODULES:
                self.errors.append(f"Forbidden module import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Verify that from-imported modules belong to the whitelist."""
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod not in self.ALLOWED_MODULES:
                self.errors.append(f"Forbidden module from-import: '{node.module}'")
        else:
            self.errors.append("Relative from-imports are forbidden")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Verify that forbidden built-in functions are not invoked."""
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in self.FORBIDDEN_CALLS:
                self.errors.append(f"Forbidden built-in call: '{func.id}()'")
        elif isinstance(func, ast.Attribute):
            if func.attr in self.FORBIDDEN_CALLS:
                self.errors.append(f"Forbidden method call: '{func.attr}()'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """Verify that dunder attributes used for sandbox escape are blocked."""
        if node.attr in self.FORBIDDEN_ATTRS:
            self.errors.append(f"Forbidden dunder attribute access: '{node.attr}'")
        self.generic_visit(node)


def validate_python_code(code: str) -> Tuple[bool, List[str]]:
    """
    Validate Python code using the ASTSecurityValidator.
    
    Returns:
        Tuple of (is_valid: bool, violations: List[str])
    """
    if not code or not code.strip():
        return False, ["Empty Python code"]

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Python syntax error: {str(e)}"]

    validator = ASTSecurityValidator()
    validator.visit(tree)

    return len(validator.errors) == 0, validator.errors
