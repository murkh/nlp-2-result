"""
One interface over the three structured strategies.

The loop nodes generate, validate and execute without knowing which engine is
behind them. Each adapter delegates to the engine's existing public methods, so
the guardrails and execution paths stay exactly the ones `/query/*` uses.
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple

from src.database.connection import get_db_manager

Generated = Tuple[str, Optional[str], Tuple[int, int]]
Executed = Tuple[List[str], List[Dict[str, Any]], Optional[str]]


class StrategyAdapter(Protocol):
    strategy: str
    supports_probes: bool

    def generate(
        self,
        query: str,
        ddl: str,
        observations: List[Dict[str, Any]],
        schema_context: Dict[str, Any],
    ) -> Generated: ...

    def prepare(self, code: str, schema_context: Dict[str, Any]) -> str: ...

    def validate(self, code: str) -> Tuple[bool, Optional[str]]: ...

    def execute(self, code: str, schema_context: Dict[str, Any]) -> Executed: ...


class _SqlAdapter:
    """
    Shared behavior for the two SQL engines.

    The engine is built on first use so validation and routing cost no database
    connection.
    """

    supports_probes = True

    def __init__(self, engine_factory: Any, validator: Any) -> None:
        self._engine_factory = engine_factory
        self._validate = validator
        self._cached_engine: Optional[Any] = None

    @property
    def _engine(self) -> Any:
        if self._cached_engine is None:
            self._cached_engine = self._engine_factory()
        return self._cached_engine

    def generate(
        self,
        query: str,
        ddl: str,
        observations: List[Dict[str, Any]],
        schema_context: Dict[str, Any],
    ) -> Generated:
        return self._engine.generate_code(query, ddl, observations)

    def prepare(self, code: str, schema_context: Dict[str, Any]) -> str:
        return code

    def validate(self, code: str) -> Tuple[bool, Optional[str]]:
        return self._validate(code)

    def execute(self, code: str, schema_context: Dict[str, Any]) -> Executed:
        return self._engine.execute_sql(code, schema_context)


class DuckDBAdapter(_SqlAdapter):
    strategy = "duckdb"

    def __init__(self) -> None:
        from src.engines.duckdb_engine import DuckDBQueryEngine, validate_duckdb_security

        super().__init__(
            lambda: DuckDBQueryEngine(db_manager=get_db_manager()),
            validate_duckdb_security,
        )


class DedicatedDBAdapter(_SqlAdapter):
    strategy = "dedicated_db"

    def __init__(self) -> None:
        from src.engines.dedicated_db import DedicatedDBEngine, validate_sql_security

        super().__init__(
            lambda: DedicatedDBEngine(db_manager=get_db_manager()),
            validate_sql_security,
        )


class PandasSandboxAdapter:
    """Strategy C has no SQL surface, so it gets no probes."""

    strategy = "pandas_sandbox"
    supports_probes = False

    def __init__(self) -> None:
        self._cached_engine: Optional[Any] = None

    @property
    def _engine(self) -> Any:
        if self._cached_engine is None:
            from src.engines.pandas_sandbox.engine import PandasSandboxEngine

            self._cached_engine = PandasSandboxEngine(db_manager=get_db_manager())
        return self._cached_engine

    def generate(
        self,
        query: str,
        ddl: str,
        observations: List[Dict[str, Any]],
        schema_context: Dict[str, Any],
    ) -> Generated:
        return self._engine.generate_code(
            query,
            ddl,
            observations,
            file_paths=schema_context.get("file_paths") or {},
        )

    def prepare(self, code: str, schema_context: Dict[str, Any]) -> str:
        return self._engine.apply_dataset_loader(
            code,
            file_paths=schema_context.get("file_paths") or {},
            table_names=schema_context.get("table_names") or [],
        )

    def validate(self, code: str) -> Tuple[bool, Optional[str]]:
        from src.engines.pandas_sandbox.ast_validator import validate_python_code

        return validate_python_code(code)

    def execute(self, code: str, schema_context: Dict[str, Any]) -> Executed:
        return self._engine.execute_code(code)


_ADAPTERS = {
    "duckdb": DuckDBAdapter,
    "dedicated_db": DedicatedDBAdapter,
    "pandas_sandbox": PandasSandboxAdapter,
}


def get_adapter(strategy: Optional[str]) -> StrategyAdapter:
    """Adapter for a strategy name. An unknown name is a programming error."""
    resolved = strategy or "duckdb"
    if resolved not in _ADAPTERS:
        raise ValueError(
            f"Unknown structured strategy '{strategy}'. Expected one of {sorted(_ADAPTERS)}."
        )
    return _ADAPTERS[resolved]()


def probes_available(adapter: StrategyAdapter, schema_context: Dict[str, Any]) -> bool:
    """Probes need a SQL surface and at least one retained column to name."""
    return bool(adapter.supports_probes and (schema_context.get("retained_columns") or {}))
