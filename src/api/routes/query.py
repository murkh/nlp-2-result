"""
Query API Routes for Multi-Strategy Knowledge Base Execution.
Provides dedicated endpoints for:
- POST /query/dedicated-db (Strategy A: PostgreSQL Text2SQL)
- POST /query/duckdb (Strategy B: DuckDB In-Memory)
- POST /query/pandas-sandbox (Strategy C: Sandboxed Python DataFrame)
- POST /query/unstructured-rag (Hybrid Dense + Sparse RAG with Citations)
- POST /query/benchmark (Parallel 3-Way Benchmark Arena)
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter

from src.api.schemas import (
    QueryBenchmarkRequest,
    QueryBenchmarkResponse,
    QueryDedicatedDBRequest,
    QueryDedicatedDBResponse,
    QueryDuckDBRequest,
    QueryDuckDBResponse,
    QueryPandasSandboxRequest,
    QueryPandasSandboxResponse,
    QueryUnstructuredRAGRequest,
    QueryUnstructuredRAGResponse,
)
from src.database.connection import get_db_manager
from src.engines.benchmark_arena import BenchmarkArenaEngine
from src.engines.dedicated_db import DedicatedDBEngine
from src.engines.duckdb_engine import DuckDBQueryEngine
from src.engines.hybrid_rag import HybridRAGEngine
from src.engines.pandas_sandbox.engine import PandasSandboxEngine

router = APIRouter(prefix="/query", tags=["Query Execution Engines"])


@router.post("/dedicated-db", response_model=QueryDedicatedDBResponse)
async def query_dedicated_db_endpoint(request: QueryDedicatedDBRequest) -> QueryDedicatedDBResponse:
    """Strategy A: Dedicated PostgreSQL Text2SQL query engine with read-only transaction and LIMIT 20."""
    engine = DedicatedDBEngine(db_manager=get_db_manager())
    return engine.execute_query(request)


@router.post("/duckdb", response_model=QueryDuckDBResponse)
async def query_duckdb_endpoint(request: QueryDuckDBRequest) -> QueryDuckDBResponse:
    """Strategy B: In-Memory DuckDB query engine over raw blob Parquet/CSV files."""
    engine = DuckDBQueryEngine(db_manager=get_db_manager())
    return engine.execute_query(request)


@router.post("/pandas-sandbox", response_model=QueryPandasSandboxResponse)
async def query_pandas_sandbox_endpoint(
    request: QueryPandasSandboxRequest,
) -> QueryPandasSandboxResponse:
    """Strategy C: Sandboxed Python DataFrame execution with AST validation and resource watchdog."""
    engine = PandasSandboxEngine(db_manager=get_db_manager())
    return engine.execute_query(request)


@router.post("/unstructured-rag", response_model=QueryUnstructuredRAGResponse)
async def query_unstructured_rag_endpoint(
    request: QueryUnstructuredRAGRequest,
) -> QueryUnstructuredRAGResponse:
    """Unstructured Hybrid RAG with Reciprocal Rank Fusion (RRF) and bracketed source citations."""
    engine = HybridRAGEngine(db_manager=get_db_manager())
    return engine.execute_query(request)


@router.post("/benchmark", response_model=QueryBenchmarkResponse)
async def query_benchmark_endpoint(request: QueryBenchmarkRequest) -> QueryBenchmarkResponse:
    """Parallel 3-Way Benchmark Arena comparing Strategy A, B, and C with equivalence checking."""
    engine = BenchmarkArenaEngine(db_manager=get_db_manager())
    return engine.execute_benchmark(request)
