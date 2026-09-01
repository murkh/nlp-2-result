"""
Query API Routes for Knowledge Base Execution.
Provides dedicated endpoints for:
- POST /query/pandas-sandbox (Sandboxed Python DataFrame execution)
- POST /query/unstructured-rag (Hybrid Dense + Sparse RAG with Citations)
"""

from fastapi import APIRouter

from src.api.schemas import (
    QueryPandasSandboxRequest,
    QueryPandasSandboxResponse,
    QueryUnstructuredRAGRequest,
    QueryUnstructuredRAGResponse,
)
from src.database.connection import get_db_manager
from src.engines.hybrid_rag import HybridRAGEngine
from src.engines.pandas_sandbox.engine import PandasSandboxEngine

router = APIRouter(prefix="/query", tags=["Query Execution Engines"])


@router.post("/pandas-sandbox", response_model=QueryPandasSandboxResponse)
async def query_pandas_sandbox_endpoint(
    request: QueryPandasSandboxRequest,
) -> QueryPandasSandboxResponse:
    """Sandboxed Python DataFrame execution with AST validation and resource watchdog."""
    engine = PandasSandboxEngine(db_manager=get_db_manager())
    return engine.execute_query(request)


@router.post("/unstructured-rag", response_model=QueryUnstructuredRAGResponse)
async def query_unstructured_rag_endpoint(
    request: QueryUnstructuredRAGRequest,
) -> QueryUnstructuredRAGResponse:
    """Unstructured Hybrid RAG with Reciprocal Rank Fusion (RRF) and bracketed source citations."""
    engine = HybridRAGEngine(db_manager=get_db_manager())
    return engine.execute_query(request)
