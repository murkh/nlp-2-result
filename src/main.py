"""
FastAPI Application Entrypoint for Multi-Agent Knowledge Base Q&A Platform.
Provides CORS configuration, lifespan startup initialization,
router registration, and healthcheck endpoints.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict

from src.api.routes import agent, ingest, query
from src.config import get_settings
from src.database.connection import get_db_manager
from src.storage.blob_store import get_blob_manager

try:
    from fastapi import FastAPI, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    _has_fastapi = True
except ImportError:
    _has_fastapi = False


@asynccontextmanager
async def lifespan(app: Any):
    """Lifespan event handler initializing database and blob directories on startup."""
    # Ensure storage paths exist
    settings = get_settings()
    settings.blob_storage_path.mkdir(parents=True, exist_ok=True)
    settings.samples_path.mkdir(parents=True, exist_ok=True)

    # Initialize DB connection and tables
    db_mgr = get_db_manager()
    _ = get_blob_manager()

    # Seed sample datasets if database catalog is empty
    try:
        if len(db_mgr.list_datasets()) == 0:
            from scripts.seed_data import seed_all_datasets
            seed_all_datasets()
    except Exception as e:
        print(f"[Lifespan] Auto-seeding notice: {e}")

    yield


def create_app() -> Any:
    """Application factory for FastAPI app with middleware and routers."""
    if not _has_fastapi:
        # Fallback application placeholder for stdlib environments
        class SimpleApp:
            def __init__(self):
                self.title = "Multi-Agent Knowledge Base Q&A Platform"
                self.version = "0.1.0"
            def get(self, *args, **kwargs):
                return lambda f: f
            def post(self, *args, **kwargs):
                return lambda f: f
        return SimpleApp()

    app = FastAPI(
        title="Multi-Agent Knowledge Base Q&A Platform",
        description=(
            "Production-grade, token-efficient Multi-Agent Knowledge Base Q&A API "
            "supporting structured and unstructured data with three execution strategies "
            "(PostgreSQL Dedicated DB, DuckDB In-Memory, Pandas Sandbox) and parallel benchmarking."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS Middleware (allows Streamlit frontend integration)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include Routers
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(agent.router)

    @app.get("/health", tags=["System"])
    async def healthcheck() -> Dict[str, str]:
        """System health and readiness check."""
        return {
            "status": "healthy",
            "service": "multiagent-knowledge-qa",
            "version": "0.1.0",
        }

    @app.get("/", tags=["System"])
    async def root() -> Dict[str, str]:
        """Root API information."""
        return {
            "title": "Multi-Agent Knowledge Base Q&A Platform",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
