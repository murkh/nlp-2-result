"""
Application Configuration Module for Multi-Agent Knowledge Base Q&A Platform.
Uses Pydantic V2 / pydantic-settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Root directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database Configuration
    db_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    db_port: int = Field(default=5432, alias="POSTGRES_PORT")
    db_user: str = Field(default="postgres", alias="POSTGRES_USER")
    db_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")
    db_name: str = Field(default="knowledge_qa", alias="POSTGRES_DB")
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/knowledge_qa",
        alias="DATABASE_URL",
    )
    async_database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/knowledge_qa",
        alias="ASYNC_DATABASE_URL",
    )

    # Storage Paths
    blob_storage_path: Path = Field(
        default=BASE_DIR / "data" / "blob",
        alias="BLOB_STORAGE_PATH",
    )
    samples_path: Path = Field(
        default=BASE_DIR / "data" / "samples",
        alias="SAMPLES_PATH",
    )

    # Embedding Configuration
    embedding_provider: Literal["mock", "openai", "fastembed"] = Field(
        default="fastembed",
        alias="EMBEDDING_PROVIDER",
    )
    # None resolves to the provider's default model (see DEFAULT_EMBEDDING_MODELS)
    embedding_model: Optional[str] = Field(
        default=None,
        alias="EMBEDDING_MODEL",
    )
    embedding_dim: int = Field(
        default=384,
        alias="EMBEDDING_DIM",
    )

    # LLM & Reasoning Configuration
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_api_url: Optional[str] = Field(default=None, alias="OPENAI_API_URL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")

    # Projection Critic
    # Widening a SELECT list against a given DDL is a small task, so it can run on a
    # cheaper/faster model than the main generator. None falls back to openai_model,
    # which keeps the feature safe behind an OpenAI-compatible gateway (bedrock-mantle)
    # that only serves a specific set of model ids.
    critic_model: Optional[str] = Field(default=None, alias="CRITIC_MODEL")
    projection_critic_enabled: bool = Field(default=True, alias="PROJECTION_CRITIC_ENABLED")

    # Structured Self-Correction Loop
    # The retry budget is safe only because every execution path is read-only
    # (FORBIDDEN_SQL_PATTERNS, FORBIDDEN_DUCKDB_PATTERNS, the AST whitelist).
    # If that surface ever permits mutation, remove the loop or make it idempotent.
    #
    # max_iters is 2 because the first correction pass carries almost all of the
    # measured gain and later passes mostly add tokens.
    structured_loop_enabled: bool = Field(default=True, alias="STRUCTURED_LOOP_ENABLED")
    structured_loop_max_iters: int = Field(default=2, alias="STRUCTURED_LOOP_MAX_ITERS")
    schema_exploration_enabled: bool = Field(default=True, alias="SCHEMA_EXPLORATION_ENABLED")
    loop_tool_call_budget: int = Field(default=4, alias="LOOP_TOOL_CALL_BUDGET")
    # Execution feedback carries the gains; model introspection is opt-in.
    reflection_enabled: bool = Field(default=False, alias="REFLECTION_ENABLED")

    # Code Execution & Sandbox Security Limits
    sandbox_max_memory_mb: int = Field(default=512, alias="SANDBOX_MAX_MEMORY_MB")
    sandbox_timeout_sec: float = Field(default=5.0, alias="SANDBOX_TIMEOUT_SEC")

    # Two-Stage Schema Pruning Limits
    default_top_k_tables: int = Field(default=3, alias="DEFAULT_TOP_K_TABLES")
    default_max_cols_per_table: int = Field(default=8, alias="DEFAULT_MAX_COLS_PER_TABLE")
    default_total_max_cols: int = Field(default=20, alias="DEFAULT_TOTAL_MAX_COLS")

    # Supervisor Routing Confidence Tiers
    # >= semantic threshold: local fast path. Between the two: LLM fallback.
    # < ambiguity threshold: clarification guardrail.
    router_semantic_threshold: float = Field(default=0.72, alias="ROUTER_SEMANTIC_THRESHOLD")
    router_ambiguity_threshold: float = Field(default=0.50, alias="ROUTER_AMBIGUITY_THRESHOLD")

    # Observability (Langfuse)
    langfuse_enabled: bool = Field(default=False, alias="LANGFUSE_ENABLED")
    langfuse_public_key: Optional[str] = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")


@lru_cache
def get_settings() -> Settings:
    """Returns a cached instance of application settings."""
    settings = Settings()
    # Ensure base storage directories exist
    settings.blob_storage_path.mkdir(parents=True, exist_ok=True)
    settings.samples_path.mkdir(parents=True, exist_ok=True)
    return settings
