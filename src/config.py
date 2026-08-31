"""
Application Configuration Module for Multi-Agent Knowledge Base Q&A Platform.
Uses Pydantic V2 / pydantic-settings when available, with standard library fallback.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

# Root directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

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

        # Code Execution & Sandbox Security Limits
        sandbox_max_memory_mb: int = Field(default=512, alias="SANDBOX_MAX_MEMORY_MB")
        sandbox_timeout_sec: float = Field(default=5.0, alias="SANDBOX_TIMEOUT_SEC")

        # Two-Stage Schema Pruning Limits
        default_top_k_tables: int = Field(default=3, alias="DEFAULT_TOP_K_TABLES")
        default_max_cols_per_table: int = Field(default=8, alias="DEFAULT_MAX_COLS_PER_TABLE")
        default_total_max_cols: int = Field(default=20, alias="DEFAULT_TOTAL_MAX_COLS")

        # Observability (Langfuse)
        langfuse_enabled: bool = Field(default=False, alias="LANGFUSE_ENABLED")
        langfuse_public_key: Optional[str] = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key: Optional[str] = Field(default=None, alias="LANGFUSE_SECRET_KEY")
        langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

except ImportError:
    # Standard library fallback when pydantic-settings is not installed
    class Settings:  # type: ignore
        def __init__(self):
            self.db_host: str = os.getenv("POSTGRES_HOST", "localhost")
            self.db_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
            self.db_user: str = os.getenv("POSTGRES_USER", "postgres")
            self.db_password: str = os.getenv("POSTGRES_PASSWORD", "postgres")
            self.db_name: str = os.getenv("POSTGRES_DB", "knowledge_qa")
            self.database_url: str = os.getenv(
                "DATABASE_URL",
                f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}",
            )
            self.async_database_url: str = os.getenv(
                "ASYNC_DATABASE_URL",
                f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}",
            )

            raw_blob = os.getenv("BLOB_STORAGE_PATH")
            self.blob_storage_path: Path = (
                Path(raw_blob) if raw_blob else BASE_DIR / "data" / "blob"
            )

            raw_samples = os.getenv("SAMPLES_PATH")
            self.samples_path: Path = (
                Path(raw_samples) if raw_samples else BASE_DIR / "data" / "samples"
            )

            self.embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "fastembed")
            self.embedding_model: Optional[str] = os.getenv("EMBEDDING_MODEL") or None
            self.embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

            self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
            self.openai_api_url: Optional[str] = os.getenv("OPENAI_API_URL")
            self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

            self.sandbox_max_memory_mb: int = int(os.getenv("SANDBOX_MAX_MEMORY_MB", "512"))
            self.sandbox_timeout_sec: float = float(os.getenv("SANDBOX_TIMEOUT_SEC", "5.0"))

            self.default_top_k_tables: int = int(os.getenv("DEFAULT_TOP_K_TABLES", "3"))
            self.default_max_cols_per_table: int = int(os.getenv("DEFAULT_MAX_COLS_PER_TABLE", "8"))
            self.default_total_max_cols: int = int(os.getenv("DEFAULT_TOTAL_MAX_COLS", "20"))

            self.langfuse_enabled: bool = os.getenv("LANGFUSE_ENABLED", "false").lower() in (
                "true",
                "1",
            )
            self.langfuse_public_key: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
            self.langfuse_secret_key: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
            self.langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


@lru_cache
def get_settings() -> Settings:
    """Returns a cached instance of application settings."""
    settings = Settings()
    # Ensure base storage directories exist
    settings.blob_storage_path.mkdir(parents=True, exist_ok=True)
    settings.samples_path.mkdir(parents=True, exist_ok=True)
    return settings
