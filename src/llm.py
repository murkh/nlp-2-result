"""
Shared OpenAI Client Provider.
Builds a single client from application settings (API key and optional
OpenAI-compatible base URL) reused across engines, router, and ingestion.
"""

import logging
from typing import Any, Optional

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

_client: Optional[Any] = None


class LLMUnavailableError(RuntimeError):
    """Raised when the configured LLM cannot be reached or returns an unusable response."""


def _build_client(settings: Settings) -> Optional[Any]:
    if not settings.openai_api_key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_url,
        )
    except Exception:
        logger.exception(
            "Failed to build OpenAI client (base_url=%s, model=%s)",
            settings.openai_api_url,
            settings.openai_model,
        )
        return None


def get_openai_client(settings: Optional[Settings] = None) -> Optional[Any]:
    """Returns a shared OpenAI client, or None when unavailable."""
    global _client
    if settings is not None:
        return _build_client(settings)
    if _client is None:
        _client = _build_client(get_settings())
    return _client


def require_openai_client(settings: Optional[Settings] = None) -> Any:
    """Returns an OpenAI client or raises LLMUnavailableError naming the configuration used."""
    resolved = settings or get_settings()
    client = get_openai_client(resolved)
    if client is None:
        raise LLMUnavailableError(
            f"No LLM client available (model={resolved.openai_model}, "
            f"base_url={resolved.openai_api_url or 'default'}). Set OPENAI_API_KEY."
        )
    return client
