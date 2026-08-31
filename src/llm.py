"""
Shared OpenAI Client Provider.
Builds a single client from application settings (API key and optional
OpenAI-compatible base URL) reused across engines, router, and ingestion.
"""

from typing import Any, Optional

from src.config import Settings, get_settings

_client: Optional[Any] = None


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
        return None


def get_openai_client(settings: Optional[Settings] = None) -> Optional[Any]:
    """Returns a shared OpenAI client, or None when unavailable."""
    global _client
    if settings is not None:
        return _build_client(settings)
    if _client is None:
        _client = _build_client(get_settings())
    return _client
