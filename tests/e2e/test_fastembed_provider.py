"""
Real (non-mock) embedding provider check.

Downloads the default HuggingFace model on first run, so it is marked `network`
and can be deselected offline with: pytest -m "not network"
"""

import math

import pytest

from src.config import Settings
from src.ingestion.metadata_extractor import EmbeddingService


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


@pytest.mark.network
class TestFastEmbedProvider:
    """The default provider must produce real, semantically meaningful vectors."""

    def test_vectors_match_configured_dimension(self):
        settings = Settings(embedding_provider="fastembed")
        service = EmbeddingService(settings=settings)

        vec = service.embed_text("quarterly revenue by region")
        assert service.provider == "fastembed"
        assert len(vec) == settings.embedding_dim

    def test_related_texts_are_closer_than_unrelated_ones(self):
        service = EmbeddingService(settings=Settings(embedding_provider="fastembed"))

        related_a, related_b, unrelated = service.embed_texts(
            [
                "total sales revenue per store",
                "how much money did each shop earn",
                "the on-call engineer must publish a post-mortem",
            ]
        )
        assert _cosine(related_a, related_b) > _cosine(related_a, unrelated)

    def test_unloadable_model_raises_instead_of_silently_mocking(self):
        settings = Settings(embedding_provider="fastembed", embedding_model="nonexistent/model")
        with pytest.raises(RuntimeError, match="FastEmbed"):
            EmbeddingService(settings=settings)


if __name__ == "__main__":
    svc = EmbeddingService(settings=Settings(embedding_provider="fastembed"))
    v = svc.embed_text("quarterly revenue by region")
    print(f"provider={svc.provider} model={svc.model} dim={len(v)}")
