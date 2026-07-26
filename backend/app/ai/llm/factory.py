"""Provider selection.

One function decides which LLM and embedding backends the whole application
uses, driven purely by configuration. Nothing else in the codebase imports a
concrete provider class, which is what makes "swap the provider" a config
change rather than a refactor.
"""

from __future__ import annotations

from functools import lru_cache

from app.ai.llm.base import EmbeddingProvider, LLMProvider
from app.ai.llm.embeddings import LocalHashEmbedder
from app.ai.llm.local_provider import LocalRetrievalProvider
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Return the configured LLM provider (cached for the process lifetime).

    Falls back to the local composer when ``openai`` is selected but no key is
    present. Degrading loudly beats refusing to boot: an operator who forgot to
    set the key gets a working app and a warning in the logs, not an outage.
    """
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            logger.warning(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is empty; "
                "falling back to the local composer"
            )
            return LocalRetrievalProvider()
        from app.ai.llm.openai_provider import OpenAIProvider

        logger.info("llm provider: openai", extra={"model": settings.llm_model})
        return OpenAIProvider()

    logger.info("llm provider: local composer (no API key required)")
    return LocalRetrievalProvider()


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured embedding provider (cached)."""
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            logger.warning(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is empty; "
                "falling back to local lexical embeddings"
            )
            return LocalHashEmbedder()
        from app.ai.llm.openai_provider import OpenAIEmbedder

        logger.info(
            "embedding provider: openai", extra={"model": settings.embedding_model}
        )
        return OpenAIEmbedder()

    logger.info("embedding provider: local hashing embedder")
    return LocalHashEmbedder()


def reset_providers() -> None:
    """Clear the caches. Used by tests that flip provider configuration."""
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()
