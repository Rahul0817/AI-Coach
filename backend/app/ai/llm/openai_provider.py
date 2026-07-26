"""OpenAI implementation of the LLM and embedding contracts.

Production concerns handled here rather than at the call site:

* **Retries with exponential backoff** on transient failures (429, 5xx) via
  ``tenacity``. Rate limits are normal at scale, not exceptional.
* **A hard timeout**, so a hung upstream cannot pin a request worker.
* **Error translation** into :class:`ExternalServiceError`, so the API returns a
  clean 502 instead of leaking a vendor stack trace to the browser.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.ai.llm.base import (
    EmbeddingProvider,
    GenerationConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
)
from app.core.config import settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)

REQUEST_TIMEOUT_SECONDS = 60.0
MAX_ATTEMPTS = 3


class OpenAIProvider(LLMProvider):
    """Chat completions backed by the OpenAI API."""

    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.llm_model
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily construct the SDK client.

        Deferred so that importing this module never requires the SDK or a key
        — the app boots fine with ``LLM_PROVIDER=local`` and openai uninstalled.
        """
        if self._client is None:
            if not settings.openai_api_key:
                raise ExternalServiceError(
                    "OPENAI_API_KEY is not configured. Set it, or switch to "
                    "LLM_PROVIDER=local."
                )
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=0,  # tenacity owns retries so backoff is uniform
            )
        return self._client

    def _retry(self):  # type: ignore[no-untyped-def]
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        return retry(
            stop=stop_after_attempt(MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        cfg = config or GenerationConfig(
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        client = self._get_client()

        @self._retry()
        async def _call() -> Any:
            return await client.chat.completions.create(
                model=self.model,
                messages=[m.to_dict() for m in messages],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                stop=cfg.stop,
            )

        try:
            response = await _call()
        except Exception as exc:
            logger.error("openai completion failed", extra={"error": str(exc)})
            raise ExternalServiceError(
                "The AI service is temporarily unavailable. Please try again."
            ) from exc

        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=choice.finish_reason or "stop",
            provider=self.name,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        cfg = config or GenerationConfig(
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        client = self._get_client()
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[m.to_dict() for m in messages],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:
            logger.error("openai stream failed", extra={"error": str(exc)})
            raise ExternalServiceError(
                "The AI stream was interrupted. Please try again."
            ) from exc

    async def health(self) -> bool:
        if not settings.openai_api_key:
            return False
        try:
            await self.complete(
                [LLMMessage(role="user", content="ping")],
                GenerationConfig(temperature=0, max_tokens=1),
            )
            return True
        except Exception:
            return False


class OpenAIEmbedder(EmbeddingProvider):
    """Document/query embeddings from the OpenAI embeddings endpoint."""

    name = "openai"
    #: ``text-embedding-3-small`` output width.
    dimensions = 1536

    #: Semantic encoders place genuinely related text well above this; unrelated
    #: text on ``text-embedding-3-small`` typically sits below 0.2.
    similarity_floor = 0.25

    #: Full weight in fusion — a semantic encoder is at least as reliable as
    #: BM25 and is the leg that handles paraphrase.
    fusion_weight = 1.0

    #: The API rejects oversized batches; chunking keeps ingestion reliable.
    BATCH_SIZE = 96

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.embedding_model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not settings.openai_api_key:
                raise ExternalServiceError(
                    "OPENAI_API_KEY is not configured for embeddings."
                )
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[start : start + self.BATCH_SIZE]
            try:
                response = await client.embeddings.create(model=self.model, input=batch)
            except Exception as exc:
                logger.error("openai embedding failed", extra={"error": str(exc)})
                raise ExternalServiceError("Embedding generation failed.") from exc
            vectors.extend(item.embedding for item in response.data)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
