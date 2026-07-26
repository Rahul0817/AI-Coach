"""Provider-agnostic LLM interface.

The brief calls for OpenAI *with the ability to change providers later*. That is
achieved by making every caller depend on :class:`LLMProvider` — an abstract
base with two methods — and never on a vendor SDK. Swapping to Anthropic,
Gemini or a self-hosted model means writing one new subclass; no agent, service
or route changes.

The dataclasses here are the wire format between the agent layer and whichever
provider is configured. They are intentionally minimal: role, content, and the
metadata needed for cost accounting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class LLMMessage:
    """One turn passed to the model."""

    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class LLMResponse:
    """A completed generation plus the metadata we persist on the message row."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    provider: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class GenerationConfig:
    """Per-call generation parameters.

    Kept separate from provider construction so a single provider instance can
    serve agents with different temperature needs — the Mental Wellness Coach
    wants warmth, the Blood Report Analyzer wants determinism.
    """

    temperature: float = 0.4
    max_tokens: int = 1200
    top_p: float = 1.0
    stop: list[str] | None = None


class LLMProvider(ABC):
    """The contract every language-model backend must satisfy."""

    #: Short identifier recorded on each message for observability.
    name: str = "base"

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        """Generate a full response."""

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Yield the response incrementally as text deltas."""

    @abstractmethod
    async def health(self) -> bool:
        """Report whether the backend is reachable and configured."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Cheap token estimate used when a provider reports no usage.

        The ~4-characters-per-token heuristic is accurate to within roughly
        10–15% for English prose, which is entirely adequate for budget
        accounting and context-window guardrails.
        """
        return max(1, len(text) // 4)


class EmbeddingProvider(ABC):
    """The contract for turning text into vectors for the RAG index."""

    name: str = "base"
    dimensions: int = 0

    #: Cosine similarity below which a hit is considered noise. This is
    #: **provider-specific and must be measured, not guessed** — a semantic
    #: encoder puts genuine matches around 0.3–0.6, while a lexical hashing
    #: embedder puts the same matches near 0.08 because a short query vector
    #: shares few active dimensions with a long document vector. Hard-coding a
    #: single global threshold silently returns nothing for one of them.
    similarity_floor: float = 0.15

    #: Weight this provider's ranking receives in hybrid fusion, relative to
    #: BM25's weight of 1.0. Providers with weaker retrieval quality declare a
    #: lower weight so the lexical leg dominates rather than being diluted.
    fusion_weight: float = 1.0

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        Separate from :meth:`embed` because some providers use asymmetric
        query/document encoders, and because query embedding is latency
        sensitive while document embedding is a batch job.
        """
