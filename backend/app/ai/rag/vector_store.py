"""Vector storage over ChromaDB, with an in-process fallback.

ChromaDB is the configured store: it persists to disk, handles metadata
filtering, and needs no separate server process, which makes it a sensible
choice for a single-tenant application at this scale.

As with Redis, there is a fallback. ``InMemoryVectorStore`` implements the same
interface with brute-force cosine similarity. Over a corpus of a few hundred
chunks an exhaustive scan takes under a millisecond, so the fallback is not
merely a stub — it is genuinely adequate at this size, and it keeps the test
suite free of native-dependency flakiness. At corpus sizes where it would stop
being adequate, Chroma is available.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.ai.rag.loader import DocumentChunk
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class VectorStore(ABC):
    """Minimal vector-store contract used by the retriever."""

    @abstractmethod
    async def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None: ...

    @abstractmethod
    async def query(
        self, vector: list[float], top_k: int, category: str | None = None
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def count(self) -> int: ...

    @abstractmethod
    async def reset(self) -> None: ...

    @abstractmethod
    async def all_chunks(self) -> list[dict[str, Any]]: ...


class InMemoryVectorStore(VectorStore):
    """Brute-force cosine search over an in-process list."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._documents: list[str] = []
        self._metadatas: list[dict[str, str]] = []

    async def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors):
            if chunk.id in self._ids:
                index = self._ids.index(chunk.id)
                self._vectors[index] = vector
                self._documents[index] = chunk.text
                self._metadatas[index] = chunk.metadata()
            else:
                self._ids.append(chunk.id)
                self._vectors.append(vector)
                self._documents.append(chunk.text)
                self._metadatas.append(chunk.metadata())

    async def query(
        self, vector: list[float], top_k: int, category: str | None = None
    ) -> list[dict[str, Any]]:
        norm_q = math.sqrt(sum(v * v for v in vector)) or 1.0
        scored: list[tuple[float, int]] = []

        for index, candidate in enumerate(self._vectors):
            if category and self._metadatas[index].get("category") != category:
                continue
            norm_c = math.sqrt(sum(v * v for v in candidate)) or 1.0
            dot = sum(a * b for a, b in zip(vector, candidate))
            scored.append((dot / (norm_q * norm_c), index))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "id": self._ids[index],
                "text": self._documents[index],
                "metadata": self._metadatas[index],
                "score": round(float(score), 4),
            }
            for score, index in scored[:top_k]
        ]

    async def count(self) -> int:
        return len(self._ids)

    async def reset(self) -> None:
        self._ids.clear()
        self._vectors.clear()
        self._documents.clear()
        self._metadatas.clear()

    async def all_chunks(self) -> list[dict[str, Any]]:
        return [
            {"id": i, "text": t, "metadata": m}
            for i, t, m in zip(self._ids, self._documents, self._metadatas)
        ]


class ChromaVectorStore(VectorStore):
    """Persistent vector storage backed by ChromaDB."""

    def __init__(self, client: Any, collection: Any) -> None:
        self._client = client
        self._collection = collection

    async def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
        )

    async def query(
        self, vector: list[float], top_k: int, category: str | None = None
    ) -> list[dict[str, Any]]:
        where = {"category": category} if category else None
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        out: list[dict[str, Any]] = []
        for identifier, text, metadata, distance in zip(ids, documents, metadatas, distances):
            # Chroma's default space is cosine *distance* (1 - similarity).
            # Converting here keeps every store returning "higher is better".
            similarity = 1.0 - float(distance)
            out.append(
                {
                    "id": identifier,
                    "text": text,
                    "metadata": dict(metadata or {}),
                    "score": round(similarity, 4),
                }
            )
        return out

    async def count(self) -> int:
        return int(self._collection.count())

    async def reset(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    async def all_chunks(self) -> list[dict[str, Any]]:
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            {"id": i, "text": t, "metadata": dict(m or {})}
            for i, t, m in zip(
                result.get("ids", []),
                result.get("documents", []),
                result.get("metadatas", []),
            )
        ]


_store: VectorStore | None = None


def build_vector_store() -> VectorStore:
    """Construct the configured store, degrading to in-memory on failure."""
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            # Cosine rather than the L2 default: our embeddings are
            # L2-normalised, which makes cosine the meaningful metric.
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "chromadb ready",
            extra={"path": str(persist_dir), "collection": settings.chroma_collection},
        )
        return ChromaVectorStore(client, collection)
    except Exception as exc:
        logger.warning(
            "chromadb unavailable, using in-memory vector store",
            extra={"error": str(exc)},
        )
        return InMemoryVectorStore()


def get_vector_store() -> VectorStore:
    """Process-wide vector store singleton."""
    global _store
    if _store is None:
        _store = build_vector_store()
    return _store


def reset_vector_store() -> None:
    """Drop the cached store. Used by tests."""
    global _store
    _store = None
