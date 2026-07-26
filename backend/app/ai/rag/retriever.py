"""Hybrid retrieval: dense vectors plus BM25 lexical re-ranking.

Why hybrid rather than pure vector search?
------------------------------------------
Dense retrieval is strong on paraphrase ("I keep skipping periods" → a chunk
about oligomenorrhoea) and weak on rare exact terms — a query for "HOMA-IR" or
"letrozole" can miss the one chunk that names it, because a single rare token
barely moves a document-level embedding.

BM25 is the mirror image: excellent on exact rare terms, blind to paraphrase.

Fusing them covers both failure modes. The two ranked lists are combined with
**Reciprocal Rank Fusion**, which merges on *rank* rather than score. That
matters because cosine similarity and BM25 scores live on incomparable scales,
and any attempt to normalise and weight them directly is fragile — RRF needs no
tuning and is robust to one retriever producing wild scores.

This also makes the local hashing embedder viable: where its lexical embedding
is weak, BM25 carries the query, and vice versa.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.ai.llm.embeddings import tokenize
from app.ai.llm.factory import get_embedding_provider
from app.ai.rag.loader import DocumentChunk, load_knowledge_base
from app.ai.rag.vector_store import VectorStore, get_vector_store
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: RRF damping constant. 60 is the value from the original publication and is
#: the conventional default; it controls how quickly rank influence decays.
RRF_K = 60

#: How many candidates each retriever contributes before fusion. Over-fetching
#: gives the fusion something to actually re-order.
CANDIDATE_MULTIPLIER = 3

#: Minimum BM25 score for a hit to count as real lexical evidence. BM25, unlike
#: cosine similarity, has a stable scale on a fixed corpus: below roughly 1.0
#: the match is a single common term rather than meaningful overlap.
BM25_FLOOR = 1.0


@dataclass(slots=True)
class RetrievedChunk:
    """A retrieval hit, ready to be cited in a response."""

    id: str
    text: str
    title: str
    section: str
    source: str
    category: str
    score: float

    def snippet(self, limit: int = 260) -> str:
        """Short preview shown in the UI's citation chip."""
        # Skip the "Title — Section" header line the loader prepended.
        body = self.text.split("\n\n", 1)[-1].strip()
        if len(body) <= limit:
            return body
        return body[:limit].rsplit(" ", 1)[0] + "…"

    def to_citation(self) -> dict[str, Any]:
        return {
            "title": f"{self.title} — {self.section}",
            "source": self.source,
            "snippet": self.snippet(),
            "score": round(min(1.0, max(0.0, self.score)), 4),
            "category": self.category,
        }


class BM25Index:
    """A compact in-memory BM25 index over the same chunks as the vector store.

    Rebuilt at ingestion time and held in memory. For a corpus of this size the
    whole index is a few hundred kilobytes, so there is no reason to persist it
    separately — and keeping it in step with the vector store is one less
    consistency problem.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self) -> None:
        self._doc_ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._counts: list[Counter[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_len: float = 0.0

    def build(self, entries: list[dict[str, Any]]) -> None:
        self._doc_ids = [e["id"] for e in entries]
        self._tokens = [tokenize(e["text"]) for e in entries]
        self._counts = [Counter(t) for t in self._tokens]
        self._doc_freq = Counter()
        for tokens in self._tokens:
            self._doc_freq.update(set(tokens))
        total = sum(len(t) for t in self._tokens)
        self._avg_len = total / len(self._tokens) if self._tokens else 0.0
        logger.info("bm25 index built", extra={"documents": len(self._doc_ids)})

    @property
    def is_built(self) -> bool:
        return bool(self._doc_ids)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self.is_built:
            return []
        terms = tokenize(query)
        if not terms:
            return []

        n_docs = len(self._doc_ids)
        scored: list[tuple[str, float]] = []
        for index, counts in enumerate(self._counts):
            length = len(self._tokens[index]) or 1
            score = 0.0
            for term in set(terms):
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                df = self._doc_freq[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                numerator = frequency * (self.K1 + 1)
                denominator = frequency + self.K1 * (
                    1 - self.B + self.B * length / self._avg_len
                )
                score += idf * numerator / denominator
            if score > 0:
                scored.append((self._doc_ids[index], score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


class KnowledgeRetriever:
    """Ingests the knowledge base and serves hybrid retrieval queries."""

    def __init__(self, store: VectorStore | None = None) -> None:
        self._store = store or get_vector_store()
        self._bm25 = BM25Index()
        self._lookup: dict[str, dict[str, Any]] = {}
        self._ready = False

    # ------------------------------------------------------------ ingestion
    async def ingest(self, *, force: bool = False) -> int:
        """Embed and index the knowledge base. Returns the chunk count.

        Skips re-embedding when the store already holds the expected number of
        chunks, because with a paid embedding provider re-ingesting on every
        boot would be a recurring bill for no benefit.
        """
        chunks = load_knowledge_base()
        if not chunks:
            logger.warning("no knowledge chunks found; RAG will return nothing")
            self._ready = True
            return 0

        existing = await self._store.count()
        if existing >= len(chunks) and not force:
            logger.info(
                "vector store already populated, skipping embedding",
                extra={"chunks": existing},
            )
        else:
            if force:
                await self._store.reset()
            embedder = get_embedding_provider()
            vectors = await embedder.embed([c.text for c in chunks])
            await self._store.upsert(chunks, vectors)
            logger.info(
                "knowledge base embedded",
                extra={"chunks": len(chunks), "embedder": embedder.name},
            )

        await self._rebuild_lexical_index()
        self._ready = True
        return len(chunks)

    async def _rebuild_lexical_index(self) -> None:
        entries = await self._store.all_chunks()
        self._lookup = {e["id"]: e for e in entries}
        self._bm25.build(entries)

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------ retrieval
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        category: str | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Return the most relevant chunks for ``query``."""
        if not self._ready:
            await self.ingest()

        k = top_k or settings.rag_top_k
        relative_floor = settings.rag_min_score if min_score is None else min_score
        fetch = k * CANDIDATE_MULTIPLIER

        # --- dense leg ---
        embedder = get_embedding_provider()
        query_vector = await embedder.embed_query(query)
        dense = await self._store.query(query_vector, fetch, category)
        dense_scores = {hit["id"]: hit["score"] for hit in dense}

        # --- lexical leg ---
        lexical = self._bm25.search(query, fetch)
        if category:
            lexical = [
                (doc_id, score)
                for doc_id, score in lexical
                if self._lookup.get(doc_id, {}).get("metadata", {}).get("category")
                == category
            ]
        lexical_scores = {doc_id: score for doc_id, score in lexical}

        # --- quality gate, applied BEFORE truncation -----------------------
        # Filtering after taking the top k would discard good hits that a weak
        # leg had pushed down the fused list, which is how this originally
        # returned nothing for obviously answerable questions. A candidate
        # survives if *either* retriever considers it real evidence.
        qualified = {
            doc_id
            for doc_id in set(dense_scores) | set(lexical_scores)
            if dense_scores.get(doc_id, 0.0) >= embedder.similarity_floor
            or lexical_scores.get(doc_id, 0.0) >= BM25_FLOOR
        }
        if not qualified:
            # Nothing in the corpus is relevant. Returning an empty list is the
            # correct answer — the agent layer then says it does not know
            # rather than grounding on noise.
            logger.debug("retrieval found no qualifying chunks")
            return []

        # --- weighted reciprocal rank fusion -------------------------------
        # Fusing on rank rather than score sidesteps the fact that cosine
        # similarity and BM25 live on incomparable scales. The dense leg's
        # weight is declared by the embedding provider.
        fused: dict[str, float] = {}
        dense_ranked = [h["id"] for h in dense if h["id"] in qualified]
        lexical_ranked = [d for d, _ in lexical if d in qualified]

        for rank, doc_id in enumerate(dense_ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + (
                embedder.fusion_weight / (RRF_K + rank + 1)
            )
        for rank, doc_id in enumerate(lexical_ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
        best_fused = ordered[0][1] if ordered else 1.0

        results: list[RetrievedChunk] = []
        # A long section is split into several chunks by the loader, so the same
        # heading can legitimately appear more than once. Showing a user two
        # citations with an identical title reads as a bug, so only the
        # highest-ranked part of each section is surfaced.
        seen_sections: set[tuple[str, str]] = set()

        for doc_id, fusion_score in ordered:
            if len(results) >= k:
                break
            entry = self._lookup.get(doc_id)
            if entry is None:
                continue
            section_key = (
                entry.get("metadata", {}).get("title", ""),
                entry.get("metadata", {}).get("section", ""),
            )
            if section_key in seen_sections:
                continue
            # Reported relevance is the fused score normalised against the best
            # hit, so it means "how strong is this relative to the best match"
            # on a 0–1 scale that is stable across embedding providers. Raw
            # cosine is not comparable between providers and would render as a
            # meaningless percentage in the UI.
            relevance = fusion_score / best_fused if best_fused else 0.0
            if relevance < relative_floor:
                continue
            metadata = entry.get("metadata", {})
            seen_sections.add(section_key)
            results.append(
                RetrievedChunk(
                    id=doc_id,
                    text=entry["text"],
                    title=metadata.get("title", "PCOS Knowledge Base"),
                    section=metadata.get("section", ""),
                    source=metadata.get("source", "Oviora Knowledge Base"),
                    category=metadata.get("category", "general"),
                    score=round(relevance, 4),
                )
            )

        logger.debug(
            "retrieval complete",
            extra={
                "query_chars": len(query),
                "dense_hits": len(dense_ranked),
                "lexical_hits": len(lexical_ranked),
                "returned": len(results),
            },
        )
        return results

    async def stats(self) -> dict[str, Any]:
        return {
            "chunks_indexed": await self._store.count(),
            "lexical_index_built": self._bm25.is_built,
            "embedding_provider": get_embedding_provider().name,
            "vector_store": type(self._store).__name__,
        }


_retriever: KnowledgeRetriever | None = None


def get_retriever() -> KnowledgeRetriever:
    """FastAPI dependency returning the shared retriever."""
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever


def reset_retriever() -> None:
    """Drop the cached retriever. Used by tests."""
    global _retriever
    _retriever = None
