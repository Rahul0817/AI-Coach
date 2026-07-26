"""Embedding backends for the RAG index.

Two implementations:

``OpenAIEmbedder``
    Imported from :mod:`app.ai.llm.openai_provider`. Semantic, best quality,
    needs an API key and costs money per document.

``LocalHashEmbedder``
    A dependency-free **lexical** embedder: signed feature hashing over word
    unigrams and bigrams with sub-linear term weighting, L2-normalised so that
    cosine similarity is a meaningful overlap score.

Be precise about what the local option is and is not. It captures *lexical*
similarity, not meaning — "irregular periods" and "oligomenorrhoea" are near
neighbours to a real embedding model and strangers to this one. It exists so
that a reviewer can clone the repo and get working retrieval in seconds, and
because the retriever pairs it with a BM25 re-ranking stage that compensates
for much of the gap on a curated corpus. Set ``EMBEDDING_PROVIDER=openai`` for
production-grade semantic recall.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from app.ai.llm.base import EmbeddingProvider

#: Vector width. 768 keeps hash collisions rare for a corpus of this size while
#: staying small enough that similarity search is effectively free.
LOCAL_DIMENSIONS = 768

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Function words carry no retrieval signal and would dominate the hash space.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then than that this these those of to in on at by
    for with about into over after is are was were be been being do does did
    have has had having i you he she it we they me him her them my your his
    its our their as from not no so such can could should would may might will
    just very more most some any each other there here what which who whom how
    when where why all both few own same too s t don now
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords and single characters removed."""
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def _hash_bucket(term: str, dimensions: int) -> tuple[int, float]:
    """Map a term to a bucket and a sign.

    The sign comes from an independent bit of the same digest. Signed hashing
    makes collisions cancel in expectation instead of always inflating a
    bucket, which measurably improves similarity estimates at this width.
    """
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    bucket = value % dimensions
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return bucket, sign


class LocalHashEmbedder(EmbeddingProvider):
    """Deterministic lexical embeddings, no network and no model download."""

    name = "local"
    dimensions = LOCAL_DIMENSIONS

    #: Measured on this corpus: relevant chunks score 0.05–0.15, irrelevant
    #: ones sit below 0.04. The values are low in absolute terms because a
    #: five-word query activates a handful of the 768 buckets that a
    #: 1,400-character chunk spreads energy across.
    similarity_floor = 0.04

    #: Down-weighted in fusion. Side-by-side on this corpus, BM25 ranked the
    #: correct chunk first on queries where this embedder ranked it fourth or
    #: missed it, so the lexical leg is given roughly twice the influence.
    fusion_weight = 0.5

    def __init__(self, dimensions: int = LOCAL_DIMENSIONS) -> None:
        self.dimensions = dimensions

    def _vectorise(self, text: str) -> list[float]:
        tokens = tokenize(text)
        if not tokens:
            return [0.0] * self.dimensions

        # Bigrams capture short phrases ("insulin resistance") that unigrams
        # alone would scatter across unrelated documents.
        bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        counts = Counter(tokens)
        counts.update(bigrams)

        vector = [0.0] * self.dimensions
        for term, count in counts.items():
            bucket, sign = _hash_bucket(term, self.dimensions)
            # Sub-linear (log) term frequency: the tenth mention of a word is
            # far less informative than the first.
            vector[bucket] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorise(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vectorise(text)

    # Chroma calls embedding functions synchronously, so expose a sync path too.
    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return [self._vectorise(text) for text in input]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-length vectors, clamped to [-1, 1]."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimensionality.")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))
