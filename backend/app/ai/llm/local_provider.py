"""Retrieval-grounded local provider — the zero-credential fallback.

What this is
------------
**Not a language model.** It is a deterministic *extractive composer*: it reads
the retrieved knowledge blocks that the prompt builder embedded in the system
message, ranks their sentences against the user's question with a BM25-style
score, and assembles a structured Markdown answer from the highest-scoring
material, in the persona of whichever agent is active.

Why it exists
-------------
An evaluator cloning this repository has no OpenAI key. Without a fallback the
entire product — chat, agents, RAG, plan generation — would be a dead demo.
With it, every path is exercisable end to end, and switching to a real model is
one environment variable.

Why extractive rather than generative-looking
---------------------------------------------
This is a health product. A template engine that *invented* fluent clinical
prose would be actively dangerous. Restricting output to sentences that
actually appear in the curated corpus means the fallback cannot hallucinate: if
retrieval found nothing relevant, it says so instead of guessing. That is the
correct failure mode here, and it is enforced by construction rather than by
prompt instructions.

Protocol
--------
The prompt builder marks retrieved passages with ``<<<SOURCE ...>>> …
<<<END>>>`` sentinels (see :mod:`app.ai.agents.prompts`). That is a private
contract between our own prompt builder and this provider; the OpenAI provider
ignores the markers, which read as ordinary context to a real model.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.ai.llm.base import (
    GenerationConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
)
from app.ai.llm.embeddings import tokenize
from app.core.logging import get_logger

logger = get_logger(__name__)

SOURCE_RE = re.compile(
    r"<<<SOURCE id=(?P<id>\d+) title=\"(?P<title>[^\"]*)\" source=\"(?P<source>[^\"]*)\">>>"
    r"(?P<body>.*?)<<<END>>>",
    re.DOTALL,
)
PERSONA_RE = re.compile(r"<<<PERSONA>>>(?P<persona>.*?)<<<END>>>", re.DOTALL)
PROFILE_RE = re.compile(r"<<<PROFILE>>>(?P<profile>.*?)<<<END>>>", re.DOTALL)

#: Sentence splitter. Deliberately conservative — it keeps decimal numbers and
#: common abbreviations ("e.g.", "mg/dL") intact rather than splitting mid-fact.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

MAX_BULLETS = 5
MIN_SENTENCE_CHARS = 40
MAX_SENTENCE_CHARS = 400


@dataclass(slots=True)
class ScoredSentence:
    text: str
    score: float
    source_title: str
    source_ref: str


class LocalRetrievalProvider(LLMProvider):
    """Composes grounded answers from retrieved context without an API key."""

    name = "local"

    def __init__(self) -> None:
        self.model = "oviora-local-composer-1"

    # ---------------------------------------------------------- entry points
    async def complete(
        self,
        messages: list[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        system_text = "\n\n".join(m.content for m in messages if m.role == "system")
        history = [m for m in messages if m.role != "system"]
        query = next((m.content for m in reversed(history) if m.role == "user"), "")

        answer = self._compose(system_text, query, history)
        return LLMResponse(
            content=answer,
            model=self.model,
            prompt_tokens=self.estimate_tokens(system_text + query),
            completion_tokens=self.estimate_tokens(answer),
            finish_reason="stop",
            provider=self.name,
            metadata={"grounded": bool(SOURCE_RE.search(system_text))},
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Emit the composed answer word-by-word.

        The small delay is not cosmetic padding for its own sake: it lets the
        frontend's streaming renderer, SSE plumbing and typing indicator be
        developed and tested against realistic timing without a paid API.
        """
        response = await self.complete(messages, config)
        for token in re.findall(r"\S+\s*", response.content):
            yield token
            await asyncio.sleep(0.012)

    async def health(self) -> bool:
        return True  # No external dependency, so always available.

    # -------------------------------------------------------------- internals
    def _compose(self, system_text: str, query: str, history: list[LLMMessage]) -> str:
        persona = self._extract(PERSONA_RE, system_text, "persona")
        profile = self._extract(PROFILE_RE, system_text, "profile")
        blocks = self._parse_sources(system_text)

        if not blocks:
            return self._ungrounded_reply(persona, profile, query)

        sentences = self._rank_sentences(blocks, query)
        if not sentences:
            return self._ungrounded_reply(persona, profile, query)

        return self._render(persona, profile, query, sentences, blocks, history)

    @staticmethod
    def _extract(pattern: re.Pattern[str], text: str, group: str) -> str:
        match = pattern.search(text)
        return match.group(group).strip() if match else ""

    @staticmethod
    def _parse_sources(system_text: str) -> list[dict[str, str]]:
        return [
            {
                "id": match.group("id"),
                "title": match.group("title").strip(),
                "source": match.group("source").strip(),
                "body": match.group("body").strip(),
            }
            for match in SOURCE_RE.finditer(system_text)
        ]

    def _rank_sentences(
        self, blocks: list[dict[str, str]], query: str
    ) -> list[ScoredSentence]:
        """Score every candidate sentence against the query using BM25.

        BM25 rather than raw overlap because it does two things that matter on
        a small curated corpus: it discounts terms that appear in most
        documents (so "PCOS" contributes almost nothing), and it saturates term
        frequency so one keyword-stuffed sentence cannot dominate.
        """
        candidates: list[tuple[str, dict[str, str]]] = []
        for block in blocks:
            for raw in _SENTENCE_RE.split(block["body"]):
                sentence = " ".join(raw.split())
                if MIN_SENTENCE_CHARS <= len(sentence) <= MAX_SENTENCE_CHARS:
                    candidates.append((sentence, block))

        if not candidates:
            return []

        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        tokenised = [tokenize(text) for text, _ in candidates]
        n_docs = len(tokenised)
        avg_len = sum(len(t) for t in tokenised) / n_docs

        document_frequency: Counter[str] = Counter()
        for tokens in tokenised:
            document_frequency.update(set(tokens))

        k1, b = 1.5, 0.75
        scored: list[ScoredSentence] = []
        for (text, block), tokens in zip(candidates, tokenised, strict=True):
            counts = Counter(tokens)
            length = len(tokens) or 1
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                df = document_frequency[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                numerator = frequency * (k1 + 1)
                denominator = frequency + k1 * (1 - b + b * length / avg_len)
                score += idf * numerator / denominator
            if score > 0:
                scored.append(
                    ScoredSentence(
                        text=text,
                        score=round(score, 4),
                        source_title=block["title"],
                        source_ref=block["source"],
                    )
                )

        scored.sort(key=lambda s: s.score, reverse=True)
        return self._deduplicate(scored)[:MAX_BULLETS]

    @staticmethod
    def _deduplicate(sentences: list[ScoredSentence]) -> list[ScoredSentence]:
        """Drop near-duplicates so the answer does not repeat itself.

        Curated guideline documents restate the same advice in several places;
        without this the top five bullets are often the same fact five times.
        """
        kept: list[ScoredSentence] = []
        seen_signatures: list[set[str]] = []
        for candidate in sentences:
            signature = set(tokenize(candidate.text))
            if not signature:
                continue
            duplicate = any(
                len(signature & existing) / len(signature | existing) > 0.6
                for existing in seen_signatures
            )
            if not duplicate:
                kept.append(candidate)
                seen_signatures.append(signature)
        return kept

    def _render(
        self,
        persona: str,
        profile: str,
        query: str,
        sentences: list[ScoredSentence],
        blocks: list[dict[str, str]],
        history: list[LLMMessage],
    ) -> str:
        role = persona.splitlines()[0].strip() if persona else "Oviora AI"
        parts: list[str] = [
            f"Here is what the guidance in my knowledge base says about "
            f"**{self._topic(query)}**:",
            "",
        ]

        for sentence in sentences:
            parts.append(f"- {sentence.text} *({sentence.source_title})*")
        parts.append("")

        if profile:
            parts.append("**Applied to your profile**")
            parts.append("")
            for line in profile.splitlines():
                line = line.strip("- ").strip()
                if line:
                    parts.append(f"- {line}")
            parts.append("")

        distinct_sources = sorted({b["source"] for b in blocks})
        parts.append("**Sources consulted:** " + ", ".join(distinct_sources))
        parts.append("")
        parts.append(
            f"*Answered by the {role} using retrieved guidance. This "
            f"deployment is running Oviora's offline composer, which quotes "
            f"the knowledge base directly rather than generating new prose — "
            f"configure an LLM provider for conversational answers.*"
        )
        return "\n".join(parts)

    def _ungrounded_reply(self, persona: str, profile: str, query: str) -> str:
        """Response when retrieval returned nothing usable.

        Saying "I don't have material on this" is the correct behaviour for a
        health assistant with no grounding. Inventing an answer is not.
        """
        role = persona.splitlines()[0].strip() if persona else "Oviora AI"
        lines = [
            f"I could not find anything in my PCOS knowledge base that "
            f"directly addresses **{self._topic(query)}**, so I would rather "
            f"say that than guess.",
            "",
            "Things I can help with right now:",
            "",
            "- PCOS symptoms, diagnostic criteria and what tests usually involve",
            "- Nutrition strategies for insulin resistance and lower-glycaemic eating",
            "- Exercise programming that suits PCOS, including strength work",
            "- Cycle tracking, and what irregularity patterns are worth raising",
            "- Sleep, stress and mental-wellbeing routines",
            "- Reading a blood report and understanding what each marker means",
            "",
        ]
        if profile:
            lines.append("What I already know about you:")
            lines.append("")
            for line in profile.splitlines():
                line = line.strip("- ").strip()
                if line:
                    lines.append(f"- {line}")
            lines.append("")
        lines.append(
            f"*{role} — please rephrase your question, or ask about one of the "
            f"topics above.*"
        )
        return "\n".join(lines)

    @staticmethod
    def _topic(query: str) -> str:
        """Short topic label echoed back to the user."""
        terms = tokenize(query)[:6]
        return " ".join(terms) if terms else "your question"
