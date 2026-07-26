"""The Router Agent: deciding which specialist answers a query.

Design decision: rules first, LLM second
----------------------------------------
The obvious implementation is to ask the LLM "which agent should handle this?"
That costs an extra round trip on **every** message — roughly doubling latency
and cost — to answer a question that is usually trivial. "What should I eat for
breakfast?" does not need a language model to classify.

So routing runs in two tiers:

1. **A weighted lexical classifier** over each agent's keywords and phrases,
   with context boosts. It is deterministic, testable, sub-millisecond, and
   confidently resolves the large majority of real queries.
2. **An LLM fallback**, used only when tier one is ambiguous — no agent clears
   the confidence bar, or the top two are within a whisker of each other.

This is the same "cheap path for the common case, expensive path for the hard
case" pattern used throughout the system, and it keeps routing off the critical
cost path without giving up accuracy on genuinely ambiguous input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ai.agents.prompts import AGENT_DEFINITIONS
from app.ai.llm.base import GenerationConfig, LLMMessage, LLMProvider
from app.core.logging import get_logger
from app.models.enums import AgentName

logger = get_logger(__name__)

#: Points per matched signal. A multi-word phrase is far more discriminative
#: than a single keyword, so it is worth substantially more.
KEYWORD_WEIGHT = 1.0
PHRASE_WEIGHT = 3.5
#: Boost for staying with the agent that answered the previous turn. Without it
#: a follow-up like "and what about dinner?" bounces to a different specialist
#: mid-conversation, which feels broken to the user.
CONTINUITY_BOOST = 1.6
#: Applied when the user explicitly attached a file of a given kind.
ATTACHMENT_BOOST = 8.0

#: Below this normalised confidence the LLM tie-breaker is consulted.
LLM_FALLBACK_THRESHOLD = 0.34
#: If the top two agents are this close, the lexical signal is not decisive.
AMBIGUITY_MARGIN = 0.10

_WORD_RE = re.compile(r"[a-z']+")


@dataclass(slots=True)
class RoutingResult:
    agent: AgentName
    confidence: float
    reason: str
    matched_signals: list[str]
    used_llm: bool = False


class AgentRouter:
    """Selects the specialist best suited to a query."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    async def route(
        self,
        query: str,
        *,
        previous_agent: AgentName | None = None,
        has_image: bool = False,
        has_report: bool = False,
        forced: AgentName | None = None,
    ) -> RoutingResult:
        """Return the chosen agent with its confidence and rationale."""
        if forced:
            return RoutingResult(
                agent=forced,
                confidence=1.0,
                reason="You selected this specialist directly.",
                matched_signals=["user_override"],
            )

        # An attachment is a near-certain signal and short-circuits scoring.
        if has_image:
            return RoutingResult(
                agent=AgentName.FOOD_ANALYZER,
                confidence=1.0,
                reason="A food photo was attached.",
                matched_signals=["image_attachment"],
            )
        if has_report:
            return RoutingResult(
                agent=AgentName.BLOOD_REPORT_ANALYZER,
                confidence=1.0,
                reason="A lab report was attached.",
                matched_signals=["report_attachment"],
            )

        scores, signals = self._score(query, previous_agent, has_image, has_report)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        total = sum(max(0.0, s) for s in scores.values())
        best_agent, best_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

        confidence = round(best_score / total, 3) if total > 0 else 0.0
        margin = (best_score - runner_up_score) / total if total > 0 else 0.0

        ambiguous = (
            best_score <= 0
            or confidence < LLM_FALLBACK_THRESHOLD
            or margin < AMBIGUITY_MARGIN
        )

        if ambiguous and self._llm is not None:
            llm_choice = await self._classify_with_llm(query)
            if llm_choice is not None:
                return RoutingResult(
                    agent=llm_choice,
                    confidence=0.7,
                    reason=(
                        "The wording did not clearly match one specialist, so "
                        "the router asked the language model to classify it."
                    ),
                    matched_signals=signals.get(llm_choice, []),
                    used_llm=True,
                )

        if best_score <= 0:
            # No signal at all — the general educator is the safe default.
            return RoutingResult(
                agent=AgentName.HEALTH_EXPERT,
                confidence=0.3,
                reason="No specialist signal was found; defaulting to general PCOS guidance.",
                matched_signals=[],
            )

        definition = AGENT_DEFINITIONS[best_agent]
        matched = signals.get(best_agent, [])
        return RoutingResult(
            agent=best_agent,
            confidence=max(0.35, confidence),
            reason=(
                f"Routed to the {definition.display_name} based on "
                + (", ".join(f"'{s}'" for s in matched[:3]) if matched else "topic match")
                + "."
            ),
            matched_signals=matched,
        )

    # -------------------------------------------------------------- scoring
    def _score(
        self,
        query: str,
        previous_agent: AgentName | None,
        has_image: bool,
        has_report: bool,
    ) -> tuple[dict[AgentName, float], dict[AgentName, list[str]]]:
        lowered = query.lower()
        # Matching on a token set rather than substrings avoids the classic
        # false positive where "period" matches inside "periodically".
        tokens = set(_WORD_RE.findall(lowered))

        scores: dict[AgentName, float] = {}
        signals: dict[AgentName, list[str]] = {}

        for name, definition in AGENT_DEFINITIONS.items():
            score = 0.0
            matched: list[str] = []

            for keyword in definition.routing_keywords:
                if keyword in tokens:
                    score += KEYWORD_WEIGHT
                    matched.append(keyword)

            for phrase in definition.routing_phrases:
                if phrase in lowered:
                    score += PHRASE_WEIGHT
                    matched.append(phrase)

            if previous_agent == name and score > 0:
                # Only boost an agent that already has some signal — otherwise
                # continuity would pin the conversation to one specialist
                # regardless of what the user asks next.
                score *= CONTINUITY_BOOST
                matched.append("conversation continuity")

            if has_image and name == AgentName.FOOD_ANALYZER:
                score += ATTACHMENT_BOOST
            if has_report and name == AgentName.BLOOD_REPORT_ANALYZER:
                score += ATTACHMENT_BOOST

            scores[name] = score
            signals[name] = matched

        return scores, signals

    # ---------------------------------------------------------- llm tiebreak
    async def _classify_with_llm(self, query: str) -> AgentName | None:
        """Ask the model to pick an agent. Returns ``None`` if it cannot."""
        if self._llm is None:
            return None

        catalogue = "\n".join(
            f"- {definition.name.value}: {definition.description}"
            for definition in AGENT_DEFINITIONS.values()
        )
        prompt = (
            "You are a routing classifier for a PCOS health assistant. Choose "
            "the single most appropriate specialist for the user's message.\n\n"
            f"Specialists:\n{catalogue}\n\n"
            f'User message: "{query}"\n\n'
            "Reply with ONLY the specialist identifier, nothing else."
        )

        try:
            response = await self._llm.complete(
                [LLMMessage(role="user", content=prompt)],
                GenerationConfig(temperature=0.0, max_tokens=20),
            )
        except Exception as exc:
            logger.warning("llm routing fallback failed", extra={"error": str(exc)})
            return None

        answer = response.content.strip().lower()
        for name in AGENT_DEFINITIONS:
            if name.value in answer:
                return name
        return None
