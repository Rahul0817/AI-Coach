"""The multi-agent orchestrator: one entry point for every chat turn.

Pipeline for a single message::

    sanitise → safety screen → route → load memory → agent.answer()
             → learn facts → summarise if due → persist

Keeping the whole sequence in one place is deliberate. Spreading it across the
route handler, the service and the agents is how steps get skipped — and the
steps here include the safety screen and the disclaimer, which must never be
skipped.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.agents.base import AgentResponse, SpecialistAgent, build_agent
from app.ai.agents.router import AgentRouter, RoutingResult
from app.ai.agents.safety import (
    DISORDERED_EATING_NOTE,
    enforce_disclaimer,
    sanitise_user_input,
    screen_input,
)
from app.ai.llm.base import LLMProvider
from app.ai.llm.factory import get_llm_provider
from app.ai.memory.manager import ConversationMemory
from app.ai.rag.retriever import KnowledgeRetriever, RetrievedChunk, get_retriever
from app.core.logging import get_logger
from app.models.chat import Conversation
from app.models.enums import AgentName

logger = get_logger(__name__)


@dataclass(slots=True)
class OrchestrationResult:
    """Everything one turn produced, ready for persistence and serialisation."""

    response: AgentResponse
    routing: RoutingResult
    memory_updates: dict[str, Any] = field(default_factory=dict)
    safety_category: str | None = None


class AgentOrchestrator:
    """Coordinates routing, memory, retrieval and generation."""

    def __init__(
        self,
        session: AsyncSession,
        llm: LLMProvider | None = None,
        retriever: KnowledgeRetriever | None = None,
    ) -> None:
        self.session = session
        self.llm = llm or get_llm_provider()
        self.retriever = retriever or get_retriever()
        self.router = AgentRouter(llm=self.llm)
        self.memory = ConversationMemory(session, self.llm)

    def agent_for(self, name: AgentName) -> SpecialistAgent:
        return build_agent(name, self.llm, self.retriever)

    # ------------------------------------------------------------- handling
    async def handle(
        self,
        *,
        user_id: uuid.UUID,
        conversation: Conversation,
        message: str,
        forced_agent: AgentName | None = None,
        previous_agent: AgentName | None = None,
        has_image: bool = False,
        has_report: bool = False,
    ) -> OrchestrationResult:
        """Process one user turn end to end."""
        clean = sanitise_user_input(message.strip())

        # ---- 1. safety screen, before anything expensive happens ----------
        verdict = screen_input(clean)
        if verdict.blocked:
            # A blocked message still gets routed for record-keeping, but the
            # model is never called and no facts are learned from it.
            return OrchestrationResult(
                response=AgentResponse(
                    content=verdict.response or "",
                    agent=AgentName.MENTAL_WELLNESS_COACH,
                    used_rag=False,
                    model="safety-guardrail",
                ),
                routing=RoutingResult(
                    agent=AgentName.MENTAL_WELLNESS_COACH,
                    confidence=1.0,
                    reason=f"Safety guardrail triggered ({verdict.category}).",
                    matched_signals=[verdict.category or "safety"],
                ),
                safety_category=verdict.category,
            )

        prefix_note: str | None = None
        if verdict.category == "disordered_eating":
            prefix_note = DISORDERED_EATING_NOTE
            forced_agent = forced_agent or AgentName.MENTAL_WELLNESS_COACH

        # ---- 2. route -----------------------------------------------------
        routing = await self.router.route(
            clean,
            previous_agent=previous_agent,
            has_image=has_image,
            has_report=has_report,
            forced=forced_agent,
        )

        # ---- 3. memory: learn, then load ----------------------------------
        # Learning happens *before* context is built so a fact stated in this
        # very message ("I'm 22, what should I eat?") personalises this answer
        # rather than only the next one.
        memory_updates = await self.memory.learn_from_message(user_id, clean)
        profile_context = await self.memory.build_profile_context(user_id)
        history = await self.memory.working_window(conversation.id)
        summary = conversation.summary

        # ---- 4. generate --------------------------------------------------
        agent = self.agent_for(routing.agent)
        response = await agent.answer(
            clean,
            history=history,
            profile_context=profile_context,
            conversation_summary=summary,
            prefix_note=prefix_note,
        )

        logger.info(
            "chat turn handled",
            extra={
                "agent": routing.agent.value,
                "routing_confidence": routing.confidence,
                "used_llm_router": routing.used_llm,
                "sources": len(response.sources),
                "tokens": response.tokens_used,
            },
        )

        return OrchestrationResult(
            response=response,
            routing=routing,
            memory_updates=memory_updates,
            safety_category=verdict.category,
        )

    # ------------------------------------------------------------ streaming
    async def stream(
        self,
        *,
        user_id: uuid.UUID,
        conversation: Conversation,
        message: str,
        forced_agent: AgentName | None = None,
        previous_agent: AgentName | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE-shaped frames for a streaming completion.

        Frame types: ``meta`` (routing decision), ``sources`` (citations),
        ``token`` (text delta), ``done`` (final accounting), ``error``.
        """
        clean = sanitise_user_input(message.strip())

        verdict = screen_input(clean)
        if verdict.blocked:
            yield {
                "type": "meta",
                "data": {
                    "agent": AgentName.MENTAL_WELLNESS_COACH.value,
                    "confidence": 1.0,
                    "reason": f"Safety guardrail triggered ({verdict.category}).",
                    "safety": verdict.category,
                },
            }
            yield {"type": "token", "content": verdict.response or ""}
            yield {
                "type": "done",
                "data": {"content": verdict.response or "", "tokens": 0,
                         "safety": verdict.category},
            }
            return

        routing = await self.router.route(
            clean, previous_agent=previous_agent, forced=forced_agent
        )
        memory_updates = await self.memory.learn_from_message(user_id, clean)
        profile_context = await self.memory.build_profile_context(user_id)
        history = await self.memory.working_window(conversation.id)

        yield {
            "type": "meta",
            "data": {
                "agent": routing.agent.value,
                "confidence": routing.confidence,
                "reason": routing.reason,
                "memory_updates": memory_updates,
            },
        }

        agent = self.agent_for(routing.agent)
        collected: list[str] = []
        citations: list[RetrievedChunk] = []

        try:
            async for delta, sources in agent.stream(
                clean,
                history=history,
                profile_context=profile_context,
                conversation_summary=conversation.summary,
            ):
                if sources:
                    citations = sources
                    yield {
                        "type": "sources",
                        "data": {"sources": [c.to_citation() for c in sources]},
                    }
                collected.append(delta)
                yield {"type": "token", "content": delta}
        except Exception as exc:
            logger.error("streaming failed", extra={"error": str(exc)})
            yield {
                "type": "error",
                "content": (
                    "The response was interrupted. Please try sending your "
                    "message again."
                ),
            }
            return

        # The disclaimer cannot be enforced mid-stream, so it is appended as a
        # final token frame — the guarantee holds for streaming responses too.
        raw = "".join(collected)
        final = enforce_disclaimer(raw)
        if final != raw:
            yield {"type": "token", "content": final[len(raw) :]}

        yield {
            "type": "done",
            "data": {
                "content": final,
                "agent": routing.agent.value,
                "sources": [c.to_citation() for c in citations],
                "tokens": agent.llm.estimate_tokens(final),
                "model": getattr(self.llm, "model", self.llm.name),
            },
        }
