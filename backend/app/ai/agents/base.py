"""The specialist agent base class and its variants.

Agents share almost all of their behaviour — retrieve, build a prompt, call the
model, enforce safety — and differ in persona, retrieval scope, temperature and
occasionally in how they post-process a response. That shape argues for a
template-method base class with narrow override points rather than eight
independent implementations, which is what this is.

Two agents genuinely need different behaviour and get subclasses:
:class:`BloodReportAgent` (must never be creative and refuses without data) and
:class:`FoodAnalyzerAgent` (works from vision output rather than retrieval).
The rest are configuration.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.ai.agents.prompts import (
    AgentDefinition,
    build_system_prompt,
    get_definition,
)
from app.ai.agents.safety import enforce_disclaimer
from app.ai.llm.base import GenerationConfig, LLMMessage, LLMProvider, LLMResponse
from app.ai.rag.retriever import KnowledgeRetriever, RetrievedChunk
from app.core.config import settings
from app.core.logging import get_logger
from app.models.enums import AgentName

logger = get_logger(__name__)


@dataclass(slots=True)
class AgentResponse:
    """What an agent hands back to the chat service."""

    content: str
    agent: AgentName
    sources: list[RetrievedChunk] = field(default_factory=list)
    tokens_used: int = 0
    model: str = ""
    used_rag: bool = False
    suggested_followups: list[str] = field(default_factory=list)


class SpecialistAgent:
    """Template-method base: retrieve → prompt → generate → enforce."""

    def __init__(
        self,
        definition: AgentDefinition,
        llm: LLMProvider,
        retriever: KnowledgeRetriever,
    ) -> None:
        self.definition = definition
        self.llm = llm
        self.retriever = retriever

    @property
    def name(self) -> AgentName:
        return self.definition.name

    # ------------------------------------------------------- override points
    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Fetch grounding context.

        Retrieval is scoped to the agent's own knowledge category first. If that
        yields nothing — a nutrition question phrased in cycle vocabulary, say —
        it retries across the whole corpus rather than answering ungrounded. A
        narrow scope that returns nothing is worse than a broad one that returns
        something relevant.
        """
        category = self.definition.knowledge_category
        chunks = await self.retriever.retrieve(query, category=category)
        if not chunks and category is not None:
            chunks = await self.retriever.retrieve(query, category=None)
        return chunks

    def generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            temperature=self.definition.temperature,
            max_tokens=settings.llm_max_tokens,
        )

    def followups(self, query: str) -> list[str]:
        """Suggested next questions, shown as chips under the answer."""
        return self.definition.example_prompts[:3]

    def post_process(self, content: str) -> str:
        """Final transformation before the response leaves the agent."""
        return enforce_disclaimer(content)

    # ---------------------------------------------------------- main entry
    async def answer(
        self,
        query: str,
        *,
        history: list[LLMMessage] | None = None,
        profile_context: dict[str, Any] | None = None,
        conversation_summary: str | None = None,
        prefix_note: str | None = None,
    ) -> AgentResponse:
        """Produce a complete grounded answer."""
        chunks = await self.retrieve(query)
        messages = self._assemble(
            query, chunks, history, profile_context, conversation_summary
        )

        response: LLMResponse = await self.llm.complete(
            messages, self.generation_config()
        )
        content = self.post_process(response.content)
        if prefix_note:
            content = f"{prefix_note}\n\n{content}"

        return AgentResponse(
            content=content,
            agent=self.name,
            sources=chunks,
            tokens_used=response.total_tokens,
            model=response.model,
            used_rag=bool(chunks),
            suggested_followups=self.followups(query),
        )

    async def stream(
        self,
        query: str,
        *,
        history: list[LLMMessage] | None = None,
        profile_context: dict[str, Any] | None = None,
        conversation_summary: str | None = None,
    ) -> AsyncIterator[tuple[str, list[RetrievedChunk]]]:
        """Yield ``(delta, sources)`` pairs for server-sent events.

        Sources are attached to the first frame so the UI can render citation
        chips immediately rather than waiting for generation to finish.
        """
        chunks = await self.retrieve(query)
        messages = self._assemble(
            query, chunks, history, profile_context, conversation_summary
        )

        emitted_sources = False
        async for delta in self.llm.stream(messages, self.generation_config()):
            if not emitted_sources:
                yield delta, chunks
                emitted_sources = True
            else:
                yield delta, []

    # ------------------------------------------------------------- internals
    def _assemble(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[LLMMessage] | None,
        profile_context: dict[str, Any] | None,
        conversation_summary: str | None,
    ) -> list[LLMMessage]:
        system = build_system_prompt(
            self.definition,
            profile_context=profile_context,
            retrieved=chunks,
            conversation_summary=conversation_summary,
        )
        messages = [LLMMessage(role="system", content=system)]
        if history:
            messages.extend(history)
        messages.append(LLMMessage(role="user", content=query))
        return messages


class BloodReportAgent(SpecialistAgent):
    """Explains lab values. Deterministic, and refuses without data."""

    def generation_config(self) -> GenerationConfig:
        # Temperature 0: there is exactly one correct explanation of what SHBG
        # measures, and creativity here is purely a source of error.
        return GenerationConfig(temperature=0.0, max_tokens=settings.llm_max_tokens)

    def followups(self, query: str) -> list[str]:
        return [
            "What questions should I ask my doctor about these results?",
            "Which of these markers relate to insulin resistance?",
            "What do reference ranges actually mean?",
        ]


class FoodAnalyzerAgent(SpecialistAgent):
    """Interprets meal photos. Retrieval is a supporting act here.

    The primary input is the vision pipeline's structured output, not the
    knowledge base, so retrieval is narrowed to nutrition guidance that helps
    phrase a recommendation rather than to establish facts about the plate.
    """

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        return await self.retriever.retrieve(query, category="nutrition", top_k=3)

    def followups(self, query: str) -> list[str]:
        return [
            "How could I make this meal more PCOS-friendly?",
            "How much protein should this meal have had?",
            "Log this to my meal diary",
        ]


class CycleAgent(SpecialistAgent):
    """Cycle interpretation, with a hard-coded contraception caveat."""

    def post_process(self, content: str) -> str:
        content = super().post_process(content)
        # Enforced in code, not left to the prompt: cycle predictions in PCOS
        # are unreliable, and someone treating them as contraception is a
        # foreseeable and serious harm.
        caveat = (
            "Cycle predictions are estimates and are especially unreliable in "
            "PCOS. **Never use them as a form of contraception.**"
        )
        if "contracept" not in content.lower():
            content = content.replace("\n\n---\n\n", f"\n\n_{caveat}_\n\n---\n\n", 1)
            if caveat not in content:
                content = f"{content}\n\n_{caveat}_"
        return content


#: Maps an agent to the class that implements it. Everything not listed here
#: uses the configuration-only base class.
AGENT_CLASSES: dict[AgentName, type[SpecialistAgent]] = {
    AgentName.BLOOD_REPORT_ANALYZER: BloodReportAgent,
    AgentName.FOOD_ANALYZER: FoodAnalyzerAgent,
    AgentName.CYCLE_TRACKER_ASSISTANT: CycleAgent,
}


def build_agent(
    name: AgentName, llm: LLMProvider, retriever: KnowledgeRetriever
) -> SpecialistAgent:
    """Instantiate the agent registered for ``name``."""
    definition = get_definition(name)
    agent_class = AGENT_CLASSES.get(name, SpecialistAgent)
    return agent_class(definition, llm, retriever)
