"""Chat orchestration and conversation persistence."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.agents.orchestrator import AgentOrchestrator
from app.core.logging import get_logger
from app.models.chat import Conversation, Message
from app.models.enums import AgentName, MessageRole
from app.repositories.chat import ConversationRepository, MessageRepository
from app.schemas.chat import ChatRequest, ConversationUpdate

logger = get_logger(__name__)

#: Characters of the first user message used to auto-title a conversation.
TITLE_LENGTH = 60


class ChatService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.conversations = ConversationRepository(session)
        self.messages = MessageRepository(session)
        self.orchestrator = AgentOrchestrator(session)

    # ------------------------------------------------------- conversations
    async def get_or_create_conversation(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID | None
    ) -> Conversation:
        if conversation_id is not None:
            return await self.conversations.get_for_user(conversation_id, user_id)
        return await self.conversations.create(user_id=user_id)

    async def list_conversations(
        self, user_id: uuid.UUID, include_archived: bool = False
    ) -> list[dict]:
        rows = await self.conversations.list_for_sidebar(
            user_id, include_archived=include_archived
        )
        out: list[dict] = []
        for conversation in rows:
            last = await self.messages.recent_window(conversation.id, 1)
            out.append(
                {
                    "id": conversation.id,
                    "title": conversation.title,
                    "is_pinned": conversation.is_pinned,
                    "is_archived": conversation.is_archived,
                    "message_count": conversation.message_count,
                    "created_at": conversation.created_at,
                    "updated_at": conversation.updated_at,
                    "last_message_preview": (last[0].content[:120] if last else None),
                }
            )
        return out

    async def get_conversation(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> Conversation:
        return await self.conversations.get_with_messages(conversation_id, user_id)

    async def update_conversation(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, payload: ConversationUpdate
    ) -> Conversation:
        conversation = await self.conversations.get_for_user(conversation_id, user_id)
        updates = payload.model_dump(exclude_unset=True)
        # Booleans must be settable to False, which the None-skipping updater
        # would otherwise discard.
        for field in ("is_pinned", "is_archived"):
            if field in updates and updates[field] is not None:
                setattr(conversation, field, updates.pop(field))
        if updates.get("title"):
            conversation.title = updates["title"]
        await self.session.commit()
        return conversation

    async def delete_conversation(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> None:
        conversation = await self.conversations.get_for_user(conversation_id, user_id)
        await self.orchestrator.memory.forget_conversation(conversation.id)
        await self.conversations.delete(conversation)
        await self.session.commit()

    # --------------------------------------------------------------- chat
    async def send(self, user_id: uuid.UUID, payload: ChatRequest) -> dict[str, Any]:
        """Handle one non-streaming chat turn end to end."""
        started = time.perf_counter()
        conversation = await self.get_or_create_conversation(
            user_id, payload.conversation_id
        )
        previous_agent = await self._previous_agent(conversation.id)

        user_message = await self._append(
            conversation, MessageRole.USER.value, payload.message
        )

        result = await self.orchestrator.handle(
            user_id=user_id,
            conversation=conversation,
            message=payload.message,
            forced_agent=payload.agent,
            previous_agent=previous_agent,
        )

        latency_ms = int((time.perf_counter() - started) * 1000)
        assistant_message = await self._append(
            conversation,
            MessageRole.ASSISTANT.value,
            result.response.content,
            agent=result.routing.agent.value,
            routing_confidence=result.routing.confidence,
            sources=[c.to_citation() for c in result.response.sources],
            tokens_used=result.response.tokens_used,
            latency_ms=latency_ms,
            model=result.response.model,
        )

        # Working memory is written after persistence so a failed commit cannot
        # leave the cache holding a turn the database never recorded.
        await self.orchestrator.memory.remember_turn(
            conversation.id, MessageRole.USER.value, payload.message
        )
        await self.orchestrator.memory.remember_turn(
            conversation.id, MessageRole.ASSISTANT.value, result.response.content
        )

        if conversation.title == "New conversation":
            conversation.title = self._derive_title(payload.message)
        await self.orchestrator.memory.maybe_summarise(conversation)
        await self.session.commit()

        return {
            "conversation_id": conversation.id,
            "message": assistant_message,
            "routing": {
                "agent": result.routing.agent,
                "confidence": result.routing.confidence,
                "reason": result.routing.reason,
                "matched_signals": result.routing.matched_signals,
            },
            "used_rag": result.response.used_rag,
            "memory_updates": result.memory_updates,
            "suggested_followups": result.response.suggested_followups,
            "user_message_id": user_message.id,
        }

    async def stream(
        self, user_id: uuid.UUID, payload: ChatRequest
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE frames, persisting the exchange once generation completes."""
        started = time.perf_counter()
        conversation = await self.get_or_create_conversation(
            user_id, payload.conversation_id
        )
        previous_agent = await self._previous_agent(conversation.id)

        await self._append(conversation, MessageRole.USER.value, payload.message)
        if conversation.title == "New conversation":
            conversation.title = self._derive_title(payload.message)
        await self.session.commit()

        # The conversation id goes out first so the client can attach to the
        # thread before any tokens arrive.
        yield {"type": "start", "data": {"conversation_id": str(conversation.id)}}

        final: dict[str, Any] | None = None
        async for frame in self.orchestrator.stream(
            user_id=user_id,
            conversation=conversation,
            message=payload.message,
            forced_agent=payload.agent,
            previous_agent=previous_agent,
        ):
            if frame["type"] == "done":
                final = frame["data"]
            yield frame

        if final is None:
            return  # An error frame was emitted; nothing to persist.

        latency_ms = int((time.perf_counter() - started) * 1000)
        await self._append(
            conversation,
            MessageRole.ASSISTANT.value,
            final.get("content", ""),
            agent=final.get("agent"),
            sources=final.get("sources", []),
            tokens_used=final.get("tokens"),
            latency_ms=latency_ms,
            model=final.get("model"),
        )
        await self.orchestrator.memory.remember_turn(
            conversation.id, MessageRole.USER.value, payload.message
        )
        await self.orchestrator.memory.remember_turn(
            conversation.id, MessageRole.ASSISTANT.value, final.get("content", "")
        )
        await self.orchestrator.memory.maybe_summarise(conversation)
        await self.session.commit()

    # ---------------------------------------------------------- internals
    async def _append(
        self, conversation: Conversation, role: str, content: str, **extra: Any
    ) -> Message:
        sequence = await self.messages.next_sequence(conversation.id)
        message = await self.messages.create(
            conversation_id=conversation.id,
            sequence=sequence,
            role=role,
            content=content,
            **extra,
        )
        conversation.message_count += 1
        await self.session.flush()
        return message

    async def _previous_agent(self, conversation_id: uuid.UUID) -> AgentName | None:
        """The agent that answered last, used for routing continuity."""
        recent = await self.messages.recent_window(conversation_id, 4)
        for message in reversed(recent):
            if message.role == MessageRole.ASSISTANT.value and message.agent:
                try:
                    return AgentName(message.agent)
                except ValueError:
                    return None
        return None

    @staticmethod
    def _derive_title(first_message: str) -> str:
        """Title a thread from its opening message, as ChatGPT does."""
        cleaned = " ".join(first_message.split())
        if len(cleaned) <= TITLE_LENGTH:
            return cleaned or "New conversation"
        return cleaned[:TITLE_LENGTH].rsplit(" ", 1)[0] + "…"
