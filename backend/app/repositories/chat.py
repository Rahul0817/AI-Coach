"""Conversation and message repositories."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.core.exceptions import NotFoundError
from app.models.chat import Conversation, Message
from app.models.enums import MessageRole
from app.repositories.base import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    model = Conversation

    async def list_for_sidebar(
        self, user_id: uuid.UUID, *, limit: int = 50, include_archived: bool = False
    ) -> list[Conversation]:
        """Pinned first, then most recently updated — ChatGPT's ordering."""
        stmt = select(Conversation).where(Conversation.user_id == user_id)
        if not include_archived:
            stmt = stmt.where(Conversation.is_archived.is_(False))
        stmt = stmt.order_by(
            Conversation.is_pinned.desc(), Conversation.updated_at.desc()
        ).limit(limit)
        return list((await self.session.execute(stmt)).scalars().all())

    async def get_with_messages(
        self, conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> Conversation:
        stmt = (
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
            .options(selectinload(Conversation.messages))
        )
        conversation = (await self.session.execute(stmt)).scalar_one_or_none()
        if conversation is None:
            raise NotFoundError(f"Conversation {conversation_id} was not found.")
        return conversation

    async def touch(self, conversation: Conversation) -> None:
        """Bump ``message_count`` so the sidebar avoids a COUNT per thread."""
        conversation.message_count += 1
        await self.session.flush()


class MessageRepository(BaseRepository[Message]):
    model = Message

    async def next_sequence(self, conversation_id: uuid.UUID) -> int:
        """Next monotonic position in the thread.

        Computed with MAX rather than COUNT so deleting a message never causes
        a sequence collision.
        """
        stmt = select(func.coalesce(func.max(Message.sequence), 0)).where(
            Message.conversation_id == conversation_id
        )
        current = int((await self.session.execute(stmt)).scalar_one())
        return current + 1

    async def recent_window(
        self, conversation_id: uuid.UUID, limit: int
    ) -> list[Message]:
        """The last ``limit`` messages, returned oldest-first for prompting.

        The database sorts descending (so the index does the work) and Python
        reverses the small result set — cheaper than an offset-based query.
        """
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.sequence.desc())
            .limit(limit)
        )
        rows = list((await self.session.execute(stmt)).scalars().all())
        return list(reversed(rows))

    async def messages_before(
        self, conversation_id: uuid.UUID, sequence: int
    ) -> list[Message]:
        """Older turns awaiting summarisation."""
        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.sequence <= sequence,
            )
            .order_by(Message.sequence.asc())
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def last_user_message(self, conversation_id: uuid.UUID) -> Message | None:
        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.role == MessageRole.USER.value,
            )
            .order_by(Message.sequence.desc())
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def total_tokens(self, user_id: uuid.UUID) -> int:
        """Lifetime token spend for a user — powers cost analytics."""
        stmt = (
            select(func.coalesce(func.sum(Message.tokens_used), 0))
            .select_from(Message)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(Conversation.user_id == user_id)
        )
        return int((await self.session.execute(stmt)).scalar_one())
