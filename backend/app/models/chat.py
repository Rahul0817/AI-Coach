"""Conversation and message persistence.

Redis holds the *working* memory (fast prompt assembly); Postgres is the
durable record. Keeping both means a Redis flush loses nothing — the memory
service rehydrates the window from these tables on the next request.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.types import GUID, JSONDict
from app.models.base import TimestampMixin, UserOwnedMixin, UUIDPrimaryKeyMixin
from app.models.enums import MessageRole

if TYPE_CHECKING:  # pragma: no cover
    from app.models.user import User


class Conversation(Base, UUIDPrimaryKeyMixin, UserOwnedMixin, TimestampMixin):
    """A chat thread, equivalent to one ChatGPT conversation."""

    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversation_user_updated", "user_id", "updated_at"),
    )

    title: Mapped[str] = mapped_column(
        String(160), default="New conversation", nullable=False
    )
    is_pinned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    #: Rolling summary of turns that have aged out of the verbatim window.
    #: This is what gives the assistant memory beyond its context limit.
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summarised_up_to: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.sequence",
        lazy="noload",
    )


class Message(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One turn in a conversation, with full AI provenance attached.

    ``agent``, ``sources`` and ``tokens_used`` are not decoration: they let the
    UI show which specialist answered and which guideline it cited, and they
    make cost attribution per user possible without a separate analytics store.
    """

    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_message_conversation_seq", "conversation_id", "sequence"),
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    #: Monotonic position within the thread. Ordering by ``created_at`` is
    #: unreliable when a user and assistant message land in the same millisecond.
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)

    role: Mapped[str] = mapped_column(String(12), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    agent: Mapped[str | None] = mapped_column(String(32), nullable=True)
    routing_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: [{"title": "...", "source": "...", "score": 0.82, "snippet": "..."}]
    sources: Mapped[list] = mapped_column(
        JSONDict, default=list, nullable=False, server_default="[]"
    )
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model: Mapped[str | None] = mapped_column(String(60), nullable=True)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

    @property
    def is_user(self) -> bool:
        return self.role == MessageRole.USER.value
