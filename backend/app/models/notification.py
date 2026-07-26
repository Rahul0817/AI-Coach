"""In-app notifications: reminders, AI insights and achievements."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin, UserOwnedMixin, UUIDPrimaryKeyMixin
from app.models.enums import NotificationType

if TYPE_CHECKING:  # pragma: no cover
    from app.models.user import User


class Notification(Base, UUIDPrimaryKeyMixin, UserOwnedMixin, TimestampMixin):
    """A message surfaced in the user's notification tray.

    ``scheduled_for`` lets the same table serve both "tell them now" and
    "remind them at 9am tomorrow" — the delivery query simply filters on
    ``scheduled_for <= now()``, so no separate scheduler storage is needed.
    """

    __tablename__ = "notifications"
    __table_args__ = (
        Index("ix_notification_user_unread", "user_id", "is_read", "created_at"),
    )

    type: Mapped[str] = mapped_column(
        String(16), default=NotificationType.SYSTEM.value, nullable=False
    )
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    #: Deep link into the app, e.g. "/dashboard/cycle".
    action_url: Mapped[str | None] = mapped_column(String(300), nullable=True)
    icon: Mapped[str] = mapped_column(String(40), default="bell", nullable=False)

    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    read_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    scheduled_for: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    #: Arbitrary payload for the frontend (chart ids, streak counts, …).
    payload: Mapped[dict] = mapped_column(
        JSONB, default=dict, nullable=False, server_default="{}"
    )

    user: Mapped["User"] = relationship(back_populates="notifications")
