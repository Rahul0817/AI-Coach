"""Notification repository."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, or_, select, update

from app.models.notification import Notification
from app.repositories.base import BaseRepository


class NotificationRepository(BaseRepository[Notification]):
    model = Notification

    async def list_deliverable(
        self, user_id: uuid.UUID, *, limit: int = 50, unread_only: bool = False
    ) -> list[Notification]:
        """Notifications that are due now — immediate ones plus matured ones."""
        now = datetime.now(UTC)
        stmt = select(Notification).where(
            Notification.user_id == user_id,
            or_(
                Notification.scheduled_for.is_(None),
                Notification.scheduled_for <= now,
            ),
        )
        if unread_only:
            stmt = stmt.where(Notification.is_read.is_(False))
        stmt = stmt.order_by(Notification.created_at.desc()).limit(limit)
        return list((await self.session.execute(stmt)).scalars().all())

    async def counts(self, user_id: uuid.UUID) -> tuple[int, int]:
        now = datetime.now(UTC)
        base = (
            Notification.user_id == user_id,
            or_(
                Notification.scheduled_for.is_(None),
                Notification.scheduled_for <= now,
            ),
        )
        total_stmt = select(func.count()).select_from(Notification).where(*base)
        unread_stmt = (
            select(func.count())
            .select_from(Notification)
            .where(*base, Notification.is_read.is_(False))
        )
        total = int((await self.session.execute(total_stmt)).scalar_one())
        unread = int((await self.session.execute(unread_stmt)).scalar_one())
        return total, unread

    async def mark_read(
        self, notification_id: uuid.UUID, user_id: uuid.UUID
    ) -> Notification:
        notification = await self.get_for_user(notification_id, user_id)
        notification.is_read = True
        notification.read_at = datetime.now(UTC)
        await self.session.flush()
        return notification

    async def mark_all_read(self, user_id: uuid.UUID) -> int:
        """Bulk update — one statement instead of N round trips."""
        stmt = (
            update(Notification)
            .where(
                Notification.user_id == user_id,
                Notification.is_read.is_(False),
            )
            .values(is_read=True, read_at=datetime.now(UTC))
        )
        result = await self.session.execute(stmt)
        return int(result.rowcount or 0)

    async def exists_recent(
        self, user_id: uuid.UUID, title: str, since: datetime
    ) -> bool:
        """Guard against duplicate insights when analytics runs repeatedly."""
        stmt = (
            select(func.count())
            .select_from(Notification)
            .where(
                Notification.user_id == user_id,
                Notification.title == title,
                Notification.created_at >= since,
            )
        )
        return int((await self.session.execute(stmt)).scalar_one()) > 0
