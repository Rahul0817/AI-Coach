"""Notifications: reminders, generated insights and achievements."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.enums import NotificationType
from app.models.notification import Notification
from app.repositories.notification import NotificationRepository
from app.schemas.notification import NotificationCreate

logger = get_logger(__name__)

#: Window used to suppress duplicate generated notifications.
DEDUPE_WINDOW = timedelta(hours=20)


class NotificationService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.notifications = NotificationRepository(session)

    async def create(
        self, user_id: uuid.UUID, payload: NotificationCreate
    ) -> Notification:
        notification = await self.notifications.create(
            user_id=user_id,
            type=payload.type.value,
            title=payload.title,
            body=payload.body,
            action_url=payload.action_url,
            icon=payload.icon,
            scheduled_for=payload.scheduled_for,
            payload=payload.payload,
        )
        await self.session.commit()
        return notification

    async def list(
        self, user_id: uuid.UUID, unread_only: bool = False, limit: int = 50
    ) -> list[Notification]:
        return await self.notifications.list_deliverable(
            user_id, limit=limit, unread_only=unread_only
        )

    async def counts(self, user_id: uuid.UUID) -> tuple[int, int]:
        return await self.notifications.counts(user_id)

    async def mark_read(
        self, user_id: uuid.UUID, notification_id: uuid.UUID
    ) -> Notification:
        notification = await self.notifications.mark_read(notification_id, user_id)
        await self.session.commit()
        return notification

    async def mark_all_read(self, user_id: uuid.UUID) -> int:
        count = await self.notifications.mark_all_read(user_id)
        await self.session.commit()
        return count

    async def delete(self, user_id: uuid.UUID, notification_id: uuid.UUID) -> None:
        await self.notifications.delete_for_user(notification_id, user_id)
        await self.session.commit()

    # ------------------------------------------------------------ generation
    async def _emit(
        self,
        user_id: uuid.UUID,
        *,
        type_: NotificationType,
        title: str,
        body: str,
        icon: str,
        action_url: str | None = None,
        payload: dict | None = None,
    ) -> Notification | None:
        """Create a generated notification unless an identical one is recent.

        Deduplication matters because these are produced by rules that run on
        every dashboard load. Without it, opening the app five times would
        produce five identical "drink more water" cards.
        """
        since = datetime.now(timezone.utc) - DEDUPE_WINDOW
        if await self.notifications.exists_recent(user_id, title, since):
            return None
        return await self.notifications.create(
            user_id=user_id,
            type=type_.value,
            title=title,
            body=body,
            icon=icon,
            action_url=action_url,
            payload=payload or {},
        )

    async def generate_from_activity(self, user_id: uuid.UUID) -> list[Notification]:
        """Derive notifications from the user's current data.

        Called after the dashboard is computed, so it reuses work already done
        rather than re-querying.
        """
        from app.services.analytics_service import AnalyticsService

        analytics = AnalyticsService(self.session)
        dashboard = await analytics.dashboard(user_id)
        created: list[Notification] = []

        # --- streak achievements ---
        for name, streak in dashboard.streaks.items():
            if streak in {7, 14, 30, 60, 100}:
                notification = await self._emit(
                    user_id,
                    type_=NotificationType.ACHIEVEMENT,
                    title=f"{streak}-day streak: {name}",
                    body=(
                        f"You have kept '{name}' going for {streak} days straight. "
                        f"That is the point where a habit stops needing willpower."
                    ),
                    icon="flame",
                    action_url="/dashboard/habits",
                    payload={"habit": name, "streak": streak},
                )
                if notification:
                    created.append(notification)

        # --- cycle reminder ---
        days_until = dashboard.cycle_summary.get("days_until_next")
        if isinstance(days_until, int) and 0 <= days_until <= 3:
            notification = await self._emit(
                user_id,
                type_=NotificationType.REMINDER,
                title="Your next period is due soon",
                body=(
                    f"Based on your logged cycles, your next period is estimated "
                    f"in {days_until} day(s). Predictions are rough in PCOS — log "
                    f"the actual date when it arrives to sharpen future estimates."
                ),
                icon="calendar-heart",
                action_url="/dashboard/cycle",
            )
            if notification:
                created.append(notification)

        # --- surface the highest-priority insight ---
        attention = [i for i in dashboard.insights if i.severity == "attention"]
        if attention:
            top = attention[0]
            notification = await self._emit(
                user_id,
                type_=NotificationType.INSIGHT,
                title=top.title,
                body=top.body,
                icon="lightbulb",
                action_url=top.action_url,
                payload={"category": top.category},
            )
            if notification:
                created.append(notification)

        if created:
            await self.session.commit()
            logger.info(
                "notifications generated",
                extra={"user_id": str(user_id), "count": len(created)},
            )
        return created

    async def schedule_daily_reminders(
        self, user_id: uuid.UUID, hour: int = 9
    ) -> list[Notification]:
        """Queue tomorrow's logging reminders.

        Stored with a ``scheduled_for`` timestamp rather than held by a
        scheduler process, so the delivery query is simply
        ``scheduled_for <= now()`` and no extra infrastructure is required.
        """
        tomorrow = date.today() + timedelta(days=1)
        when = datetime.combine(
            tomorrow, datetime.min.time(), tzinfo=timezone.utc
        ).replace(hour=hour)

        templates = [
            ("Log your morning check-in",
             "Weight, sleep and mood take under a minute and are what make your "
             "trends meaningful.", "sunrise", "/dashboard"),
            ("How is your water going?",
             "A quick tap to log your intake keeps the streak alive.",
             "droplet", "/dashboard/water"),
        ]

        created: list[Notification] = []
        for title, body, icon, url in templates:
            notification = await self.notifications.create(
                user_id=user_id,
                type=NotificationType.REMINDER.value,
                title=title,
                body=body,
                icon=icon,
                action_url=url,
                scheduled_for=when,
            )
            created.append(notification)

        await self.session.commit()
        return created

    async def welcome(self, user_id: uuid.UUID, name: str) -> Notification:
        """First notification a new account sees."""
        return await self.create(
            user_id,
            NotificationCreate(
                type=NotificationType.SYSTEM,
                title=f"Welcome to Oviora, {name.split()[0]}",
                body=(
                    "Start by completing your profile — height, weight and cycle "
                    "history let every recommendation be about you rather than "
                    "about people in general. Then try asking the AI anything "
                    "about PCOS."
                ),
                icon="sparkles",
                action_url="/dashboard/profile",
            ),
        )
