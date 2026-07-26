"""Cycle, symptom and prediction repositories."""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import func, select

from app.models.health import CycleLog, Prediction, SymptomLog
from app.repositories.base import BaseRepository


class CycleRepository(BaseRepository[CycleLog]):
    model = CycleLog

    async def list_ordered(
        self, user_id: uuid.UUID, *, limit: int = 24
    ) -> list[CycleLog]:
        """Most recent cycles first — the order the tracker UI renders."""
        stmt = (
            select(CycleLog)
            .where(CycleLog.user_id == user_id)
            .order_by(CycleLog.start_date.desc())
            .limit(limit)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def latest(self, user_id: uuid.UUID) -> CycleLog | None:
        stmt = (
            select(CycleLog)
            .where(CycleLog.user_id == user_id)
            .order_by(CycleLog.start_date.desc())
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def previous_before(
        self, user_id: uuid.UUID, start: date
    ) -> CycleLog | None:
        """The cycle immediately preceding ``start``.

        Used to back-fill ``cycle_length_days`` on the earlier record when a new
        cycle is logged — that value is only knowable once the next one begins.
        """
        stmt = (
            select(CycleLog)
            .where(CycleLog.user_id == user_id, CycleLog.start_date < start)
            .order_by(CycleLog.start_date.desc())
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def find_by_start(
        self, user_id: uuid.UUID, start: date
    ) -> CycleLog | None:
        return await self.find_one_by(user_id=user_id, start_date=start)

    async def average_length(self, user_id: uuid.UUID) -> float | None:
        stmt = select(func.avg(CycleLog.cycle_length_days)).where(
            CycleLog.user_id == user_id,
            CycleLog.cycle_length_days.isnot(None),
        )
        value = (await self.session.execute(stmt)).scalar_one_or_none()
        return round(float(value), 1) if value is not None else None


class SymptomRepository(BaseRepository[SymptomLog]):
    model = SymptomLog

    async def upsert(
        self,
        user_id: uuid.UUID,
        logged_on: date,
        symptom: str,
        severity: int,
        notes: str | None,
    ) -> SymptomLog:
        """One row per (user, day, symptom); re-logging updates the severity."""
        existing = await self.find_one_by(
            user_id=user_id, logged_on=logged_on, symptom=symptom
        )
        if existing:
            return await self.set_fields(existing, severity=severity, notes=notes)
        return await self.create(
            user_id=user_id,
            logged_on=logged_on,
            symptom=symptom,
            severity=severity,
            notes=notes,
        )

    async def frequency_summary(
        self, user_id: uuid.UUID, start: date, end: date
    ) -> list[dict]:
        """Occurrence count and mean severity per symptom in a window.

        Aggregated in SQL rather than Python: for a user with two years of daily
        logs this is one indexed scan instead of shipping thousands of rows.
        """
        stmt = (
            select(
                SymptomLog.symptom,
                func.count().label("occurrences"),
                func.avg(SymptomLog.severity).label("avg_severity"),
                func.max(SymptomLog.logged_on).label("last_logged"),
            )
            .where(
                SymptomLog.user_id == user_id,
                SymptomLog.logged_on >= start,
                SymptomLog.logged_on <= end,
            )
            .group_by(SymptomLog.symptom)
            .order_by(func.count().desc())
        )
        rows = (await self.session.execute(stmt)).all()
        return [
            {
                "symptom": row.symptom,
                "occurrences": int(row.occurrences),
                "average_severity": round(float(row.avg_severity), 2),
                "last_logged": row.last_logged,
            }
            for row in rows
        ]


class PredictionRepository(BaseRepository[Prediction]):
    model = Prediction

    async def latest(self, user_id: uuid.UUID) -> Prediction | None:
        stmt = (
            select(Prediction)
            .where(Prediction.user_id == user_id)
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def history(
        self, user_id: uuid.UUID, *, limit: int = 20
    ) -> list[Prediction]:
        stmt = (
            select(Prediction)
            .where(Prediction.user_id == user_id)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
        )
        return list((await self.session.execute(stmt)).scalars().all())
