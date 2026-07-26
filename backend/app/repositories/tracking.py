"""Repositories for the daily trackers.

Several of these expose an ``upsert`` helper. Water, sleep, weight and mood are
naturally *one row per day*, and the UI edits them repeatedly as the day goes on
— so "create or update" is the primitive the service layer actually wants, and
the unique constraint in the schema guarantees it stays true.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.models.tracking import (
    Habit,
    HabitEntry,
    MealLog,
    MoodLog,
    SleepLog,
    WaterLog,
    WeightLog,
    WorkoutLog,
)
from app.repositories.base import BaseRepository


class HabitRepository(BaseRepository[Habit]):
    model = Habit

    async def list_active(self, user_id: uuid.UUID) -> list[Habit]:
        stmt = (
            select(Habit)
            .where(Habit.user_id == user_id, Habit.is_archived.is_(False))
            .options(selectinload(Habit.entries))
            .order_by(Habit.created_at.asc())
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def get_with_entries(self, habit_id: uuid.UUID, user_id: uuid.UUID) -> Habit:
        stmt = (
            select(Habit)
            .where(Habit.id == habit_id, Habit.user_id == user_id)
            .options(selectinload(Habit.entries))
        )
        habit = (await self.session.execute(stmt)).scalar_one_or_none()
        if habit is None:
            from app.core.exceptions import NotFoundError

            raise NotFoundError(f"Habit {habit_id} was not found.")
        return habit


class HabitEntryRepository(BaseRepository[HabitEntry]):
    model = HabitEntry

    async def upsert(
        self,
        habit_id: uuid.UUID,
        logged_on: date,
        completed_count: int,
        note: str | None = None,
    ) -> HabitEntry:
        existing = await self.find_one_by(habit_id=habit_id, logged_on=logged_on)
        if existing:
            return await self.set_fields(
                existing, completed_count=completed_count, note=note
            )
        return await self.create(
            habit_id=habit_id,
            logged_on=logged_on,
            completed_count=completed_count,
            note=note,
        )

    async def entries_since(self, habit_id: uuid.UUID, since: date) -> list[HabitEntry]:
        stmt = (
            select(HabitEntry)
            .where(HabitEntry.habit_id == habit_id, HabitEntry.logged_on >= since)
            .order_by(HabitEntry.logged_on.desc())
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def delete_for_day(self, habit_id: uuid.UUID, logged_on: date) -> bool:
        existing = await self.find_one_by(habit_id=habit_id, logged_on=logged_on)
        if existing is None:
            return False
        await self.delete(existing)
        return True


class MealRepository(BaseRepository[MealLog]):
    model = MealLog

    async def for_day(self, user_id: uuid.UUID, day: date) -> list[MealLog]:
        stmt = (
            select(MealLog)
            .where(MealLog.user_id == user_id, MealLog.logged_on == day)
            .order_by(MealLog.created_at.asc())
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def daily_totals(
        self, user_id: uuid.UUID, start: date, end: date
    ) -> list[dict]:
        """Per-day macro totals, aggregated in the database."""
        stmt = (
            select(
                MealLog.logged_on,
                func.sum(MealLog.calories).label("calories"),
                func.sum(MealLog.protein_g).label("protein"),
                func.sum(MealLog.carbs_g).label("carbs"),
                func.sum(MealLog.fat_g).label("fat"),
                func.sum(MealLog.fibre_g).label("fibre"),
                func.count().label("meals"),
            )
            .where(
                MealLog.user_id == user_id,
                MealLog.logged_on >= start,
                MealLog.logged_on <= end,
            )
            .group_by(MealLog.logged_on)
            .order_by(MealLog.logged_on.asc())
        )
        rows = (await self.session.execute(stmt)).all()
        return [
            {
                "date": r.logged_on,
                "calories": round(float(r.calories or 0), 1),
                "protein_g": round(float(r.protein or 0), 1),
                "carbs_g": round(float(r.carbs or 0), 1),
                "fat_g": round(float(r.fat or 0), 1),
                "fibre_g": round(float(r.fibre or 0), 1),
                "meals": int(r.meals),
            }
            for r in rows
        ]


class WorkoutRepository(BaseRepository[WorkoutLog]):
    model = WorkoutLog

    async def weekly_minutes(
        self, user_id: uuid.UUID, start: date, end: date
    ) -> list[dict]:
        stmt = (
            select(
                WorkoutLog.logged_on,
                func.sum(WorkoutLog.duration_minutes).label("minutes"),
                func.count().label("sessions"),
            )
            .where(
                WorkoutLog.user_id == user_id,
                WorkoutLog.logged_on >= start,
                WorkoutLog.logged_on <= end,
            )
            .group_by(WorkoutLog.logged_on)
            .order_by(WorkoutLog.logged_on.asc())
        )
        rows = (await self.session.execute(stmt)).all()
        return [
            {
                "date": r.logged_on,
                "minutes": int(r.minutes or 0),
                "sessions": int(r.sessions),
            }
            for r in rows
        ]


class _DailyUpsertRepository(BaseRepository):
    """Shared upsert for trackers constrained to one row per user per day."""

    async def upsert(self, user_id: uuid.UUID, logged_on: date, **fields):
        existing = await self.find_one_by(user_id=user_id, logged_on=logged_on)
        if existing:
            return await self.set_fields(existing, **fields)
        return await self.create(user_id=user_id, logged_on=logged_on, **fields)

    async def for_day(self, user_id: uuid.UUID, day: date):
        return await self.find_one_by(user_id=user_id, logged_on=day)


class WaterRepository(_DailyUpsertRepository):
    model = WaterLog


class SleepRepository(_DailyUpsertRepository):
    model = SleepLog

    async def average_hours(
        self, user_id: uuid.UUID, start: date, end: date
    ) -> float | None:
        stmt = select(func.avg(SleepLog.hours)).where(
            SleepLog.user_id == user_id,
            SleepLog.logged_on >= start,
            SleepLog.logged_on <= end,
        )
        value = (await self.session.execute(stmt)).scalar_one_or_none()
        return round(float(value), 1) if value is not None else None


class WeightRepository(_DailyUpsertRepository):
    model = WeightLog

    async def latest(self, user_id: uuid.UUID) -> WeightLog | None:
        stmt = (
            select(WeightLog)
            .where(WeightLog.user_id == user_id)
            .order_by(WeightLog.logged_on.desc())
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def change_over(self, user_id: uuid.UUID, days: int) -> float | None:
        """Net weight change across the last ``days`` days, or ``None``.

        Returns ``None`` unless there are at least two measurements, because a
        single data point cannot express a trend.
        """
        since = date.today() - timedelta(days=days)
        stmt = (
            select(WeightLog)
            .where(WeightLog.user_id == user_id, WeightLog.logged_on >= since)
            .order_by(WeightLog.logged_on.asc())
        )
        rows = list((await self.session.execute(stmt)).scalars().all())
        if len(rows) < 2:
            return None
        return round(rows[-1].weight_kg - rows[0].weight_kg, 1)


class MoodRepository(_DailyUpsertRepository):
    model = MoodLog
