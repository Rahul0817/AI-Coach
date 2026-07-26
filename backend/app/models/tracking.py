"""Daily-tracking tables: habits, meals, workouts, water, sleep, weight, mood.

Every tracker follows the same shape — user-owned, dated, with a uniqueness
constraint that makes "log today's water" idempotent where that is the natural
semantic (one weight per day) and additive where it is not (many meals per day).
That consistency is what lets the analytics service treat them uniformly.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.types import GUID
from app.models.base import TimestampMixin, UserOwnedMixin, UUIDPrimaryKeyMixin
from app.models.enums import (
    HabitFrequency,
    Intensity,
    MealSource,
    MealType,
    MoodLabel,
    WorkoutType,
)

if TYPE_CHECKING:  # pragma: no cover
    from app.models.user import User


class Habit(Base, UUIDPrimaryKeyMixin, UserOwnedMixin, TimestampMixin):
    """A recurring behaviour the user wants to build (e.g. "10k steps")."""

    __tablename__ = "habits"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_habit_user_name"),
        CheckConstraint("target_per_period >= 1", name="habit_target_positive"),
    )

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    frequency: Mapped[str] = mapped_column(
        String(12), default=HabitFrequency.DAILY.value, nullable=False
    )
    target_per_period: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    icon: Mapped[str] = mapped_column(String(40), default="sparkles", nullable=False)
    colour: Mapped[str] = mapped_column(String(16), default="#8b5cf6", nullable=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    user: Mapped["User"] = relationship(back_populates="habits")
    entries: Mapped[list["HabitEntry"]] = relationship(
        back_populates="habit", cascade="all, delete-orphan", lazy="selectin"
    )


class HabitEntry(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A completion tick for one habit on one day."""

    __tablename__ = "habit_entries"
    __table_args__ = (
        UniqueConstraint("habit_id", "logged_on", name="uq_habit_entry_day"),
        Index("ix_habit_entry_date", "habit_id", "logged_on"),
    )

    habit_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("habits.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    logged_on: Mapped[date] = mapped_column(Date, nullable=False)
    completed_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    note: Mapped[str | None] = mapped_column(String(300), nullable=True)

    habit: Mapped["Habit"] = relationship(back_populates="entries")


class MealLog(Base, UUIDPrimaryKeyMixin, UserOwnedMixin, TimestampMixin):
    """A logged food item with macronutrients and a PCOS-relevant GI estimate."""

    __tablename__ = "meal_logs"
    __table_args__ = (
        CheckConstraint("calories >= 0", name="calories_non_negative"),
        Index("ix_meal_user_date", "user_id", "logged_on"),
    )

    logged_on: Mapped[date] = mapped_column(Date, nullable=False)
    meal_type: Mapped[str] = mapped_column(
        String(12), default=MealType.SNACK.value, nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    portion: Mapped[str | None] = mapped_column(String(80), nullable=True)

    calories: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    protein_g: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    carbs_g: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    fat_g: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    fibre_g: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    # Estimated glycaemic index: the single most actionable number for insulin
    # resistance, which affects a large share of people with PCOS.
    glycemic_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    source: Mapped[str] = mapped_column(
        String(20), default=MealSource.MANUAL.value, nullable=False
    )
    image_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship(back_populates="meals")


class WorkoutLog(Base, UUIDPrimaryKeyMixin, UserOwnedMixin, TimestampMixin):
    """A completed exercise session."""

    __tablename__ = "workout_logs"
    __table_args__ = (
        CheckConstraint("duration_minutes BETWEEN 1 AND 600", name="duration_range"),
        Index("ix_workout_user_date", "user_id", "logged_on"),
    )

    logged_on: Mapped[date] = mapped_column(Date, nullable=False)
    workout_type: Mapped[str] = mapped_column(
        String(16), default=WorkoutType.WALKING.value, nullable=False
    )
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    intensity: Mapped[str] = mapped_column(
        String(12), default=Intensity.MODERATE.value, nullable=False
    )
    calories_burned: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship(back_populates="workouts")


class WaterLog(Base, UUIDPrimaryKeyMixin, UserOwnedMixin, TimestampMixin):
    """Daily hydration total. One row per user per day, upserted on change."""

    __tablename__ = "water_logs"
    __table_args__ = (
        UniqueConstraint("user_id", "logged_on", name="uq_water_user_day"),
        CheckConstraint("millilitres BETWEEN 0 AND 15000", name="water_range"),
    )

    logged_on: Mapped[date] = mapped_column(Date, nullable=False)
    millilitres: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    goal_millilitres: Mapped[int] = mapped_column(Integer, default=2500, nullable=False)

    user: Mapped["User"] = relationship(back_populates="water_logs")

    @property
    def goal_percentage(self) -> float:
        if self.goal_millilitres <= 0:
            return 0.0
        return round(min(100.0, self.millilitres / self.goal_millilitres * 100), 1)


class SleepLog(Base, UUIDPrimaryKeyMixin, UserOwnedMixin, TimestampMixin):
    """A night of sleep, keyed by the date the user woke up."""

    __tablename__ = "sleep_logs"
    __table_args__ = (
        UniqueConstraint("user_id", "logged_on", name="uq_sleep_user_day"),
        CheckConstraint("hours BETWEEN 0 AND 24", name="sleep_hours_range"),
        CheckConstraint("quality BETWEEN 1 AND 5", name="sleep_quality_range"),
    )

    logged_on: Mapped[date] = mapped_column(Date, nullable=False)
    hours: Mapped[float] = mapped_column(Float, nullable=False)
    quality: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    bedtime: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    wake_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship(back_populates="sleep_logs")


class WeightLog(Base, UUIDPrimaryKeyMixin, UserOwnedMixin, TimestampMixin):
    """A body-weight measurement. At most one per day."""

    __tablename__ = "weight_logs"
    __table_args__ = (
        UniqueConstraint("user_id", "logged_on", name="uq_weight_user_day"),
        CheckConstraint("weight_kg BETWEEN 25 AND 350", name="weight_log_range"),
    )

    logged_on: Mapped[date] = mapped_column(Date, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    body_fat_percentage: Mapped[float | None] = mapped_column(Float, nullable=True)
    waist_cm: Mapped[float | None] = mapped_column(Float, nullable=True)

    user: Mapped["User"] = relationship(back_populates="weight_logs")


class MoodLog(Base, UUIDPrimaryKeyMixin, UserOwnedMixin, TimestampMixin):
    """A daily mood check-in with optional free-text journalling."""

    __tablename__ = "mood_logs"
    __table_args__ = (
        UniqueConstraint("user_id", "logged_on", name="uq_mood_user_day"),
        CheckConstraint("energy_level BETWEEN 1 AND 5", name="energy_range"),
        CheckConstraint("stress_level BETWEEN 1 AND 5", name="stress_range"),
    )

    logged_on: Mapped[date] = mapped_column(Date, nullable=False)
    mood: Mapped[str] = mapped_column(
        String(12), default=MoodLabel.OKAY.value, nullable=False
    )
    energy_level: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    stress_level: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    journal: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship(back_populates="mood_logs")

    #: Numeric encoding so moods can be averaged on charts.
    MOOD_SCORES = {
        MoodLabel.TERRIBLE.value: 1,
        MoodLabel.LOW.value: 2,
        MoodLabel.OKAY.value: 3,
        MoodLabel.GOOD.value: 4,
        MoodLabel.GREAT.value: 5,
    }

    @property
    def mood_score(self) -> int:
        return self.MOOD_SCORES.get(self.mood, 3)
