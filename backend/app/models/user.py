"""Account and profile tables.

``users`` holds only what authentication needs; everything health-related lives
in ``profiles``. Splitting them is not gratuitous normalisation — it means the
login path reads a small hot table, and it keeps sensitive health attributes out
of the row that gets loaded on every single authenticated request.
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
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import ActivityLevel, DiagnosisStatus, DietaryPreference

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.chat import Conversation
    from app.models.health import CycleLog, Prediction, SymptomLog
    from app.models.notification import Notification
    from app.models.report import BloodReport
    from app.models.tracking import (
        Habit, MealLog, MoodLog, SleepLog, WaterLog, WeightLog, WorkoutLog,
    )


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Authentication identity."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(320), unique=True, nullable=False, index=True
    )
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Consecutive failed logins; reset on success. Used for lockout backoff.
    failed_login_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    profile: Mapped["Profile | None"] = relationship(
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="noload"
    )
    cycles: Mapped[list["CycleLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="noload"
    )
    symptoms: Mapped[list["SymptomLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="noload"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="noload"
    )
    habits: Mapped[list["Habit"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="noload"
    )
    meals: Mapped[list["MealLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="noload"
    )
    workouts: Mapped[list["WorkoutLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="noload"
    )
    water_logs: Mapped[list["WaterLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="noload"
    )
    sleep_logs: Mapped[list["SleepLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="noload"
    )
    weight_logs: Mapped[list["WeightLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="noload"
    )
    mood_logs: Mapped[list["MoodLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="noload"
    )
    reports: Mapped[list["BloodReport"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="noload"
    )
    notifications: Mapped[list["Notification"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="noload"
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<User {self.email}>"


class Profile(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Health context used to personalise every AI response and prediction."""

    __tablename__ = "profiles"
    __table_args__ = (
        CheckConstraint("height_cm IS NULL OR height_cm BETWEEN 90 AND 250",
                        name="height_range"),
        CheckConstraint("weight_kg IS NULL OR weight_kg BETWEEN 25 AND 350",
                        name="weight_range"),
        CheckConstraint(
            "average_cycle_length IS NULL OR average_cycle_length BETWEEN 15 AND 120",
            name="cycle_length_range",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    height_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)

    activity_level: Mapped[str] = mapped_column(
        String(20), default=ActivityLevel.MODERATE.value, nullable=False
    )
    diagnosis_status: Mapped[str] = mapped_column(
        String(24), default=DiagnosisStatus.UNDIAGNOSED.value, nullable=False
    )
    dietary_preference: Mapped[str] = mapped_column(
        String(20), default=DietaryPreference.OMNIVORE.value, nullable=False
    )

    average_cycle_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    average_period_length: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Free-form personalisation inputs surfaced to the agents.
    primary_goal: Mapped[str | None] = mapped_column(String(200), nullable=True)
    allergies: Mapped[list[str]] = mapped_column(
        JSONB, default=list, nullable=False, server_default="[]"
    )
    medical_conditions: Mapped[list[str]] = mapped_column(
        JSONB, default=list, nullable=False, server_default="[]"
    )
    medications: Mapped[list[str]] = mapped_column(
        JSONB, default=list, nullable=False, server_default="[]"
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    timezone: Mapped[str] = mapped_column(
        String(64), default="UTC", nullable=False
    )
    # Long-lived facts the memory layer extracted from conversation, e.g.
    # {"age": 22, "prefers": "vegetarian breakfasts"}.
    ai_memory: Mapped[dict] = mapped_column(
        JSONB, default=dict, nullable=False, server_default="{}"
    )

    user: Mapped["User"] = relationship(back_populates="profile")

    # ------------------------------------------------------------- computed
    @property
    def age(self) -> int | None:
        """Age in whole years, or ``None`` when no birth date is recorded."""
        if not self.date_of_birth:
            return None
        today = date.today()
        return (
            today.year
            - self.date_of_birth.year
            - ((today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day))
        )

    @property
    def bmi(self) -> float | None:
        """Body-mass index rounded to one decimal place."""
        if not self.height_cm or not self.weight_kg or self.height_cm <= 0:
            return None
        metres = self.height_cm / 100
        return round(self.weight_kg / (metres * metres), 1)

    @property
    def bmi_category(self) -> str | None:
        """WHO band for :attr:`bmi` — descriptive only, never diagnostic."""
        bmi = self.bmi
        if bmi is None:
            return None
        if bmi < 18.5:
            return "underweight"
        if bmi < 25:
            return "healthy"
        if bmi < 30:
            return "overweight"
        return "obese"

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Profile user_id={self.user_id}>"
