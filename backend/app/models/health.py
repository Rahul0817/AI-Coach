"""Cycle tracking, symptom logging and ML risk-assessment records."""

from __future__ import annotations

import uuid
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    Date,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin, UserOwnedMixin, UUIDPrimaryKeyMixin
from app.models.enums import FlowIntensity, RiskBand

if TYPE_CHECKING:  # pragma: no cover
    from app.models.user import User


class CycleLog(Base, UUIDPrimaryKeyMixin, UserOwnedMixin, TimestampMixin):
    """One menstrual cycle, recorded from its first day of bleeding.

    ``cycle_length_days`` is stored rather than derived on read: it is written
    once when the *next* cycle starts, and pre-computing it keeps the analytics
    queries (irregularity variance, average length) simple index scans.
    """

    __tablename__ = "cycle_logs"
    __table_args__ = (
        UniqueConstraint("user_id", "start_date", name="uq_cycle_user_start"),
        CheckConstraint(
            "period_length_days IS NULL OR period_length_days BETWEEN 1 AND 21",
            name="period_length_range",
        ),
        CheckConstraint(
            "cycle_length_days IS NULL OR cycle_length_days BETWEEN 10 AND 180",
            name="cycle_length_range",
        ),
        Index("ix_cycle_user_start_desc", "user_id", "start_date"),
    )

    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_length_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cycle_length_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    flow_intensity: Mapped[str] = mapped_column(
        String(16), default=FlowIntensity.MEDIUM.value, nullable=False
    )
    pain_level: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 0–10
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship(back_populates="cycles")

    @property
    def is_irregular(self) -> bool:
        """True when this cycle sits outside the commonly cited 21–35 day band.

        Descriptive only. A single long cycle is normal; the pattern across many
        cycles is what the analytics layer reports.
        """
        if self.cycle_length_days is None:
            return False
        return not 21 <= self.cycle_length_days <= 35


class SymptomLog(Base, UUIDPrimaryKeyMixin, UserOwnedMixin, TimestampMixin):
    """A single symptom observation on a given day."""

    __tablename__ = "symptom_logs"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "logged_on", "symptom", name="uq_symptom_user_day_type"
        ),
        CheckConstraint("severity BETWEEN 1 AND 5", name="severity_range"),
        Index("ix_symptom_user_date", "user_id", "logged_on"),
    )

    logged_on: Mapped[date] = mapped_column(Date, nullable=False)
    symptom: Mapped[str] = mapped_column(String(40), nullable=False)
    severity: Mapped[int] = mapped_column(Integer, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship(back_populates="symptoms")


class Prediction(Base, UUIDPrimaryKeyMixin, UserOwnedMixin, TimestampMixin):
    """A stored ML risk assessment together with its SHAP explanation.

    Persisting the exact feature vector and ``model_version`` alongside the
    score is what makes the system auditable: months later you can answer
    "why did the model say that?" and reproduce the result bit-for-bit, which
    is table stakes for anything health adjacent.
    """

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("risk_score BETWEEN 0 AND 1", name="risk_score_range"),
        Index("ix_prediction_user_created", "user_id", "created_at"),
    )

    model_version: Mapped[str] = mapped_column(String(40), nullable=False)
    model_name: Mapped[str] = mapped_column(String(60), nullable=False)

    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_band: Mapped[str] = mapped_column(
        String(12), default=RiskBand.LOW.value, nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # [{"feature": "cycle_length_days", "value": 46, "shap": 0.18,
    #   "direction": "increases", "explanation": "..."}]
    explanation: Mapped[list] = mapped_column(
        JSONB, default=list, nullable=False, server_default="[]"
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship(back_populates="predictions")

    @staticmethod
    def band_for(score: float) -> RiskBand:
        """Map a probability to a coarse band used across the UI.

        Thresholds are product decisions, not clinical cut-offs, and are kept in
        one place so the API, the dashboard and the agents never disagree.
        """
        if score < 0.35:
            return RiskBand.LOW
        if score < 0.65:
            return RiskBand.MODERATE
        return RiskBand.HIGH
