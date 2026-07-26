"""Uploaded blood reports and the biomarkers extracted from them by OCR."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.types import GUID
from app.models.base import TimestampMixin, UserOwnedMixin, UUIDPrimaryKeyMixin
from app.models.enums import BiomarkerFlag, ReportStatus

if TYPE_CHECKING:  # pragma: no cover
    from app.models.user import User


class BloodReport(Base, UUIDPrimaryKeyMixin, UserOwnedMixin, TimestampMixin):
    """A lab report uploaded by the user, plus its OCR output and AI summary.

    The raw extracted text is retained because OCR is imperfect: when a user
    disputes a parsed value, support needs to see exactly what the pipeline read.
    """

    __tablename__ = "blood_reports"
    __table_args__ = (Index("ix_report_user_created", "user_id", "created_at"),)

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    content_type: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default=ReportStatus.PENDING.value, nullable=False
    )

    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)

    user: Mapped["User"] = relationship(back_populates="reports")
    biomarkers: Mapped[list["Biomarker"]] = relationship(
        back_populates="report", cascade="all, delete-orphan", lazy="selectin"
    )


class Biomarker(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One parsed lab value, normalised to a canonical name and unit."""

    __tablename__ = "biomarkers"
    __table_args__ = (Index("ix_biomarker_report_name", "report_id", "name"),)

    report_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("blood_reports.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    #: Canonical key, e.g. "fasting_insulin".
    name: Mapped[str] = mapped_column(String(60), nullable=False)
    #: Human label as printed on the report, e.g. "Fasting Insulin".
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(30), nullable=False)

    reference_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    reference_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    flag: Mapped[str] = mapped_column(
        String(12), default=BiomarkerFlag.UNKNOWN.value, nullable=False
    )
    #: Plain-language explanation shown to the user.
    interpretation: Mapped[str | None] = mapped_column(Text, nullable=True)

    report: Mapped["BloodReport"] = relationship(back_populates="biomarkers")
