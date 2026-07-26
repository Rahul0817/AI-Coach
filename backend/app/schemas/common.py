"""Schema primitives reused across every API surface."""

from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

#: Appended to every medically-adjacent AI response. Oviora is an educational
#: product, not a diagnostic device, and the disclaimer is enforced in code
#: rather than left to prompt compliance.
MEDICAL_DISCLAIMER = (
    "**Important:** Oviora AI provides educational information and lifestyle "
    "guidance only. It does not diagnose, treat, or cure any medical condition. "
    "Please consult a qualified healthcare professional — such as a "
    "gynaecologist or endocrinologist — before making decisions about your "
    "health, medication, or treatment."
)


class ORMModel(BaseModel):
    """Base for response schemas read directly from SQLAlchemy objects."""

    model_config = ConfigDict(from_attributes=True)


class MessageResponse(BaseModel):
    """Generic acknowledgement body for actions with nothing else to return."""

    message: str = Field(..., examples=["Habit archived successfully."])
    success: bool = True


class ErrorResponse(BaseModel):
    """The single error shape every failing endpoint returns."""

    error: str = Field(..., examples=["not_found"])
    message: str = Field(..., examples=["The requested resource does not exist."])
    details: dict | None = None
    request_id: str | None = None


class PaginationMeta(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


class Page(BaseModel, Generic[T]):
    """Envelope for paginated collections."""

    items: list[T]
    meta: PaginationMeta

    @classmethod
    def build(cls, items: list[T], total: int, page: int, page_size: int) -> Page[T]:
        total_pages = max(1, -(-total // page_size))  # ceil division
        return cls(
            items=items,
            meta=PaginationMeta(
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
                has_next=page < total_pages,
                has_previous=page > 1,
            ),
        )


class PaginationParams(BaseModel):
    """Query parameters shared by list endpoints."""

    page: int = Field(1, ge=1, le=10_000)
    page_size: int = Field(20, ge=1, le=100)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class DateRange(BaseModel):
    """Inclusive analytics window."""

    start: datetime
    end: datetime


class HealthCheck(BaseModel):
    status: str
    version: str
    environment: str
    checks: dict[str, str]
