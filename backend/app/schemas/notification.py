"""Notification schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import NotificationType
from app.schemas.common import ORMModel


class NotificationCreate(BaseModel):
    type: NotificationType = NotificationType.SYSTEM
    title: str = Field(..., min_length=1, max_length=160)
    body: str = Field(..., min_length=1, max_length=2000)
    action_url: str | None = Field(None, max_length=300)
    icon: str = Field("bell", max_length=40)
    scheduled_for: datetime | None = None
    payload: dict = Field(default_factory=dict)


class NotificationResponse(ORMModel):
    id: uuid.UUID
    type: str
    title: str
    body: str
    action_url: str | None
    icon: str
    is_read: bool
    read_at: datetime | None
    scheduled_for: datetime | None
    payload: dict
    created_at: datetime


class NotificationCounts(BaseModel):
    total: int
    unread: int
