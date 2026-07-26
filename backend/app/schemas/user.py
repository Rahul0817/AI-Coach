"""User and profile schemas."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, EmailStr, Field, computed_field, field_validator

from app.models.enums import ActivityLevel, DiagnosisStatus, DietaryPreference
from app.schemas.common import ORMModel


class ProfileUpdate(BaseModel):
    """Partial profile update. Every field is optional — PATCH semantics."""

    date_of_birth: date | None = None
    height_cm: float | None = Field(None, ge=90, le=250)
    weight_kg: float | None = Field(None, ge=25, le=350)
    activity_level: ActivityLevel | None = None
    diagnosis_status: DiagnosisStatus | None = None
    dietary_preference: DietaryPreference | None = None
    average_cycle_length: int | None = Field(None, ge=15, le=120)
    average_period_length: int | None = Field(None, ge=1, le=21)
    primary_goal: str | None = Field(None, max_length=200)
    allergies: list[str] | None = None
    medical_conditions: list[str] | None = None
    medications: list[str] | None = None
    notes: str | None = Field(None, max_length=2000)
    timezone: str | None = Field(None, max_length=64)

    @field_validator("date_of_birth")
    @classmethod
    def _plausible_dob(cls, v: date | None) -> date | None:
        if v is None:
            return v
        today = date.today()
        age = today.year - v.year - ((today.month, today.day) < (v.month, v.day))
        if not 10 <= age <= 100:
            raise ValueError("Date of birth implies an implausible age (10–100).")
        return v

    @field_validator("allergies", "medical_conditions", "medications")
    @classmethod
    def _clean_list(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if len(v) > 40:
            raise ValueError("At most 40 entries are allowed.")
        cleaned = [item.strip() for item in v if item and item.strip()]
        for item in cleaned:
            if len(item) > 120:
                raise ValueError("Each entry must be 120 characters or fewer.")
        return cleaned


class ProfileResponse(ORMModel):
    id: uuid.UUID
    user_id: uuid.UUID
    date_of_birth: date | None
    height_cm: float | None
    weight_kg: float | None
    activity_level: str
    diagnosis_status: str
    dietary_preference: str
    average_cycle_length: int | None
    average_period_length: int | None
    primary_goal: str | None
    allergies: list[str]
    medical_conditions: list[str]
    medications: list[str]
    notes: str | None
    timezone: str
    created_at: datetime
    updated_at: datetime

    # Derived values are exposed to the client so BMI logic lives in exactly
    # one place instead of being reimplemented in TypeScript.
    age: int | None = None
    bmi: float | None = None
    bmi_category: str | None = None


class UserResponse(ORMModel):
    id: uuid.UUID
    email: EmailStr
    full_name: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login_at: datetime | None = None


class UserWithProfile(UserResponse):
    profile: ProfileResponse | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def onboarding_complete(self) -> bool:
        """Drives the "finish your profile" prompt on the dashboard."""
        p = self.profile
        return bool(p and p.date_of_birth and p.height_cm and p.weight_kg)


class UserUpdate(BaseModel):
    full_name: str | None = Field(None, min_length=2, max_length=120)
    email: EmailStr | None = None


class DataExport(BaseModel):
    """Complete portable copy of a user's data (GDPR right to portability)."""

    exported_at: datetime
    user: dict
    profile: dict | None
    cycles: list[dict]
    symptoms: list[dict]
    predictions: list[dict]
    meals: list[dict]
    workouts: list[dict]
    habits: list[dict]
    water: list[dict]
    sleep: list[dict]
    weight: list[dict]
    mood: list[dict]
    conversations: list[dict]
    reports: list[dict]
