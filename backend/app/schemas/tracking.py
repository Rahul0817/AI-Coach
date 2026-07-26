"""Schemas for the daily trackers and the AI generators that feed them."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator

from app.models.enums import (
    DietaryPreference,
    HabitFrequency,
    Intensity,
    MealSource,
    MealType,
    MoodLabel,
    WorkoutType,
)
from app.schemas.common import ORMModel


def _not_future(v: date) -> date:
    if v > date.today():
        raise ValueError("Cannot log an entry for a future date.")
    return v


# -------------------------------------------------------------------- habits
class HabitCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(None, max_length=500)
    frequency: HabitFrequency = HabitFrequency.DAILY
    target_per_period: int = Field(1, ge=1, le=50)
    icon: str = Field("sparkles", max_length=40)
    colour: str = Field("#8b5cf6", pattern=r"^#(?:[0-9a-fA-F]{3}){1,2}$")


class HabitUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    description: str | None = Field(None, max_length=500)
    target_per_period: int | None = Field(None, ge=1, le=50)
    icon: str | None = Field(None, max_length=40)
    colour: str | None = Field(None, pattern=r"^#(?:[0-9a-fA-F]{3}){1,2}$")
    is_archived: bool | None = None


class HabitEntryCreate(BaseModel):
    logged_on: date = Field(default_factory=date.today)
    completed_count: int = Field(1, ge=0, le=50)
    note: str | None = Field(None, max_length=300)

    _check = field_validator("logged_on")(_not_future)


class HabitEntryResponse(ORMModel):
    id: uuid.UUID
    logged_on: date
    completed_count: int
    note: str | None


class HabitResponse(ORMModel):
    id: uuid.UUID
    name: str
    description: str | None
    frequency: str
    target_per_period: int
    icon: str
    colour: str
    is_archived: bool
    created_at: datetime
    #: Consecutive-day streak, computed by the service layer.
    current_streak: int = 0
    longest_streak: int = 0
    completed_today: bool = False
    completion_rate_30d: float = 0.0


# --------------------------------------------------------------------- meals
class MealCreate(BaseModel):
    logged_on: date = Field(default_factory=date.today)
    meal_type: MealType = MealType.SNACK
    name: str = Field(..., min_length=1, max_length=200)
    portion: str | None = Field(None, max_length=80)
    calories: float = Field(0, ge=0, le=10_000)
    protein_g: float = Field(0, ge=0, le=500)
    carbs_g: float = Field(0, ge=0, le=1000)
    fat_g: float = Field(0, ge=0, le=500)
    fibre_g: float = Field(0, ge=0, le=200)
    glycemic_index: int | None = Field(None, ge=0, le=110)
    source: MealSource = MealSource.MANUAL
    image_url: str | None = Field(None, max_length=500)
    notes: str | None = Field(None, max_length=1000)

    _check = field_validator("logged_on")(_not_future)


class MealResponse(ORMModel):
    id: uuid.UUID
    logged_on: date
    meal_type: str
    name: str
    portion: str | None
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fibre_g: float
    glycemic_index: int | None
    source: str
    image_url: str | None
    notes: str | None
    created_at: datetime


class NutritionSummary(BaseModel):
    logged_on: date
    total_calories: float
    total_protein_g: float
    total_carbs_g: float
    total_fat_g: float
    total_fibre_g: float
    meal_count: int
    calorie_goal: float
    protein_goal_g: float
    #: Share of calories from carbohydrates — the number a PCOS-aware plan
    #: cares most about, because of the insulin-resistance link.
    carb_percentage: float
    average_glycemic_index: float | None


# ------------------------------------------------------------------ workouts
class WorkoutCreate(BaseModel):
    logged_on: date = Field(default_factory=date.today)
    workout_type: WorkoutType = WorkoutType.WALKING
    title: str = Field(..., min_length=1, max_length=160)
    duration_minutes: int = Field(..., ge=1, le=600)
    intensity: Intensity = Intensity.MODERATE
    calories_burned: float | None = Field(None, ge=0, le=5000)
    notes: str | None = Field(None, max_length=1000)

    _check = field_validator("logged_on")(_not_future)


class WorkoutResponse(ORMModel):
    id: uuid.UUID
    logged_on: date
    workout_type: str
    title: str
    duration_minutes: int
    intensity: str
    calories_burned: float | None
    notes: str | None
    created_at: datetime


# --------------------------------------------------------- simple day-loggers
class WaterUpsert(BaseModel):
    logged_on: date = Field(default_factory=date.today)
    millilitres: int = Field(..., ge=0, le=15_000)
    goal_millilitres: int = Field(2500, ge=500, le=8000)

    _check = field_validator("logged_on")(_not_future)


class WaterResponse(ORMModel):
    id: uuid.UUID
    logged_on: date
    millilitres: int
    goal_millilitres: int
    goal_percentage: float = 0.0


class SleepUpsert(BaseModel):
    logged_on: date = Field(default_factory=date.today)
    hours: float = Field(..., ge=0, le=24)
    quality: int = Field(3, ge=1, le=5)
    bedtime: datetime | None = None
    wake_time: datetime | None = None
    notes: str | None = Field(None, max_length=500)

    _check = field_validator("logged_on")(_not_future)


class SleepResponse(ORMModel):
    id: uuid.UUID
    logged_on: date
    hours: float
    quality: int
    bedtime: datetime | None
    wake_time: datetime | None
    notes: str | None


class WeightUpsert(BaseModel):
    logged_on: date = Field(default_factory=date.today)
    weight_kg: float = Field(..., ge=25, le=350)
    body_fat_percentage: float | None = Field(None, ge=3, le=70)
    waist_cm: float | None = Field(None, ge=40, le=200)

    _check = field_validator("logged_on")(_not_future)


class WeightResponse(ORMModel):
    id: uuid.UUID
    logged_on: date
    weight_kg: float
    body_fat_percentage: float | None
    waist_cm: float | None
    bmi: float | None = None


class MoodUpsert(BaseModel):
    logged_on: date = Field(default_factory=date.today)
    mood: MoodLabel = MoodLabel.OKAY
    energy_level: int = Field(3, ge=1, le=5)
    stress_level: int = Field(3, ge=1, le=5)
    journal: str | None = Field(None, max_length=4000)

    _check = field_validator("logged_on")(_not_future)


class MoodResponse(ORMModel):
    id: uuid.UUID
    logged_on: date
    mood: str
    energy_level: int
    stress_level: int
    journal: str | None
    mood_score: int = 3


# -------------------------------------------------------- AI plan generators
class WorkoutPlanRequest(BaseModel):
    days_per_week: int = Field(4, ge=1, le=7)
    minutes_per_session: int = Field(35, ge=10, le=120)
    intensity: Intensity = Intensity.MODERATE
    focus: str | None = Field(
        None, max_length=200,
        examples=["insulin sensitivity and strength"],
    )
    equipment: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class PlannedWorkout(BaseModel):
    day: str
    workout_type: WorkoutType
    title: str
    duration_minutes: int
    intensity: Intensity
    exercises: list[str]
    rationale: str


class WorkoutPlanResponse(BaseModel):
    plan_name: str
    days_per_week: int
    weekly_schedule: list[PlannedWorkout]
    guidance: str
    disclaimer: str


class DietPlanRequest(BaseModel):
    target_calories: int | None = Field(None, ge=1000, le=4000)
    dietary_preference: DietaryPreference | None = None
    allergies: list[str] = Field(default_factory=list)
    cuisine: str | None = Field(None, max_length=80, examples=["North Indian"])
    days: int = Field(1, ge=1, le=7)


class PlannedMeal(BaseModel):
    meal_type: MealType
    name: str
    description: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fibre_g: float
    glycemic_index: int
    pcos_benefit: str


class PlannedDay(BaseModel):
    day: str
    meals: list[PlannedMeal]
    total_calories: float
    total_protein_g: float


class DietPlanResponse(BaseModel):
    plan_name: str
    days: list[PlannedDay]
    principles: list[str]
    guidance: str
    disclaimer: str
