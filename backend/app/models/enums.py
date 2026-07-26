"""Domain enumerations shared by ORM models and Pydantic schemas.

These are plain ``str`` enums stored as VARCHAR with a CHECK-style application
constraint rather than native Postgres ENUM types. Native enums require a
migration (and an exclusive table lock) to add a value; string enums let the
product add, say, a new mood label without a schema change.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """Serialises as its value in JSON while remaining comparable to ``str``."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class ActivityLevel(StrEnum):
    SEDENTARY = "sedentary"
    LIGHT = "light"
    MODERATE = "moderate"
    ACTIVE = "active"
    VERY_ACTIVE = "very_active"


class DiagnosisStatus(StrEnum):
    """Self-reported clinical context — never inferred by the app."""

    UNDIAGNOSED = "undiagnosed"
    SUSPECTED = "suspected"
    DIAGNOSED = "diagnosed"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class DietaryPreference(StrEnum):
    OMNIVORE = "omnivore"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    EGGETARIAN = "eggetarian"
    PESCATARIAN = "pescatarian"


class FlowIntensity(StrEnum):
    SPOTTING = "spotting"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class SymptomType(StrEnum):
    ACNE = "acne"
    HAIR_LOSS = "hair_loss"
    HIRSUTISM = "hirsutism"
    WEIGHT_GAIN = "weight_gain"
    FATIGUE = "fatigue"
    MOOD_SWINGS = "mood_swings"
    BLOATING = "bloating"
    PELVIC_PAIN = "pelvic_pain"
    IRREGULAR_PERIODS = "irregular_periods"
    SUGAR_CRAVINGS = "sugar_cravings"
    SLEEP_DISTURBANCE = "sleep_disturbance"
    HEADACHE = "headache"


class RiskBand(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class MealType(StrEnum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class MealSource(StrEnum):
    """How a meal entry entered the system — used for analytics confidence."""

    MANUAL = "manual"
    IMAGE_ANALYSIS = "image_analysis"
    AI_PLAN = "ai_plan"


class WorkoutType(StrEnum):
    STRENGTH = "strength"
    CARDIO = "cardio"
    HIIT = "hiit"
    YOGA = "yoga"
    WALKING = "walking"
    PILATES = "pilates"
    MOBILITY = "mobility"
    REST = "rest"


class Intensity(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class HabitFrequency(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"


class MoodLabel(StrEnum):
    GREAT = "great"
    GOOD = "good"
    OKAY = "okay"
    LOW = "low"
    TERRIBLE = "terrible"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AgentName(StrEnum):
    """The specialist agents the router can dispatch to."""

    ROUTER = "router"
    HEALTH_EXPERT = "health_expert"
    NUTRITION_COACH = "nutrition_coach"
    FITNESS_COACH = "fitness_coach"
    MENTAL_WELLNESS_COACH = "mental_wellness_coach"
    BLOOD_REPORT_ANALYZER = "blood_report_analyzer"
    FOOD_ANALYZER = "food_analyzer"
    HABIT_COACH = "habit_coach"
    CYCLE_TRACKER_ASSISTANT = "cycle_tracker_assistant"


class ReportStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class NotificationType(StrEnum):
    REMINDER = "reminder"
    INSIGHT = "insight"
    ACHIEVEMENT = "achievement"
    SYSTEM = "system"


class BiomarkerFlag(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    UNKNOWN = "unknown"
