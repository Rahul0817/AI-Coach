"""Data-access layer. Services depend on these, never on raw SQLAlchemy."""

from app.repositories.base import BaseRepository
from app.repositories.chat import ConversationRepository, MessageRepository
from app.repositories.health import (
    CycleRepository,
    PredictionRepository,
    SymptomRepository,
)
from app.repositories.notification import NotificationRepository
from app.repositories.report import BiomarkerRepository, BloodReportRepository
from app.repositories.tracking import (
    HabitEntryRepository,
    HabitRepository,
    MealRepository,
    MoodRepository,
    SleepRepository,
    WaterRepository,
    WeightRepository,
    WorkoutRepository,
)
from app.repositories.user import ProfileRepository, UserRepository

__all__ = [
    "BaseRepository",
    "BiomarkerRepository",
    "BloodReportRepository",
    "ConversationRepository",
    "CycleRepository",
    "HabitEntryRepository",
    "HabitRepository",
    "MealRepository",
    "MessageRepository",
    "MoodRepository",
    "NotificationRepository",
    "PredictionRepository",
    "ProfileRepository",
    "SleepRepository",
    "SymptomRepository",
    "UserRepository",
    "WaterRepository",
    "WeightRepository",
    "WorkoutRepository",
]
