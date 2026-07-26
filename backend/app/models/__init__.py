"""Importing this package registers every table on ``Base.metadata``.

Alembic autogenerate and ``init_models()`` both rely on that side effect, so
the explicit re-export list below is load-bearing, not cosmetic.
"""

from app.models.chat import Conversation, Message
from app.models.health import CycleLog, Prediction, SymptomLog
from app.models.notification import Notification
from app.models.report import Biomarker, BloodReport
from app.models.tracking import (
    Habit,
    HabitEntry,
    MealLog,
    MoodLog,
    SleepLog,
    WaterLog,
    WeightLog,
    WorkoutLog,
)
from app.models.user import Profile, User

__all__ = [
    "Biomarker",
    "BloodReport",
    "Conversation",
    "CycleLog",
    "Habit",
    "HabitEntry",
    "MealLog",
    "Message",
    "MoodLog",
    "Notification",
    "Prediction",
    "Profile",
    "SleepLog",
    "SymptomLog",
    "User",
    "WaterLog",
    "WeightLog",
    "WorkoutLog",
]
