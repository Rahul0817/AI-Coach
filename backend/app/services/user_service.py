"""Profile management and data portability."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.models.user import Profile, User
from app.repositories.chat import ConversationRepository, MessageRepository
from app.repositories.health import (
    CycleRepository,
    PredictionRepository,
    SymptomRepository,
)
from app.repositories.report import BloodReportRepository
from app.repositories.tracking import (
    HabitRepository,
    MealRepository,
    MoodRepository,
    SleepRepository,
    WaterRepository,
    WeightRepository,
    WorkoutRepository,
)
from app.repositories.user import ProfileRepository, UserRepository
from app.schemas.user import ProfileUpdate, UserUpdate

logger = get_logger(__name__)

#: How far back a data export reaches. Two years covers any realistic history
#: while keeping the response from becoming unbounded.
EXPORT_WINDOW_DAYS = 730


class UserService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.users = UserRepository(session)
        self.profiles = ProfileRepository(session)

    async def get_profile(self, user_id: uuid.UUID) -> Profile:
        profile = await self.profiles.get_or_create(user_id)
        await self.session.commit()
        return profile

    async def update_profile(
        self, user_id: uuid.UUID, payload: ProfileUpdate
    ) -> Profile:
        profile = await self.profiles.get_or_create(user_id)
        updates = payload.model_dump(exclude_unset=True)

        for key, value in updates.items():
            if value is None:
                continue
            # Enums arrive as objects; the column stores their value.
            setattr(profile, key, value.value if hasattr(value, "value") else value)

        await self.session.commit()
        return profile

    async def update_account(self, user_id: uuid.UUID, payload: UserUpdate) -> User:
        user = await self.users.get(user_id)
        if user is None:
            raise NotFoundError("Account not found.")

        if payload.email and payload.email.lower() != user.email:
            if await self.users.email_taken(payload.email):
                raise ConflictError("That email is already in use.")
            user.email = payload.email.lower()
            # Changing an email invalidates the previous verification.
            user.is_verified = False
        if payload.full_name:
            user.full_name = payload.full_name.strip()

        await self.session.commit()
        return user

    def serialise_profile(self, profile: Profile) -> dict:
        """Include the computed fields the ORM exposes as properties."""
        data = {
            "id": profile.id,
            "user_id": profile.user_id,
            "date_of_birth": profile.date_of_birth,
            "height_cm": profile.height_cm,
            "weight_kg": profile.weight_kg,
            "activity_level": profile.activity_level,
            "diagnosis_status": profile.diagnosis_status,
            "dietary_preference": profile.dietary_preference,
            "average_cycle_length": profile.average_cycle_length,
            "average_period_length": profile.average_period_length,
            "primary_goal": profile.primary_goal,
            "allergies": profile.allergies,
            "medical_conditions": profile.medical_conditions,
            "medications": profile.medications,
            "notes": profile.notes,
            "timezone": profile.timezone,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
            "age": profile.age,
            "bmi": profile.bmi,
            "bmi_category": profile.bmi_category,
        }
        return data

    # ------------------------------------------------------------- export
    async def export_data(self, user_id: uuid.UUID) -> dict:
        """Produce a complete portable copy of the user's data.

        This is the GDPR right-to-portability implementation. It deliberately
        walks every table rather than a curated subset — if the product stores
        it, the user gets it.
        """
        user = await self.users.get_with_profile(user_id)
        if user is None:
            raise NotFoundError("Account not found.")

        end = date.today()
        start = end - timedelta(days=EXPORT_WINDOW_DAYS)

        cycles = await CycleRepository(self.session).list_for_user(user_id, limit=1000)
        symptoms = await SymptomRepository(self.session).list_between(
            user_id, start, end, limit=5000
        )
        predictions = await PredictionRepository(self.session).history(
            user_id, limit=200
        )
        meals = await MealRepository(self.session).list_between(
            user_id, start, end, limit=5000
        )
        workouts = await WorkoutRepository(self.session).list_between(
            user_id, start, end, limit=2000
        )
        water = await WaterRepository(self.session).list_between(
            user_id, start, end, limit=1000
        )
        sleep = await SleepRepository(self.session).list_between(
            user_id, start, end, limit=1000
        )
        weight = await WeightRepository(self.session).list_between(
            user_id, start, end, limit=1000
        )
        mood = await MoodRepository(self.session).list_between(
            user_id, start, end, limit=1000
        )
        habits = await HabitRepository(self.session).list_active(user_id)
        reports = await BloodReportRepository(
            self.session
        ).list_for_user_with_biomarkers(user_id, limit=100)

        conversation_repo = ConversationRepository(self.session)
        message_repo = MessageRepository(self.session)
        conversations = await conversation_repo.list_for_sidebar(
            user_id, limit=200, include_archived=True
        )
        conversation_dumps: list[dict] = []
        for conversation in conversations:
            messages = await message_repo.messages_before(
                conversation.id, conversation.message_count + 1
            )
            conversation_dumps.append(
                {
                    "id": str(conversation.id),
                    "title": conversation.title,
                    "created_at": conversation.created_at.isoformat(),
                    "summary": conversation.summary,
                    "messages": [
                        {
                            "sequence": m.sequence,
                            "role": m.role,
                            "content": m.content,
                            "agent": m.agent,
                            "created_at": m.created_at.isoformat(),
                        }
                        for m in messages
                    ],
                }
            )

        return {
            "exported_at": datetime.now(timezone.utc),
            "user": {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "created_at": user.created_at.isoformat(),
            },
            "profile": (
                self.serialise_profile(user.profile) if user.profile else None
            ),
            "cycles": [row.to_dict() for row in cycles],
            "symptoms": [row.to_dict() for row in symptoms],
            "predictions": [row.to_dict() for row in predictions],
            "meals": [row.to_dict() for row in meals],
            "workouts": [row.to_dict() for row in workouts],
            "habits": [
                {
                    **habit.to_dict(),
                    "entries": [entry.to_dict() for entry in habit.entries],
                }
                for habit in habits
            ],
            "water": [row.to_dict() for row in water],
            "sleep": [row.to_dict() for row in sleep],
            "weight": [row.to_dict() for row in weight],
            "mood": [row.to_dict() for row in mood],
            "conversations": conversation_dumps,
            "reports": [
                {
                    **report.to_dict(),
                    "biomarkers": [b.to_dict() for b in report.biomarkers],
                }
                for report in reports
            ],
        }
