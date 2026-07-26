"""Integration tests against a real database.

These run against SQLite rather than mocks, because the layer most likely to
break — queries, constraints, cascade behaviour — is exactly the layer a mock
would replace with a no-op.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import NotFoundError
from app.models.user import User
from app.repositories.chat import ConversationRepository, MessageRepository
from app.repositories.health import CycleRepository, SymptomRepository
from app.repositories.tracking import MealRepository, WaterRepository
from app.repositories.user import ProfileRepository, UserRepository

pytestmark = pytest.mark.integration


async def _make_user(session, email: str = "a@example.com") -> User:
    users = UserRepository(session)
    user = await users.create(
        email=email, hashed_password="hashed", full_name="Test User"
    )
    await session.commit()
    return user


class TestUserRepository:
    async def test_email_lookup_is_case_insensitive(self, session) -> None:
        await _make_user(session, "Asha@Example.com".lower())
        users = UserRepository(session)
        assert await users.get_by_email("ASHA@EXAMPLE.COM") is not None

    async def test_unique_email_enforced_by_the_database(self, session) -> None:
        """The constraint must exist in the schema, not only in service code —
        a race between two concurrent registrations would slip past a
        code-level check."""
        await _make_user(session, "dup@example.com")
        users = UserRepository(session)
        with pytest.raises(IntegrityError):
            await users.create(
                email="dup@example.com", hashed_password="x", full_name="Other"
            )
        await session.rollback()

    async def test_lockout_after_repeated_failures(self, session) -> None:
        user = await _make_user(session)
        users = UserRepository(session)
        for _ in range(5):
            await users.record_failed_login(user)
        assert users.is_locked(user)

    async def test_successful_login_clears_the_lock(self, session) -> None:
        user = await _make_user(session)
        users = UserRepository(session)
        for _ in range(5):
            await users.record_failed_login(user)
        await users.record_successful_login(user)
        assert not users.is_locked(user)
        assert user.failed_login_count == 0


class TestOwnershipEnforcement:
    async def test_cannot_read_another_users_row(self, session) -> None:
        """This is the IDOR guard. Ownership is checked in SQL, so a handler
        that forgets to compare user_id still cannot leak data."""
        owner = await _make_user(session, "owner@example.com")
        intruder = await _make_user(session, "intruder@example.com")

        cycles = CycleRepository(session)
        cycle = await cycles.create(user_id=owner.id, start_date=date(2026, 1, 1))
        await session.commit()

        assert await cycles.get_for_user(cycle.id, owner.id) is not None
        with pytest.raises(NotFoundError):
            await cycles.get_for_user(cycle.id, intruder.id)

    async def test_missing_row_raises_not_found(self, session) -> None:
        user = await _make_user(session)
        with pytest.raises(NotFoundError):
            await CycleRepository(session).get_for_user(uuid.uuid4(), user.id)


class TestCascadeDeletion:
    async def test_deleting_a_user_removes_their_health_data(self, session) -> None:
        """Right-to-erasure is enforced at the database level, so no
        application bug can leave orphaned health records behind."""
        user = await _make_user(session)
        cycles = CycleRepository(session)
        symptoms = SymptomRepository(session)

        await cycles.create(user_id=user.id, start_date=date(2026, 1, 1))
        await symptoms.create(
            user_id=user.id, logged_on=date(2026, 1, 2), symptom="fatigue", severity=3
        )
        await session.commit()

        await UserRepository(session).delete(user)
        await session.commit()

        assert await cycles.count_for_user(user.id) == 0
        assert await symptoms.count_for_user(user.id) == 0


class TestUpsertSemantics:
    async def test_water_is_one_row_per_day(self, session) -> None:
        user = await _make_user(session)
        water = WaterRepository(session)
        today = date.today()

        await water.upsert(user.id, today, millilitres=1000, goal_millilitres=2500)
        await water.upsert(user.id, today, millilitres=1800, goal_millilitres=2500)
        await session.commit()

        rows = await water.list_between(user.id, today, today)
        assert len(rows) == 1
        assert rows[0].millilitres == 1800

    async def test_symptom_upsert_updates_severity(self, session) -> None:
        user = await _make_user(session)
        symptoms = SymptomRepository(session)
        today = date.today()

        await symptoms.upsert(user.id, today, "fatigue", 2, None)
        await symptoms.upsert(user.id, today, "fatigue", 5, "much worse")
        await session.commit()

        rows = await symptoms.list_between(user.id, today, today)
        assert len(rows) == 1
        assert rows[0].severity == 5


class TestAggregationQueries:
    async def test_daily_macro_totals_group_correctly(self, session) -> None:
        user = await _make_user(session)
        meals = MealRepository(session)
        today = date.today()

        for calories, protein in [(300, 20), (500, 30), (200, 10)]:
            await meals.create(
                user_id=user.id, logged_on=today, meal_type="lunch",
                name="Meal", calories=calories, protein_g=protein,
                carbs_g=40, fat_g=10, fibre_g=5,
            )
        await meals.create(
            user_id=user.id, logged_on=today - timedelta(days=1),
            meal_type="dinner", name="Yesterday", calories=999, protein_g=1,
            carbs_g=1, fat_g=1, fibre_g=1,
        )
        await session.commit()

        totals = await meals.daily_totals(user.id, today, today)
        assert len(totals) == 1
        assert totals[0]["calories"] == 1000
        assert totals[0]["protein_g"] == 60
        assert totals[0]["meals"] == 3

    async def test_symptom_frequency_summary(self, session) -> None:
        user = await _make_user(session)
        symptoms = SymptomRepository(session)
        start = date.today() - timedelta(days=10)

        for offset, severity in enumerate([2, 4, 3]):
            await symptoms.create(
                user_id=user.id, logged_on=start + timedelta(days=offset),
                symptom="fatigue", severity=severity,
            )
        await session.commit()

        summary = await symptoms.frequency_summary(user.id, start, date.today())
        fatigue = next(row for row in summary if row["symptom"] == "fatigue")
        assert fatigue["occurrences"] == 3
        assert fatigue["average_severity"] == pytest.approx(3.0)


class TestConversationRepository:
    async def test_sequence_is_monotonic_and_survives_deletion(self, session) -> None:
        """Sequence is derived from MAX, not COUNT, so deleting a message
        cannot cause a later collision."""
        user = await _make_user(session)
        conversations = ConversationRepository(session)
        messages = MessageRepository(session)

        conversation = await conversations.create(user_id=user.id)
        await session.commit()

        created = []
        for index in range(3):
            sequence = await messages.next_sequence(conversation.id)
            created.append(
                await messages.create(
                    conversation_id=conversation.id, sequence=sequence,
                    role="user", content=f"message {index}",
                )
            )
        await session.commit()
        assert [m.sequence for m in created] == [1, 2, 3]

        await messages.delete(created[1])
        await session.commit()
        assert await messages.next_sequence(conversation.id) == 4

    async def test_recent_window_returns_oldest_first(self, session) -> None:
        """The prompt needs chronological order even though the index scans
        descending."""
        user = await _make_user(session)
        conversation = await ConversationRepository(session).create(user_id=user.id)
        messages = MessageRepository(session)
        await session.commit()

        for index in range(6):
            await messages.create(
                conversation_id=conversation.id, sequence=index + 1,
                role="user", content=f"m{index}",
            )
        await session.commit()

        window = await messages.recent_window(conversation.id, 3)
        assert [m.content for m in window] == ["m3", "m4", "m5"]

    async def test_pinned_conversations_sort_first(self, session) -> None:
        user = await _make_user(session)
        conversations = ConversationRepository(session)

        await conversations.create(user_id=user.id, title="ordinary")
        pinned = await conversations.create(user_id=user.id, title="pinned")
        pinned.is_pinned = True
        await session.commit()

        rows = await conversations.list_for_sidebar(user.id)
        assert rows[0].title == "pinned"


class TestProfileMemory:
    async def test_ai_memory_merge_persists(self, session) -> None:
        """JSONB change detection does not see in-place dict mutation, so the
        repository must assign a fresh dict — this pins that behaviour."""
        user = await _make_user(session)
        profiles = ProfileRepository(session)
        profile = await profiles.get_or_create(user.id)
        await session.commit()

        await profiles.merge_ai_memory(profile, {"age": 22})
        await session.commit()
        await profiles.merge_ai_memory(profile, {"prefers": "vegetarian"})
        await session.commit()

        session.expunge_all()
        reloaded = await ProfileRepository(session).get_by_user(user.id)
        assert reloaded is not None
        assert reloaded.ai_memory == {"age": 22, "prefers": "vegetarian"}
