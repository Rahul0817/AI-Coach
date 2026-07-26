"""Daily trackers: habits, meals, workouts, water, sleep, weight, mood."""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from itertools import pairwise

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError
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
from app.repositories.user import ProfileRepository
from app.schemas.tracking import (
    HabitCreate,
    HabitEntryCreate,
    HabitUpdate,
    MealCreate,
    MoodUpsert,
    NutritionSummary,
    SleepUpsert,
    WaterUpsert,
    WeightUpsert,
    WorkoutCreate,
)

#: MET values (metabolic equivalents) per workout type at moderate intensity,
#: used to estimate energy expenditure when the user does not supply a figure.
MET_VALUES = {
    "strength": 5.0,
    "cardio": 7.0,
    "hiit": 9.0,
    "yoga": 3.0,
    "walking": 3.5,
    "pilates": 3.5,
    "mobility": 2.5,
    "rest": 1.0,
}
INTENSITY_MULTIPLIER = {"low": 0.75, "moderate": 1.0, "high": 1.3}


class HabitService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.habits = HabitRepository(session)
        self.entries = HabitEntryRepository(session)

    async def create(self, user_id: uuid.UUID, payload: HabitCreate) -> Habit:
        if await self.habits.exists(user_id=user_id, name=payload.name.strip()):
            raise ConflictError(f"You already have a habit called '{payload.name}'.")
        habit = await self.habits.create(
            user_id=user_id,
            name=payload.name.strip(),
            description=payload.description,
            frequency=payload.frequency.value,
            target_per_period=payload.target_per_period,
            icon=payload.icon,
            colour=payload.colour,
        )
        await self.session.commit()
        return habit

    async def update(
        self, user_id: uuid.UUID, habit_id: uuid.UUID, payload: HabitUpdate
    ) -> Habit:
        habit = await self.habits.get_for_user(habit_id, user_id)
        updates = payload.model_dump(exclude_unset=True)
        # is_archived is a genuine boolean toggle, so False must be honoured
        # rather than treated as "unset" by the PATCH-style update helper.
        archived = updates.pop("is_archived", None)
        await self.habits.update(habit, **updates)
        if archived is not None:
            habit.is_archived = archived
            await self.session.flush()
        await self.session.commit()
        return habit

    async def delete(self, user_id: uuid.UUID, habit_id: uuid.UUID) -> None:
        await self.habits.delete_for_user(habit_id, user_id)
        await self.session.commit()

    async def check_in(
        self, user_id: uuid.UUID, habit_id: uuid.UUID, payload: HabitEntryCreate
    ) -> HabitEntry:
        await self.habits.get_for_user(habit_id, user_id)  # ownership check
        if payload.completed_count == 0:
            # Zero means "undo today's tick" rather than "log a zero".
            await self.entries.delete_for_day(habit_id, payload.logged_on)
            await self.session.commit()
            return HabitEntry(
                habit_id=habit_id, logged_on=payload.logged_on, completed_count=0
            )
        entry = await self.entries.upsert(
            habit_id, payload.logged_on, payload.completed_count, payload.note
        )
        await self.session.commit()
        return entry

    async def list_with_stats(self, user_id: uuid.UUID) -> list[dict]:
        """Habits enriched with streaks and completion rate."""
        habits = await self.habits.list_active(user_id)
        today = date.today()
        out: list[dict] = []

        for habit in habits:
            completed_days = {
                entry.logged_on
                for entry in habit.entries
                if entry.completed_count >= habit.target_per_period
            }
            current = self._current_streak(completed_days, today)
            longest = self._longest_streak(completed_days)
            window_start = today - timedelta(days=29)
            in_window = sum(1 for day in completed_days if day >= window_start)

            out.append(
                {
                    "id": habit.id,
                    "name": habit.name,
                    "description": habit.description,
                    "frequency": habit.frequency,
                    "target_per_period": habit.target_per_period,
                    "icon": habit.icon,
                    "colour": habit.colour,
                    "is_archived": habit.is_archived,
                    "created_at": habit.created_at,
                    "current_streak": current,
                    "longest_streak": longest,
                    "completed_today": today in completed_days,
                    "completion_rate_30d": round(in_window / 30 * 100, 1),
                }
            )
        return out

    @staticmethod
    def _current_streak(days: set[date], today: date) -> int:
        """Consecutive completed days ending today or yesterday.

        Yesterday counts as an unbroken streak because the day is not over —
        resetting someone's 40-day streak at midnight before they have had a
        chance to log is exactly the behaviour that makes people abandon a
        habit tracker.
        """
        if not days:
            return 0
        cursor = today if today in days else today - timedelta(days=1)
        if cursor not in days:
            return 0
        streak = 0
        while cursor in days:
            streak += 1
            cursor -= timedelta(days=1)
        return streak

    @staticmethod
    def _longest_streak(days: set[date]) -> int:
        if not days:
            return 0
        ordered = sorted(days)
        longest = run = 1
        for previous, current in pairwise(ordered):
            run = run + 1 if (current - previous).days == 1 else 1
            longest = max(longest, run)
        return longest


class MealService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.meals = MealRepository(session)
        self.profiles = ProfileRepository(session)

    async def log(self, user_id: uuid.UUID, payload: MealCreate) -> MealLog:
        meal = await self.meals.create(
            user_id=user_id,
            logged_on=payload.logged_on,
            meal_type=payload.meal_type.value,
            name=payload.name,
            portion=payload.portion,
            calories=payload.calories,
            protein_g=payload.protein_g,
            carbs_g=payload.carbs_g,
            fat_g=payload.fat_g,
            fibre_g=payload.fibre_g,
            glycemic_index=payload.glycemic_index,
            source=payload.source.value,
            image_url=payload.image_url,
            notes=payload.notes,
        )
        await self.session.commit()
        return meal

    async def for_day(self, user_id: uuid.UUID, day: date) -> list[MealLog]:
        return await self.meals.for_day(user_id, day)

    async def delete(self, user_id: uuid.UUID, meal_id: uuid.UUID) -> None:
        await self.meals.delete_for_user(meal_id, user_id)
        await self.session.commit()

    async def daily_summary(self, user_id: uuid.UUID, day: date) -> NutritionSummary:
        meals = await self.meals.for_day(user_id, day)
        profile = await self.profiles.get_by_user(user_id)

        calories = sum(m.calories for m in meals)
        protein = sum(m.protein_g for m in meals)
        carbs = sum(m.carbs_g for m in meals)
        fat = sum(m.fat_g for m in meals)
        fibre = sum(m.fibre_g for m in meals)

        calorie_goal, protein_goal = self._goals(profile)

        # Carbohydrate provides 4 kcal per gram.
        carb_percentage = round(carbs * 4 / calories * 100, 1) if calories > 0 else 0.0

        gi_values = [
            (m.glycemic_index, m.carbs_g)
            for m in meals
            if m.glycemic_index is not None and m.carbs_g > 0
        ]
        carb_weight = sum(c for _, c in gi_values)
        average_gi = (
            round(sum(gi * c for gi, c in gi_values) / carb_weight, 1)
            if carb_weight > 0
            else None
        )

        return NutritionSummary(
            logged_on=day,
            total_calories=round(calories, 1),
            total_protein_g=round(protein, 1),
            total_carbs_g=round(carbs, 1),
            total_fat_g=round(fat, 1),
            total_fibre_g=round(fibre, 1),
            meal_count=len(meals),
            calorie_goal=calorie_goal,
            protein_goal_g=protein_goal,
            carb_percentage=carb_percentage,
            average_glycemic_index=average_gi,
        )

    @staticmethod
    def _goals(profile) -> tuple[float, float]:  # type: ignore[no-untyped-def]
        """Derive calorie and protein targets from the profile.

        Uses Mifflin-St Jeor for BMR — the equation with the best accuracy in
        validation studies — then applies an activity factor. Protein is set at
        the upper end of the general range because higher protein supports
        satiety and lean-mass retention, both of which matter here.
        """
        if profile is None or not profile.weight_kg or not profile.height_cm:
            return 1800.0, 90.0

        age = profile.age or 28
        # Female coefficient; Oviora's user base is women with PCOS.
        bmr = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * age - 161
        factors = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9,
        }
        maintenance = bmr * factors.get(profile.activity_level, 1.55)
        protein_goal = round(profile.weight_kg * 1.4, 1)
        return round(maintenance, 0), protein_goal


class WorkoutService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.workouts = WorkoutRepository(session)
        self.profiles = ProfileRepository(session)

    async def log(self, user_id: uuid.UUID, payload: WorkoutCreate) -> WorkoutLog:
        calories = payload.calories_burned
        if calories is None:
            calories = await self._estimate_calories(
                user_id,
                payload.workout_type.value,
                payload.intensity.value,
                payload.duration_minutes,
            )

        workout = await self.workouts.create(
            user_id=user_id,
            logged_on=payload.logged_on,
            workout_type=payload.workout_type.value,
            title=payload.title,
            duration_minutes=payload.duration_minutes,
            intensity=payload.intensity.value,
            calories_burned=calories,
            notes=payload.notes,
        )
        await self.session.commit()
        return workout

    async def _estimate_calories(
        self, user_id: uuid.UUID, workout_type: str, intensity: str, minutes: int
    ) -> float:
        """MET-based estimate: kcal = MET × 3.5 × kg / 200 × minutes."""
        profile = await self.profiles.get_by_user(user_id)
        weight = profile.weight_kg if profile and profile.weight_kg else 65.0
        met = MET_VALUES.get(workout_type, 4.0) * INTENSITY_MULTIPLIER.get(intensity, 1.0)
        return round(met * 3.5 * weight / 200 * minutes, 1)

    async def list_range(
        self, user_id: uuid.UUID, start: date, end: date
    ) -> list[WorkoutLog]:
        return await self.workouts.list_between(user_id, start, end)

    async def delete(self, user_id: uuid.UUID, workout_id: uuid.UUID) -> None:
        await self.workouts.delete_for_user(workout_id, user_id)
        await self.session.commit()


class DailyTrackerService:
    """Water, sleep, weight and mood — all one-row-per-day upserts."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.water = WaterRepository(session)
        self.sleep = SleepRepository(session)
        self.weight = WeightRepository(session)
        self.mood = MoodRepository(session)
        self.profiles = ProfileRepository(session)

    async def upsert_water(self, user_id: uuid.UUID, payload: WaterUpsert) -> WaterLog:
        row = await self.water.upsert(
            user_id,
            payload.logged_on,
            millilitres=payload.millilitres,
            goal_millilitres=payload.goal_millilitres,
        )
        await self.session.commit()
        return row

    async def upsert_sleep(self, user_id: uuid.UUID, payload: SleepUpsert) -> SleepLog:
        row = await self.sleep.upsert(
            user_id,
            payload.logged_on,
            hours=payload.hours,
            quality=payload.quality,
            bedtime=payload.bedtime,
            wake_time=payload.wake_time,
            notes=payload.notes,
        )
        await self.session.commit()
        return row

    async def upsert_weight(self, user_id: uuid.UUID, payload: WeightUpsert) -> WeightLog:
        row = await self.weight.upsert(
            user_id,
            payload.logged_on,
            weight_kg=payload.weight_kg,
            body_fat_percentage=payload.body_fat_percentage,
            waist_cm=payload.waist_cm,
        )
        # Mirror the most recent weight onto the profile so BMI, calorie goals
        # and the AI prompt context all stay current without a separate step.
        profile = await self.profiles.get_or_create(user_id)
        latest = await self.weight.latest(user_id)
        if latest is not None and latest.logged_on == payload.logged_on:
            profile.weight_kg = payload.weight_kg
        await self.session.commit()
        return row

    async def upsert_mood(self, user_id: uuid.UUID, payload: MoodUpsert) -> MoodLog:
        row = await self.mood.upsert(
            user_id,
            payload.logged_on,
            mood=payload.mood.value,
            energy_level=payload.energy_level,
            stress_level=payload.stress_level,
            journal=payload.journal,
        )
        await self.session.commit()
        return row

    async def water_range(self, user_id: uuid.UUID, start: date, end: date):
        return await self.water.list_between(user_id, start, end)

    async def sleep_range(self, user_id: uuid.UUID, start: date, end: date):
        return await self.sleep.list_between(user_id, start, end)

    async def weight_range(self, user_id: uuid.UUID, start: date, end: date):
        return await self.weight.list_between(user_id, start, end)

    async def mood_range(self, user_id: uuid.UUID, start: date, end: date):
        return await self.mood.list_between(user_id, start, end)
