"""Daily trackers: habits, meals, workouts, water, sleep, weight and mood."""

from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, Query, status

from app.api.deps import (
    CurrentUserId,
    DailyTrackerServiceDep,
    DateWindowDep,
    HabitServiceDep,
    MealServiceDep,
    PlanServiceDep,
    WorkoutServiceDep,
)
from app.schemas.common import ErrorResponse, MessageResponse
from app.schemas.tracking import (
    DietPlanRequest,
    DietPlanResponse,
    HabitCreate,
    HabitEntryCreate,
    HabitEntryResponse,
    HabitResponse,
    HabitUpdate,
    MealCreate,
    MealResponse,
    MoodResponse,
    MoodUpsert,
    NutritionSummary,
    SleepResponse,
    SleepUpsert,
    WaterResponse,
    WaterUpsert,
    WeightResponse,
    WeightUpsert,
    WorkoutCreate,
    WorkoutPlanRequest,
    WorkoutPlanResponse,
    WorkoutResponse,
)

router = APIRouter(tags=["Daily Tracking"])


# ------------------------------------------------------------------- habits
@router.post(
    "/habits",
    response_model=HabitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a habit",
    responses={409: {"model": ErrorResponse, "description": "Habit name in use"}},
)
async def create_habit(
    payload: HabitCreate, user_id: CurrentUserId, service: HabitServiceDep
) -> HabitResponse:
    habit = await service.create(user_id, payload)
    return HabitResponse.model_validate(habit)


@router.get(
    "/habits",
    response_model=list[HabitResponse],
    summary="List habits with streaks",
)
async def list_habits(
    user_id: CurrentUserId, service: HabitServiceDep
) -> list[HabitResponse]:
    """Active habits enriched with current streak, longest streak and 30-day rate.

    A streak stays alive if yesterday was completed — resetting it at midnight
    before the user has had a chance to log is exactly what makes people
    abandon a habit tracker.
    """
    return [HabitResponse(**row) for row in await service.list_with_stats(user_id)]


@router.patch(
    "/habits/{habit_id}", response_model=HabitResponse, summary="Update a habit"
)
async def update_habit(
    habit_id: uuid.UUID,
    payload: HabitUpdate,
    user_id: CurrentUserId,
    service: HabitServiceDep,
) -> HabitResponse:
    await service.update(user_id, habit_id, payload)
    rows = await service.list_with_stats(user_id)
    row = next((r for r in rows if r["id"] == habit_id), None)
    if row is None:  # archived habits drop out of the active list
        habit = await service.habits.get_for_user(habit_id, user_id)
        return HabitResponse.model_validate(habit)
    return HabitResponse(**row)


@router.post(
    "/habits/{habit_id}/check-in",
    response_model=HabitEntryResponse,
    summary="Tick a habit for a day",
)
async def check_in_habit(
    habit_id: uuid.UUID,
    payload: HabitEntryCreate,
    user_id: CurrentUserId,
    service: HabitServiceDep,
) -> HabitEntryResponse:
    """Log a completion. Sending ``completed_count: 0`` undoes that day's tick."""
    entry = await service.check_in(user_id, habit_id, payload)
    return HabitEntryResponse(
        id=getattr(entry, "id", uuid.uuid4()),
        logged_on=entry.logged_on,
        completed_count=entry.completed_count,
        note=entry.note,
    )


@router.delete(
    "/habits/{habit_id}", response_model=MessageResponse, summary="Delete a habit"
)
async def delete_habit(
    habit_id: uuid.UUID, user_id: CurrentUserId, service: HabitServiceDep
) -> MessageResponse:
    await service.delete(user_id, habit_id)
    return MessageResponse(message="Habit deleted.")


# -------------------------------------------------------------------- meals
@router.post(
    "/meals",
    response_model=MealResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Log a meal",
)
async def log_meal(
    payload: MealCreate, user_id: CurrentUserId, service: MealServiceDep
) -> MealResponse:
    meal = await service.log(user_id, payload)
    return MealResponse.model_validate(meal)


@router.get("/meals", response_model=list[MealResponse], summary="Meals for a day")
async def list_meals(
    user_id: CurrentUserId,
    service: MealServiceDep,
    day: date = Query(default_factory=date.today),
) -> list[MealResponse]:
    meals = await service.for_day(user_id, day)
    return [MealResponse.model_validate(m) for m in meals]


@router.get(
    "/meals/summary",
    response_model=NutritionSummary,
    summary="Nutrition totals for a day",
)
async def nutrition_summary(
    user_id: CurrentUserId,
    service: MealServiceDep,
    day: date = Query(default_factory=date.today),
) -> NutritionSummary:
    """Daily macro totals against personalised goals.

    Calorie targets use Mifflin-St Jeor for BMR with an activity factor;
    protein is set at the upper end of the general range, since satiety and
    lean-mass retention both matter here.
    """
    return await service.daily_summary(user_id, day)


@router.delete(
    "/meals/{meal_id}", response_model=MessageResponse, summary="Delete a meal"
)
async def delete_meal(
    meal_id: uuid.UUID, user_id: CurrentUserId, service: MealServiceDep
) -> MessageResponse:
    await service.delete(user_id, meal_id)
    return MessageResponse(message="Meal deleted.")


# ----------------------------------------------------------------- workouts
@router.post(
    "/workouts",
    response_model=WorkoutResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Log a workout",
)
async def log_workout(
    payload: WorkoutCreate, user_id: CurrentUserId, service: WorkoutServiceDep
) -> WorkoutResponse:
    """Record a session. Energy burn is MET-estimated when not supplied."""
    workout = await service.log(user_id, payload)
    return WorkoutResponse.model_validate(workout)


@router.get("/workouts", response_model=list[WorkoutResponse], summary="List workouts")
async def list_workouts(
    user_id: CurrentUserId, service: WorkoutServiceDep, window: DateWindowDep
) -> list[WorkoutResponse]:
    rows = await service.list_range(user_id, window.start, window.end)
    return [WorkoutResponse.model_validate(w) for w in rows]


@router.delete(
    "/workouts/{workout_id}",
    response_model=MessageResponse,
    summary="Delete a workout",
)
async def delete_workout(
    workout_id: uuid.UUID, user_id: CurrentUserId, service: WorkoutServiceDep
) -> MessageResponse:
    await service.delete(user_id, workout_id)
    return MessageResponse(message="Workout deleted.")


# ------------------------------------------------------------ daily loggers
@router.put("/water", response_model=WaterResponse, summary="Set today's water")
async def upsert_water(
    payload: WaterUpsert, user_id: CurrentUserId, service: DailyTrackerServiceDep
) -> WaterResponse:
    """Water is one row per day, so this is idempotent — PUT, not POST."""
    row = await service.upsert_water(user_id, payload)
    return WaterResponse(
        id=row.id,
        logged_on=row.logged_on,
        millilitres=row.millilitres,
        goal_millilitres=row.goal_millilitres,
        goal_percentage=row.goal_percentage,
    )


@router.get("/water", response_model=list[WaterResponse], summary="Water history")
async def list_water(
    user_id: CurrentUserId, service: DailyTrackerServiceDep, window: DateWindowDep
) -> list[WaterResponse]:
    rows = await service.water_range(user_id, window.start, window.end)
    return [
        WaterResponse(
            id=r.id,
            logged_on=r.logged_on,
            millilitres=r.millilitres,
            goal_millilitres=r.goal_millilitres,
            goal_percentage=r.goal_percentage,
        )
        for r in rows
    ]


@router.put("/sleep", response_model=SleepResponse, summary="Log a night's sleep")
async def upsert_sleep(
    payload: SleepUpsert, user_id: CurrentUserId, service: DailyTrackerServiceDep
) -> SleepResponse:
    row = await service.upsert_sleep(user_id, payload)
    return SleepResponse.model_validate(row)


@router.get("/sleep", response_model=list[SleepResponse], summary="Sleep history")
async def list_sleep(
    user_id: CurrentUserId, service: DailyTrackerServiceDep, window: DateWindowDep
) -> list[SleepResponse]:
    rows = await service.sleep_range(user_id, window.start, window.end)
    return [SleepResponse.model_validate(r) for r in rows]


@router.put("/weight", response_model=WeightResponse, summary="Log weight")
async def upsert_weight(
    payload: WeightUpsert, user_id: CurrentUserId, service: DailyTrackerServiceDep
) -> WeightResponse:
    """Record a weigh-in. The profile's current weight is mirrored from the
    most recent entry, so BMI and calorie goals stay current automatically."""
    row = await service.upsert_weight(user_id, payload)
    profile = await service.profiles.get_by_user(user_id)
    return WeightResponse(
        id=row.id,
        logged_on=row.logged_on,
        weight_kg=row.weight_kg,
        body_fat_percentage=row.body_fat_percentage,
        waist_cm=row.waist_cm,
        bmi=profile.bmi if profile else None,
    )


@router.get("/weight", response_model=list[WeightResponse], summary="Weight history")
async def list_weight(
    user_id: CurrentUserId, service: DailyTrackerServiceDep, window: DateWindowDep
) -> list[WeightResponse]:
    rows = await service.weight_range(user_id, window.start, window.end)
    return [WeightResponse.model_validate(r) for r in rows]


@router.put("/mood", response_model=MoodResponse, summary="Log mood")
async def upsert_mood(
    payload: MoodUpsert, user_id: CurrentUserId, service: DailyTrackerServiceDep
) -> MoodResponse:
    row = await service.upsert_mood(user_id, payload)
    return MoodResponse(
        id=row.id,
        logged_on=row.logged_on,
        mood=row.mood,
        energy_level=row.energy_level,
        stress_level=row.stress_level,
        journal=row.journal,
        mood_score=row.mood_score,
    )


@router.get("/mood", response_model=list[MoodResponse], summary="Mood history")
async def list_mood(
    user_id: CurrentUserId, service: DailyTrackerServiceDep, window: DateWindowDep
) -> list[MoodResponse]:
    rows = await service.mood_range(user_id, window.start, window.end)
    return [
        MoodResponse(
            id=r.id,
            logged_on=r.logged_on,
            mood=r.mood,
            energy_level=r.energy_level,
            stress_level=r.stress_level,
            journal=r.journal,
            mood_score=r.mood_score,
        )
        for r in rows
    ]


# ----------------------------------------------------------------- planning
@router.post(
    "/plans/workout",
    response_model=WorkoutPlanResponse,
    summary="Generate a workout plan",
)
async def generate_workout_plan(
    payload: WorkoutPlanRequest, user_id: CurrentUserId, service: PlanServiceDep
) -> WorkoutPlanResponse:
    """Build a weekly programme weighted toward resistance training.

    Deterministic rather than LLM-generated: the same request returns the same
    plan, which makes it testable and makes it feel considered rather than
    random. Strength is prioritised because muscle is the body's largest
    glucose sink, which is the mechanism that matters in PCOS.
    """
    return await service.generate_workout_plan(user_id, payload)


@router.post(
    "/plans/diet", response_model=DietPlanResponse, summary="Generate a meal plan"
)
async def generate_diet_plan(
    payload: DietPlanRequest, user_id: CurrentUserId, service: PlanServiceDep
) -> DietPlanResponse:
    """Build a meal plan from the same food database the tracker uses.

    Generated in code, not by a language model, for two reasons: the macros
    have to actually add up, and allergy and dietary exclusions have to hold
    with certainty rather than with high probability. Exclusions are applied as
    a hard filter before any food is selected.
    """
    return await service.generate_diet_plan(user_id, payload)
