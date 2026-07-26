"""Workout and diet plan generation.

These are **deterministic generators**, not LLM calls, and that is a considered
choice rather than a shortcut:

* A meal plan has to add up. An LLM asked for 1,800 kcal will confidently
  produce a day that totals 2,340, and the user has no way to tell. Here the
  macros come from the same food database the tracker uses, so the numbers on
  the plan are the numbers that get logged.
* Plans must respect hard constraints — allergies, vegetarianism — with
  certainty. A probabilistic system that puts peanuts in a plan for someone
  with a peanut allergy is unacceptable at any error rate.
* They are free and instant, so a user can regenerate as often as they like.

The LLM's role in this product is conversation and explanation, where fluency
is the value. Where arithmetic and constraints are the value, code wins.
"""

from __future__ import annotations

import random
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.multimodal.nutrition_db import FOODS, FoodItem
from app.core.logging import get_logger
from app.models.enums import DietaryPreference, Intensity, MealType, WorkoutType
from app.repositories.user import ProfileRepository
from app.schemas.common import MEDICAL_DISCLAIMER
from app.schemas.tracking import (
    DietPlanRequest,
    DietPlanResponse,
    PlannedDay,
    PlannedMeal,
    PlannedWorkout,
    WorkoutPlanRequest,
    WorkoutPlanResponse,
)

logger = get_logger(__name__)

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]

#: Exercise pools by equipment availability. Compound movements first — they
#: recruit the most muscle, which is what drives glucose disposal.
STRENGTH_BODYWEIGHT = [
    "Goblet squat to chair — 3 × 10",
    "Incline push-up (hands on a table) — 3 × 8",
    "Glute bridge — 3 × 12",
    "Reverse lunge — 3 × 8 each leg",
    "Bent-over row with a bag or bottles — 3 × 12",
    "Plank — 3 × 30 seconds",
    "Bird dog — 3 × 10 each side",
    "Wall sit — 3 × 30 seconds",
]
STRENGTH_EQUIPPED = [
    "Goblet squat — 4 × 8",
    "Romanian deadlift — 4 × 8",
    "Dumbbell bench or floor press — 3 × 10",
    "Single-arm dumbbell row — 3 × 10 each side",
    "Walking lunge — 3 × 10 each leg",
    "Lat pulldown or assisted pull-up — 3 × 10",
    "Overhead press — 3 × 8",
    "Farmer's carry — 3 × 40 metres",
    "Hip thrust — 4 × 10",
]
CARDIO_BLOCKS = [
    "Brisk walk at a pace where talking is possible but singing is not",
    "Cycling at a steady, conversational effort",
    "Swimming — continuous laps at an easy pace",
    "Incline treadmill walk — 6% grade",
    "Elliptical — steady state",
]
HIIT_BLOCKS = [
    "30 seconds hard / 90 seconds easy × 8 rounds",
    "40 seconds work / 20 seconds rest × 10 rounds (bodyweight circuit)",
    "1 minute hard / 2 minutes easy × 6 rounds",
]
YOGA_BLOCKS = [
    "Sun salutations — 5 rounds, unhurried",
    "Hip-opening sequence with long holds",
    "Restorative sequence: legs up the wall, supported twist, savasana",
    "Gentle vinyasa flow focused on breath",
]
MOBILITY_BLOCKS = [
    "Thoracic spine rotations — 10 each side",
    "90/90 hip switches — 10 each side",
    "Cat-cow — 10 slow repetitions",
    "Deep squat hold — 3 × 45 seconds",
]


class PlanService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.profiles = ProfileRepository(session)

    # ------------------------------------------------------------- workouts
    async def generate_workout_plan(
        self, user_id: uuid.UUID, payload: WorkoutPlanRequest
    ) -> WorkoutPlanResponse:
        profile = await self.profiles.get_by_user(user_id)
        has_equipment = bool(payload.equipment) and payload.equipment != ["none"]
        strength_pool = STRENGTH_EQUIPPED if has_equipment else STRENGTH_BODYWEIGHT

        # Deterministic per user and per request shape: regenerating the same
        # request returns the same plan, which makes it feel considered rather
        # than random, and makes the endpoint testable.
        rng = random.Random(
            f"{user_id}-{payload.days_per_week}-{payload.intensity.value}"
        )

        # Strength is prioritised over cardio for PCOS: muscle is the primary
        # glucose sink, so it is the highest-value use of limited sessions.
        template = self._session_template(payload.days_per_week)
        schedule: list[PlannedWorkout] = []

        for index, kind in enumerate(template):
            day = DAY_NAMES[index % 7]
            if kind == WorkoutType.STRENGTH:
                exercises = rng.sample(strength_pool, min(5, len(strength_pool)))
                title = "Full-body strength"
                rationale = (
                    "Resistance training increases the muscle available to clear "
                    "glucose from your blood, which improves insulin sensitivity "
                    "whether or not your weight changes."
                )
            elif kind == WorkoutType.CARDIO:
                exercises = [
                    rng.choice(CARDIO_BLOCKS),
                    "Finish with 5 minutes of easy walking to cool down",
                ]
                title = "Steady-state cardio"
                rationale = (
                    "Moderate aerobic work improves glucose uptake for 24 to 48 "
                    "hours afterwards, so doing it often matters more than doing "
                    "it hard."
                )
            elif kind == WorkoutType.HIIT:
                exercises = [
                    "5 minutes easy warm-up",
                    rng.choice(HIIT_BLOCKS),
                    "5 minutes easy cool-down",
                ]
                title = "Interval session"
                rationale = (
                    "One interval session a week is time-efficient. Skip it if "
                    "you slept badly — intensity without recovery raises cortisol "
                    "and works against you."
                )
            elif kind == WorkoutType.YOGA:
                exercises = [rng.choice(YOGA_BLOCKS)]
                title = "Yoga and stress regulation"
                rationale = (
                    "Yoga has shown benefits for anxiety and perceived stress in "
                    "PCOS trials, and lower stress supports a more regular cycle."
                )
            else:
                exercises = rng.sample(MOBILITY_BLOCKS, 3) + [
                    "20-minute easy walk"
                ]
                title = "Active recovery"
                rationale = (
                    "Recovery days are where adaptation happens. Gentle movement "
                    "beats complete rest for how you feel the next day."
                )

            if payload.limitations:
                rationale += (
                    f" Adjusted for: {', '.join(payload.limitations)} — swap any "
                    f"movement that causes pain."
                )

            schedule.append(
                PlannedWorkout(
                    day=day,
                    workout_type=kind,
                    title=title,
                    duration_minutes=payload.minutes_per_session,
                    intensity=(
                        Intensity.HIGH if kind == WorkoutType.HIIT
                        else Intensity.LOW if kind in {WorkoutType.YOGA, WorkoutType.MOBILITY}
                        else payload.intensity
                    ),
                    exercises=exercises,
                    rationale=rationale,
                )
            )

        activity = profile.activity_level if profile else "moderate"
        guidance = self._workout_guidance(payload, activity)

        return WorkoutPlanResponse(
            plan_name=(
                f"{payload.days_per_week}-day PCOS training plan"
                + (f" — {payload.focus}" if payload.focus else "")
            ),
            days_per_week=payload.days_per_week,
            weekly_schedule=schedule,
            guidance=guidance,
            disclaimer=MEDICAL_DISCLAIMER,
        )

    @staticmethod
    def _session_template(days: int) -> list[WorkoutType]:
        """Weekly split, weighted toward strength."""
        templates = {
            1: [WorkoutType.STRENGTH],
            2: [WorkoutType.STRENGTH, WorkoutType.STRENGTH],
            3: [WorkoutType.STRENGTH, WorkoutType.CARDIO, WorkoutType.STRENGTH],
            4: [WorkoutType.STRENGTH, WorkoutType.CARDIO, WorkoutType.STRENGTH,
                WorkoutType.YOGA],
            5: [WorkoutType.STRENGTH, WorkoutType.CARDIO, WorkoutType.STRENGTH,
                WorkoutType.HIIT, WorkoutType.YOGA],
            6: [WorkoutType.STRENGTH, WorkoutType.CARDIO, WorkoutType.STRENGTH,
                WorkoutType.HIIT, WorkoutType.YOGA, WorkoutType.MOBILITY],
            7: [WorkoutType.STRENGTH, WorkoutType.CARDIO, WorkoutType.STRENGTH,
                WorkoutType.HIIT, WorkoutType.YOGA, WorkoutType.CARDIO,
                WorkoutType.MOBILITY],
        }
        return templates.get(days, templates[4])

    @staticmethod
    def _workout_guidance(payload: WorkoutPlanRequest, activity: str) -> str:
        total = payload.days_per_week * payload.minutes_per_session
        parts = [
            f"This plan totals about {total} minutes a week"
            + (
                ", which meets the 150-minute guideline."
                if total >= 150
                else f". Building toward 150 minutes is the target — add "
                     f"10 minutes per session every fortnight."
            ),
        ]
        if activity == "sedentary":
            parts.append(
                "You are starting from a sedentary baseline, so treat the first "
                "fortnight as a rehearsal. Complete the sessions at an easy "
                "effort; the aim is to prove the schedule fits your life before "
                "you make it hard."
            )
        parts.append(
            "Progressive overload is what makes strength work keep paying off: "
            "each week add a repetition, a small amount of load, or a set. If "
            "nothing increases, adaptation stops."
        )
        parts.append(
            "Post-meal walks are not on the schedule because they are not a "
            "workout — but ten minutes after your largest meal is one of the "
            "highest-return habits available for blood-sugar control."
        )
        return " ".join(parts)

    # ----------------------------------------------------------------- diet
    async def generate_diet_plan(
        self, user_id: uuid.UUID, payload: DietPlanRequest
    ) -> DietPlanResponse:
        profile = await self.profiles.get_by_user(user_id)

        preference = payload.dietary_preference or (
            DietaryPreference(profile.dietary_preference)
            if profile else DietaryPreference.OMNIVORE
        )
        allergies = {a.lower() for a in (payload.allergies or [])}
        if profile and profile.allergies:
            allergies |= {a.lower() for a in profile.allergies}

        target = payload.target_calories
        if target is None:
            from app.services.tracking_service import MealService

            target, _ = MealService._goals(profile)
            target = int(target)

        rng = random.Random(f"{user_id}-{target}-{preference.value}-{payload.days}")
        pool = self._allowed_foods(preference, allergies)

        days: list[PlannedDay] = []
        for index in range(payload.days):
            meals = self._compose_day(pool, target, rng, index)
            days.append(
                PlannedDay(
                    day=DAY_NAMES[index % 7],
                    meals=meals,
                    total_calories=round(sum(m.calories for m in meals), 1),
                    total_protein_g=round(sum(m.protein_g for m in meals), 1),
                )
            )

        return DietPlanResponse(
            plan_name=(
                f"{payload.days}-day PCOS meal plan "
                f"(~{target} kcal, {preference.value})"
            ),
            days=days,
            principles=[
                "Every meal pairs carbohydrate with protein or fat, so glucose "
                "rises gently rather than spiking.",
                "Protein is front-loaded at breakfast, which reliably reduces "
                "cravings later in the day.",
                "Fibre targets 25–30g daily from legumes, vegetables and whole "
                "grains to slow glucose absorption.",
                "Lower-glycaemic staples are preferred, but nothing is banned — "
                "restrictive plans get abandoned, and abandonment helps nobody.",
                "Portions are estimates. Adjust to your appetite rather than "
                "forcing a number.",
            ],
            guidance=(
                f"This plan targets roughly {target} kcal a day with protein at "
                f"about {round(sum(m.protein_g for m in days[0].meals))}g. "
                f"Swap any meal for another from the same slot — the structure "
                f"matters far more than the specific foods. If you are hungry, "
                f"add protein or vegetables rather than tolerating it; hunger is "
                f"what makes plans fail."
            ),
            disclaimer=MEDICAL_DISCLAIMER,
        )

    @staticmethod
    def _allowed_foods(
        preference: DietaryPreference, allergies: set[str]
    ) -> dict[str, list[FoodItem]]:
        """Filter the food database by dietary preference and allergies.

        Exclusions are applied as a hard filter before any selection happens, so
        a forbidden food cannot reach the plan by any code path.
        """
        animal_flesh = {"chicken_breast", "chicken_curry", "fish", "salmon"}
        all_animal = animal_flesh | {
            "egg", "paneer", "greek_yogurt", "curd", "milk", "whey_protein"
        }

        excluded: set[str] = set()
        if preference == DietaryPreference.VEGETARIAN:
            excluded |= animal_flesh | {"egg"}
        elif preference == DietaryPreference.VEGAN:
            excluded |= all_animal
        elif preference == DietaryPreference.EGGETARIAN:
            excluded |= animal_flesh
        elif preference == DietaryPreference.PESCATARIAN:
            excluded |= {"chicken_breast", "chicken_curry"}

        def blocked(item: FoodItem) -> bool:
            if item.key in excluded:
                return True
            haystack = " ".join([item.name.lower(), *[a.lower() for a in item.aliases]])
            return any(allergen and allergen in haystack for allergen in allergies)

        buckets: dict[str, list[FoodItem]] = {
            "protein": [], "grain": [], "legume": [], "vegetable": [],
            "fruit": [], "fat": [],
        }
        for item in FOODS.values():
            if item.category in buckets and not blocked(item) and item.pcos_score >= 45:
                buckets[item.category].append(item)

        # Best PCOS scores first so selection favours them.
        for key in buckets:
            buckets[key].sort(key=lambda i: i.pcos_score, reverse=True)
        return buckets

    def _compose_day(
        self,
        pool: dict[str, list[FoodItem]],
        target: int,
        rng: random.Random,
        day_index: int,
    ) -> list[PlannedMeal]:
        """Build four meals hitting roughly the calorie target.

        Energy is split 25/35/30/10 across breakfast, lunch, dinner and a snack
        — a distribution that keeps lunch substantial and dinner moderate, which
        suits evening blood-glucose control.
        """
        splits = [
            (MealType.BREAKFAST, 0.25),
            (MealType.LUNCH, 0.35),
            (MealType.DINNER, 0.30),
            (MealType.SNACK, 0.10),
        ]
        meals: list[PlannedMeal] = []

        for meal_type, share in splits:
            budget = target * share
            components = self._pick_components(pool, meal_type, rng, day_index)
            if not components:
                continue

            # Scale portions so the assembled meal lands near its budget.
            base_calories = sum(
                item.calories * item.typical_grams / 100 for item in components
            )
            scale = (budget / base_calories) if base_calories > 0 else 1.0
            scale = max(0.5, min(2.0, scale))  # keep portions realistic

            totals = {"cal": 0.0, "p": 0.0, "c": 0.0, "f": 0.0, "fib": 0.0}
            gi_weighted = 0.0
            for item in components:
                grams = item.typical_grams * scale
                factor = grams / 100
                totals["cal"] += item.calories * factor
                totals["p"] += item.protein_g * factor
                totals["c"] += item.carbs_g * factor
                totals["f"] += item.fat_g * factor
                totals["fib"] += item.fibre_g * factor
                gi_weighted += item.glycemic_index * (item.carbs_g * factor)

            average_gi = (
                round(gi_weighted / totals["c"]) if totals["c"] > 0 else 0
            )
            names = " + ".join(item.name for item in components)

            meals.append(
                PlannedMeal(
                    meal_type=meal_type,
                    name=names,
                    description=self._describe(components, scale),
                    calories=round(totals["cal"], 1),
                    protein_g=round(totals["p"], 1),
                    carbs_g=round(totals["c"], 1),
                    fat_g=round(totals["f"], 1),
                    fibre_g=round(totals["fib"], 1),
                    glycemic_index=average_gi,
                    pcos_benefit=self._benefit(totals, average_gi),
                )
            )
        return meals

    @staticmethod
    def _pick_components(
        pool: dict[str, list[FoodItem]],
        meal_type: MealType,
        rng: random.Random,
        day_index: int,
    ) -> list[FoodItem]:
        """Choose foods for one meal slot, always including a protein source."""
        def take(category: str, offset: int) -> FoodItem | None:
            items = pool.get(category) or []
            if not items:
                return None
            # Rotate through the ranked list by day so a seven-day plan is
            # varied rather than repeating the top-scoring food every day.
            return items[(day_index + offset) % len(items)]

        if meal_type == MealType.BREAKFAST:
            picks = [take("protein", 0), take("grain", 1), take("fruit", 2)]
        elif meal_type == MealType.LUNCH:
            picks = [take("legume", day_index), take("grain", 0), take("vegetable", 1)]
        elif meal_type == MealType.DINNER:
            picks = [take("protein", 3), take("vegetable", 2), take("grain", 3)]
        else:
            picks = [take("fat", day_index), take("fruit", day_index + 1)]

        chosen = [item for item in picks if item is not None]
        # Guarantee a protein source in every main meal.
        if meal_type != MealType.SNACK and not any(
            item.category in {"protein", "legume"} for item in chosen
        ):
            fallback = (pool.get("legume") or pool.get("protein") or [None])[0]
            if fallback is not None:
                chosen.append(fallback)
        return chosen

    @staticmethod
    def _describe(components: list[FoodItem], scale: float) -> str:
        parts = [
            f"{round(item.typical_grams * scale)}g {item.name.lower()}"
            for item in components
        ]
        return "Approximately " + ", ".join(parts) + "."

    @staticmethod
    def _benefit(totals: dict[str, float], average_gi: int) -> str:
        reasons: list[str] = []
        if totals["p"] >= 20:
            reasons.append(f"{totals['p']:.0f}g of protein to steady blood sugar")
        if totals["fib"] >= 6:
            reasons.append(f"{totals['fib']:.0f}g of fibre to slow glucose absorption")
        if average_gi and average_gi <= 55:
            reasons.append(f"a low glycaemic index of {average_gi}")
        if not reasons:
            reasons.append("a balanced mix of macronutrients")
        return "Provides " + ", and ".join(reasons) + "."
