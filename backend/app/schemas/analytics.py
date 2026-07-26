"""Dashboard and analytics response shapes.

These schemas are deliberately chart-ready: the frontend passes ``TimeSeries``
straight into Chart.js without reshaping. Doing the aggregation server-side
keeps the payload small and means the dashboard renders one request, not twelve.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class TrendPoint(BaseModel):
    label: str
    value: float | None


class TimeSeries(BaseModel):
    metric: str
    unit: str
    points: list[TrendPoint]
    average: float | None = None
    change_percentage: float | None = None
    #: "up" | "down" | "flat" — whether the metric is *improving*, which is not
    #: the same as increasing (falling weight can be an "up" trend).
    direction: str = "flat"


class MetricCard(BaseModel):
    """One KPI tile on the dashboard."""

    key: str
    label: str
    value: float | None
    unit: str
    secondary_label: str | None = None
    secondary_value: str | None = None
    change_percentage: float | None = None
    direction: str = "flat"
    goal: float | None = None
    progress_percentage: float | None = None
    icon: str = "activity"


class AIInsight(BaseModel):
    """A generated observation with a concrete next action."""

    title: str
    body: str
    category: str
    severity: str  # "info" | "positive" | "attention"
    action_label: str | None = None
    action_url: str | None = None


class DashboardResponse(BaseModel):
    generated_at: date
    greeting: str
    metrics: list[MetricCard]
    weight_trend: TimeSeries
    sleep_trend: TimeSeries
    calorie_trend: TimeSeries
    water_trend: TimeSeries
    mood_trend: TimeSeries
    workout_minutes_trend: TimeSeries
    macro_split: dict[str, float]
    habit_completion: list[dict]
    cycle_summary: dict
    latest_risk: dict | None
    insights: list[AIInsight]
    streaks: dict[str, int]


class WeeklyProgress(BaseModel):
    week_start: date
    week_end: date
    workouts_completed: int
    workout_minutes: int
    average_sleep_hours: float | None
    average_calories: float | None
    average_water_ml: float | None
    habits_completion_rate: float
    mood_average: float | None
    weight_change_kg: float | None
    highlights: list[str]


class MonthlyAnalytics(BaseModel):
    month: str
    days_tracked: int
    weight_series: TimeSeries
    sleep_series: TimeSeries
    calorie_series: TimeSeries
    mood_series: TimeSeries
    workout_series: TimeSeries
    symptom_frequency: list[dict]
    cycle_regularity: dict
    macro_averages: dict[str, float]
    consistency_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Share of days in the month with at least one log entry.",
    )
    insights: list[AIInsight]
