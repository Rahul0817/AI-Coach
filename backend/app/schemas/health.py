"""Cycle, symptom and risk-prediction schemas."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, Field, model_validator

from app.models.enums import (
    ActivityLevel,
    FlowIntensity,
    RiskBand,
    SymptomType,
)
from app.schemas.common import ORMModel


# --------------------------------------------------------------------- cycle
class CycleCreate(BaseModel):
    start_date: date
    end_date: date | None = None
    period_length_days: int | None = Field(None, ge=1, le=21)
    flow_intensity: FlowIntensity = FlowIntensity.MEDIUM
    pain_level: int | None = Field(None, ge=0, le=10)
    notes: str | None = Field(None, max_length=1000)

    @model_validator(mode="after")
    def _coherent_dates(self) -> CycleCreate:
        if self.start_date > date.today():
            raise ValueError("Cycle start date cannot be in the future.")
        if self.end_date:
            if self.end_date < self.start_date:
                raise ValueError("End date cannot precede start date.")
            span = (self.end_date - self.start_date).days + 1
            if span > 21:
                raise ValueError(
                    "A period longer than 21 days is not accepted here; "
                    "please seek medical advice."
                )
            # Derive the period length when the client did not supply one.
            if self.period_length_days is None:
                object.__setattr__(self, "period_length_days", span)
        return self


class CycleUpdate(BaseModel):
    end_date: date | None = None
    period_length_days: int | None = Field(None, ge=1, le=21)
    flow_intensity: FlowIntensity | None = None
    pain_level: int | None = Field(None, ge=0, le=10)
    notes: str | None = Field(None, max_length=1000)


class CycleResponse(ORMModel):
    id: uuid.UUID
    start_date: date
    end_date: date | None
    period_length_days: int | None
    cycle_length_days: int | None
    flow_intensity: str
    pain_level: int | None
    notes: str | None
    created_at: datetime
    is_irregular: bool = False


class CycleInsights(BaseModel):
    """Aggregate view of cycle regularity, shown on the tracker page."""

    total_cycles: int
    average_cycle_length: float | None
    shortest_cycle: int | None
    longest_cycle: int | None
    #: Standard deviation of cycle length. High variance is the clinically
    #: interesting signal for PCOS, more so than any single long cycle.
    cycle_length_std_dev: float | None
    irregular_cycle_count: int
    regularity_label: str
    predicted_next_start: date | None
    predicted_fertile_window: tuple[date, date] | None
    days_until_next: int | None
    average_period_length: float | None


# ------------------------------------------------------------------ symptoms
class SymptomCreate(BaseModel):
    logged_on: date = Field(default_factory=date.today)
    symptom: SymptomType
    severity: int = Field(..., ge=1, le=5)
    notes: str | None = Field(None, max_length=500)

    @model_validator(mode="after")
    def _not_future(self) -> SymptomCreate:
        if self.logged_on > date.today():
            raise ValueError("Cannot log a symptom for a future date.")
        return self


class SymptomResponse(ORMModel):
    id: uuid.UUID
    logged_on: date
    symptom: str
    severity: int
    notes: str | None
    created_at: datetime


class SymptomSummary(BaseModel):
    symptom: str
    occurrences: int
    average_severity: float
    last_logged: date
    trend: str  # "increasing" | "stable" | "decreasing"


# ---------------------------------------------------------------- prediction
class RiskAssessmentRequest(BaseModel):
    """Feature vector for the PCOS risk model.

    Field names mirror the trained model's expected columns exactly. Keeping the
    API contract and the model schema identical removes an entire class of
    silent-mismatch bugs — the model's own feature list is the source of truth
    and is validated against this schema at startup.
    """

    age: int = Field(..., ge=10, le=100, examples=[24])
    bmi: float = Field(..., ge=12.0, le=60.0, examples=[27.4])
    cycle_length_days: int = Field(
        ...,
        ge=10,
        le=180,
        examples=[42],
        description="Average days between the start of consecutive periods.",
    )
    cycle_irregularity: int = Field(
        ...,
        ge=0,
        le=1,
        examples=[1],
        description="1 if periods are self-reported as irregular, else 0.",
    )
    weight_gain: int = Field(
        ..., ge=0, le=1, description="Recent unexplained weight gain."
    )
    hair_growth: int = Field(
        ..., ge=0, le=1, description="Excess facial/body hair (hirsutism)."
    )
    skin_darkening: int = Field(..., ge=0, le=1, description="Acanthosis nigricans.")
    hair_loss: int = Field(..., ge=0, le=1, description="Scalp hair thinning.")
    pimples: int = Field(..., ge=0, le=1, description="Persistent acne.")
    fast_food: int = Field(..., ge=0, le=1, description="Frequent fast-food intake.")
    exercise_hours_per_week: float = Field(..., ge=0, le=40, examples=[2.5])
    sleep_hours: float = Field(..., ge=0, le=16, examples=[6.5])
    stress_level: int = Field(..., ge=1, le=5, examples=[4])
    family_history: int = Field(
        ..., ge=0, le=1, description="PCOS in a first-degree relative."
    )
    activity_level: ActivityLevel = ActivityLevel.MODERATE

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 24,
                "bmi": 27.4,
                "cycle_length_days": 42,
                "cycle_irregularity": 1,
                "weight_gain": 1,
                "hair_growth": 1,
                "skin_darkening": 0,
                "hair_loss": 1,
                "pimples": 1,
                "fast_food": 1,
                "exercise_hours_per_week": 2.5,
                "sleep_hours": 6.5,
                "stress_level": 4,
                "family_history": 1,
                "activity_level": "light",
            }
        }
    }


class FeatureContribution(BaseModel):
    """One row of the SHAP explanation."""

    feature: str
    display_name: str
    value: float | str
    shap_value: float
    direction: str  # "increases" | "decreases"
    explanation: str


class RiskAssessmentResponse(ORMModel):
    id: uuid.UUID | None = None
    risk_score: float = Field(..., ge=0, le=1)
    risk_percentage: float
    risk_band: RiskBand
    confidence: float
    model_name: str
    model_version: str
    summary: str
    top_factors: list[FeatureContribution]
    recommendations: list[str]
    disclaimer: str
    created_at: datetime | None = None


class PredictionHistoryItem(ORMModel):
    id: uuid.UUID
    risk_score: float
    risk_band: str
    confidence: float
    model_version: str
    created_at: datetime


class ModelMetadata(BaseModel):
    """Served at ``/predict/model-info`` so the UI can show provenance."""

    model_name: str
    model_version: str
    trained_at: str
    n_training_samples: int
    features: list[str]
    metrics: dict[str, float]
    all_model_scores: dict[str, dict[str, float]]
