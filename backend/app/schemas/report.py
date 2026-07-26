"""Blood-report OCR and food-image analysis schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import BiomarkerFlag, MealType, ReportStatus
from app.schemas.common import ORMModel


class BiomarkerResponse(ORMModel):
    id: uuid.UUID
    name: str
    display_name: str
    value: float
    unit: str
    reference_low: float | None
    reference_high: float | None
    flag: BiomarkerFlag
    interpretation: str | None


class BloodReportResponse(ORMModel):
    id: uuid.UUID
    filename: str
    file_url: str | None
    content_type: str
    status: ReportStatus
    ocr_confidence: float | None
    summary: str | None
    error_message: str | None
    created_at: datetime
    biomarkers: list[BiomarkerResponse] = Field(default_factory=list)


class BloodReportDetail(BloodReportResponse):
    extracted_text: str | None = None
    #: Written by the Blood Report Analyzer agent, grounded in the biomarkers.
    ai_explanation: str | None = None
    disclaimer: str | None = None


class FoodItem(BaseModel):
    name: str
    confidence: float = Field(..., ge=0, le=1)
    estimated_grams: float
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fibre_g: float
    glycemic_index: int


class HealthierSwap(BaseModel):
    replace: str
    with_alternative: str
    reason: str
    calories_saved: float


class FoodAnalysisResponse(BaseModel):
    """Result of analysing a meal photo."""

    image_url: str | None
    detected_items: list[FoodItem]
    total_calories: float
    total_protein_g: float
    total_carbs_g: float
    total_fat_g: float
    total_fibre_g: float
    average_glycemic_index: int
    #: How well this meal fits a PCOS-supportive pattern, 0–100.
    pcos_score: int = Field(..., ge=0, le=100)
    pcos_verdict: str
    assessment: str
    healthier_swaps: list[HealthierSwap]
    suggested_meal_type: MealType
    disclaimer: str


class VoiceTranscription(BaseModel):
    text: str
    duration_seconds: float | None = None
    language: str = "en"
    confidence: float | None = None


class SpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    voice: str = Field("nova", max_length=40)
    speed: float = Field(1.0, ge=0.5, le=2.0)
