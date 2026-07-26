"""Cycle tracking, symptom logging and ML risk prediction."""

from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, Query, status

from app.api.deps import (
    CurrentUserId,
    CycleServiceDep,
    DateWindowDep,
    PredictionServiceDep,
    SymptomServiceDep,
)
from app.ml.predictor import get_predictor
from app.schemas.common import ErrorResponse, MessageResponse
from app.schemas.health import (
    CycleCreate,
    CycleInsights,
    CycleResponse,
    CycleUpdate,
    ModelMetadata,
    PredictionHistoryItem,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    SymptomCreate,
    SymptomResponse,
    SymptomSummary,
)

router = APIRouter(tags=["Cycle, Symptoms & Risk"])


# ------------------------------------------------------------------- cycles
@router.post(
    "/cycles",
    response_model=CycleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Log a cycle",
    responses={409: {"model": ErrorResponse, "description": "Cycle already logged"}},
)
async def log_cycle(
    payload: CycleCreate, user_id: CurrentUserId, service: CycleServiceDep
) -> CycleResponse:
    """Record a cycle by its first day of bleeding.

    Logging a new start date also completes the *previous* cycle's length,
    which is only knowable once the next one begins.
    """
    cycle = await service.log_cycle(user_id, payload)
    return CycleResponse.model_validate(cycle)


@router.get("/cycles", response_model=list[CycleResponse], summary="List cycles")
async def list_cycles(
    user_id: CurrentUserId,
    service: CycleServiceDep,
    limit: int = Query(24, ge=1, le=120),
) -> list[CycleResponse]:
    cycles = await service.list_cycles(user_id, limit)
    return [CycleResponse.model_validate(c) for c in cycles]


@router.get(
    "/cycles/insights",
    response_model=CycleInsights,
    summary="Cycle regularity and next-period estimate",
)
async def cycle_insights(
    user_id: CurrentUserId, service: CycleServiceDep
) -> CycleInsights:
    """Regularity statistics plus a predicted next start.

    Regularity is judged on *variance*, not on any single cycle: consistently
    38-day cycles are predictable, whereas cycles swinging between 24 and 50
    days are the pattern that matters clinically — even at the same mean.

    Predictions in PCOS carry real uncertainty and **must never be used as
    contraception.**
    """
    return await service.insights(user_id)


@router.patch(
    "/cycles/{cycle_id}",
    response_model=CycleResponse,
    summary="Update a cycle",
    responses={404: {"model": ErrorResponse}},
)
async def update_cycle(
    cycle_id: uuid.UUID,
    payload: CycleUpdate,
    user_id: CurrentUserId,
    service: CycleServiceDep,
) -> CycleResponse:
    cycle = await service.update_cycle(user_id, cycle_id, payload)
    return CycleResponse.model_validate(cycle)


@router.delete(
    "/cycles/{cycle_id}", response_model=MessageResponse, summary="Delete a cycle"
)
async def delete_cycle(
    cycle_id: uuid.UUID, user_id: CurrentUserId, service: CycleServiceDep
) -> MessageResponse:
    await service.delete_cycle(user_id, cycle_id)
    return MessageResponse(message="Cycle deleted.")


# ----------------------------------------------------------------- symptoms
@router.post(
    "/symptoms",
    response_model=SymptomResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Log a symptom",
)
async def log_symptom(
    payload: SymptomCreate, user_id: CurrentUserId, service: SymptomServiceDep
) -> SymptomResponse:
    """Record a symptom and severity. Re-logging the same day updates it."""
    entry = await service.log(user_id, payload)
    return SymptomResponse.model_validate(entry)


@router.get(
    "/symptoms", response_model=list[SymptomResponse], summary="List symptom logs"
)
async def list_symptoms(
    user_id: CurrentUserId, service: SymptomServiceDep, window: DateWindowDep
) -> list[SymptomResponse]:
    entries = await service.list_range(user_id, window.start, window.end)
    return [SymptomResponse.model_validate(e) for e in entries]


@router.get(
    "/symptoms/summary",
    response_model=list[SymptomSummary],
    summary="Symptom frequency and trend",
)
async def symptom_summary(
    user_id: CurrentUserId,
    service: SymptomServiceDep,
    days: int = Query(90, ge=7, le=365),
) -> list[SymptomSummary]:
    """Per-symptom occurrence, mean severity and direction of travel.

    The trend compares the two halves of the window, so it describes whether a
    symptom is genuinely worsening rather than reacting to a single bad day.
    """
    return await service.summary(user_id, days)


@router.delete(
    "/symptoms/{symptom_id}",
    response_model=MessageResponse,
    summary="Delete a symptom log",
)
async def delete_symptom(
    symptom_id: uuid.UUID, user_id: CurrentUserId, service: SymptomServiceDep
) -> MessageResponse:
    await service.delete(user_id, symptom_id)
    return MessageResponse(message="Symptom log deleted.")


# --------------------------------------------------------------- prediction
@router.post(
    "/predict",
    response_model=RiskAssessmentResponse,
    summary="Run a PCOS risk assessment",
    responses={
        503: {
            "model": ErrorResponse,
            "description": "The model artifact is not loaded on this server.",
        }
    },
)
async def predict_risk(
    payload: RiskAssessmentRequest,
    user_id: CurrentUserId,
    service: PredictionServiceDep,
) -> RiskAssessmentResponse:
    """Score a symptom questionnaire and explain the result.

    **This is a screening estimate, not a diagnosis.** The response includes a
    SHAP-derived breakdown of which answers moved the estimate and by how much,
    a calibrated confidence, and recommendations targeted only at the factors
    the user can actually change.

    Scores are clamped to [0.02, 0.97]: no questionnaire is ever certain, and
    reporting 100% would be indefensible.
    """
    prediction = await service.assess(user_id, payload)
    return RiskAssessmentResponse(**service.response_payload(prediction))


@router.get(
    "/predict/latest",
    response_model=RiskAssessmentResponse,
    summary="Most recent assessment",
    responses={404: {"model": ErrorResponse}},
)
async def latest_prediction(
    user_id: CurrentUserId, service: PredictionServiceDep
) -> RiskAssessmentResponse:
    prediction = await service.latest(user_id)
    return RiskAssessmentResponse(**service.response_payload(prediction))


@router.get(
    "/predict/history",
    response_model=list[PredictionHistoryItem],
    summary="Assessment history",
)
async def prediction_history(
    user_id: CurrentUserId,
    service: PredictionServiceDep,
    limit: int = Query(20, ge=1, le=100),
) -> list[PredictionHistoryItem]:
    """Past assessments, so a user can see their estimate change over time."""
    rows = await service.history(user_id, limit)
    return [PredictionHistoryItem.model_validate(r) for r in rows]


@router.get(
    "/predict/model-info",
    response_model=ModelMetadata,
    summary="Model provenance and metrics",
    responses={503: {"model": ErrorResponse}},
)
async def model_info() -> ModelMetadata:
    """Which model is serving, how it scored, and what it was trained on.

    Exposed deliberately. A health-adjacent prediction that a user cannot
    interrogate is one they have no reason to trust.
    """
    predictor = get_predictor()
    return ModelMetadata(**predictor.metadata())


@router.get(
    "/predict/model-curves",
    summary="ROC curve, confusion matrix and global importance",
    responses={503: {"model": ErrorResponse}},
)
async def model_curves() -> dict:
    """Evaluation artifacts for the model-transparency page."""
    predictor = get_predictor()
    return {
        "roc_curve": predictor.roc_curve(),
        "confusion_matrix": predictor.confusion_matrix(),
        "permutation_importance": predictor.global_importance(),
    }
