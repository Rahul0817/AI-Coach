"""Upload endpoints: blood reports, food photos and voice."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, Query, UploadFile, status
from fastapi.responses import Response

from app.ai.multimodal import voice as voice_module
from app.ai.multimodal.vision import (
    MAX_IMAGE_BYTES,
    SUPPORTED_IMAGE_TYPES,
    build_analysis_payload,
    get_vision_provider,
)
from app.api.deps import CurrentUserId, ReportServiceDep
from app.core.exceptions import UnsupportedMediaError
from app.schemas.common import ErrorResponse, MessageResponse
from app.schemas.report import (
    BloodReportDetail,
    BloodReportResponse,
    FoodAnalysisResponse,
    SpeechRequest,
    VoiceTranscription,
)
from app.services.storage_service import get_storage

router = APIRouter(tags=["Reports, Vision & Voice"])

UPLOAD_ERRORS = {
    415: {"model": ErrorResponse, "description": "Unsupported or mismatched file type"},
    413: {"model": ErrorResponse, "description": "File too large"},
}


# ------------------------------------------------------------ blood reports
@router.post(
    "/reports",
    response_model=BloodReportDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and parse a blood report",
    responses=UPLOAD_ERRORS,
)
async def upload_report(
    user_id: CurrentUserId,
    service: ReportServiceDep,
    file: UploadFile = File(..., description="PDF or image of a lab report."),
) -> BloodReportDetail:
    """Extract biomarkers from a lab report and explain them.

    Digital PDFs are read from their embedded text layer, which is both faster
    and more accurate than rasterising and running OCR; scans fall back to
    Tesseract. Reference ranges printed on the report always take precedence
    over built-in ones, because ranges are assay-specific.

    The declared content type is verified against the file's magic bytes, so a
    renamed file cannot slip through.
    """
    data = await file.read()
    report = await service.upload_and_process(
        user_id, data, file.filename or "report", file.content_type or ""
    )
    return BloodReportDetail.model_validate(report)


@router.get(
    "/reports",
    response_model=list[BloodReportResponse],
    summary="List uploaded reports",
)
async def list_reports(
    user_id: CurrentUserId,
    service: ReportServiceDep,
    limit: int = Query(20, ge=1, le=100),
) -> list[BloodReportResponse]:
    rows = await service.list_reports(user_id, limit)
    return [BloodReportResponse.model_validate(r) for r in rows]


@router.get(
    "/reports/{report_id}",
    response_model=BloodReportDetail,
    summary="Get a report with its biomarkers",
    responses={404: {"model": ErrorResponse}},
)
async def get_report(
    report_id: uuid.UUID, user_id: CurrentUserId, service: ReportServiceDep
) -> BloodReportDetail:
    report = await service.get_report(user_id, report_id)
    return BloodReportDetail.model_validate(report)


@router.post(
    "/reports/{report_id}/explain",
    summary="Explain a report with the Blood Report Analyzer",
    responses={404: {"model": ErrorResponse}},
)
async def explain_report(
    report_id: uuid.UUID, user_id: CurrentUserId, service: ReportServiceDep
) -> dict:
    """Run the specialist agent over the parsed values.

    The agent explains what each marker measures and why it was ordered, and
    finishes with questions to ask a clinician. It will not tell the user
    whether they have a condition — that boundary is enforced in the prompt and
    checked by the safety layer.
    """
    return await service.explain(user_id, report_id)


@router.get(
    "/reports/biomarkers/{name}/trend",
    summary="Track one biomarker across reports",
)
async def biomarker_trend(
    name: str, user_id: CurrentUserId, service: ReportServiceDep
) -> list[dict]:
    """History of a single marker across every report uploaded."""
    return await service.biomarker_trend(user_id, name)


@router.delete(
    "/reports/{report_id}",
    response_model=MessageResponse,
    summary="Delete a report",
)
async def delete_report(
    report_id: uuid.UUID, user_id: CurrentUserId, service: ReportServiceDep
) -> MessageResponse:
    await service.delete_report(user_id, report_id)
    return MessageResponse(message="Report and its extracted data deleted.")


# --------------------------------------------------------------- food vision
@router.post(
    "/vision/food",
    response_model=FoodAnalysisResponse,
    summary="Analyse a meal photo",
    responses=UPLOAD_ERRORS,
)
async def analyse_food(
    user_id: CurrentUserId,
    file: UploadFile = File(..., description="Photo of a meal."),
    caption: str | None = Form(
        None,
        description=(
            "Optional description of the plate. Required when no vision model "
            "is configured — see `analysis_method` in the response."
        ),
    ),
) -> FoodAnalysisResponse:
    """Identify foods, estimate nutrition and score the meal for PCOS.

    When a vision model is configured it analyses the image directly. When one
    is not, the response says so plainly in ``analysis_method`` and the
    breakdown is built from ``caption`` instead — a system with no vision
    capability claiming to "see grilled chicken" would be lying to the user
    about their own health data.

    The nutrition maths, glycaemic-load scoring and swap suggestions are
    identical on both paths.
    """
    data = await file.read()
    content_type = file.content_type or ""

    storage = get_storage()
    if content_type not in SUPPORTED_IMAGE_TYPES:
        raise UnsupportedMediaError(
            f"'{content_type}' is not a supported image format. Use JPEG, PNG or WebP."
        )
    if len(data) > MAX_IMAGE_BYTES:
        raise UnsupportedMediaError("Image exceeds the 8MB limit.")
    storage.validate(data, content_type, SUPPORTED_IMAGE_TYPES)

    image_url: str | None = None
    try:
        stored = await storage.upload(data, content_type, "meals", user_id)
        image_url = stored.url
    except Exception:  # noqa: S110
        # Storage is not on the critical path — the analysis is what the user
        # asked for, and losing the thumbnail should not fail the request.
        pass

    provider = get_vision_provider()
    analysis = await provider.analyse(data, content_type, caption)
    return FoodAnalysisResponse(**build_analysis_payload(analysis, image_url))


# --------------------------------------------------------------------- voice
@router.get("/voice/capabilities", summary="Which voice paths are available")
async def voice_capabilities() -> dict:
    """Report the available speech paths so the client need not probe.

    The default is browser-native Web Speech: it runs on-device, costs nothing,
    and no audio ever leaves the user's machine — which matters for a health
    product. Server transcription exists for browsers without support.
    """
    return voice_module.capabilities()


@router.post(
    "/voice/transcribe",
    response_model=VoiceTranscription,
    summary="Transcribe uploaded audio",
    responses={
        415: {"model": ErrorResponse},
        502: {
            "model": ErrorResponse,
            "description": "Server transcription is not configured.",
        },
    },
)
async def transcribe_audio(
    user_id: CurrentUserId,
    file: UploadFile = File(..., description="Recorded audio."),
    language: str = Form("en"),
) -> VoiceTranscription:
    """Transcribe audio server-side via Whisper.

    Prefer the browser path where available — it is faster, free, and keeps the
    recording on-device.
    """
    data = await file.read()
    result = await voice_module.transcribe(
        data, file.filename or "audio.webm", file.content_type or "", language
    )
    return VoiceTranscription(
        text=result.text,
        duration_seconds=result.duration_seconds,
        language=result.language,
        confidence=result.confidence,
    )


@router.post(
    "/voice/speak",
    summary="Synthesise speech from text",
    response_class=Response,
    responses={
        200: {"content": {"audio/mpeg": {}}, "description": "MP3 audio."},
        502: {"model": ErrorResponse},
    },
)
async def synthesise_speech(payload: SpeechRequest, user_id: CurrentUserId) -> Response:
    """Render an assistant reply to audio.

    Markdown is stripped first — otherwise the synthesiser articulates
    "asterisk asterisk Important asterisk asterisk" and reads the horizontal
    rule before the disclaimer as a run of dashes.
    """
    audio = await voice_module.synthesise(payload.text, payload.voice, payload.speed)
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'inline; filename="oviora-speech.mp3"'},
    )
