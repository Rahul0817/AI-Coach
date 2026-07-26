"""Health checks and service metadata.

Two distinct probes, because Kubernetes needs to tell two different questions
apart:

``/health/live`` — is the process alive? Never touches a dependency, so a
database blip cannot cause the orchestrator to kill an otherwise healthy pod.

``/health/ready`` — can it serve traffic? Checks every dependency and returns
503 when one is down, so the load balancer stops sending requests.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.ai.llm.factory import get_embedding_provider, get_llm_provider
from app.ai.multimodal.ocr import tesseract_available
from app.ai.rag.retriever import get_retriever
from app.core.config import settings
from app.core.database import get_engine
from app.core.redis_client import get_cache
from app.ml.predictor import get_predictor
from app.services.storage_service import get_storage

router = APIRouter(tags=["System"])

VERSION = "1.0.0"


@router.get("/health/live", summary="Liveness probe")
async def liveness() -> dict:
    """Confirm the process is running. Deliberately dependency-free."""
    return {"status": "alive", "version": VERSION}


@router.get("/health/ready", summary="Readiness probe")
async def readiness(response: Response) -> dict:
    """Check every dependency and report per-component status."""
    checks: dict[str, str] = {}
    healthy = True

    # --- database ---
    try:
        from sqlalchemy import text

        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"unavailable: {type(exc).__name__}"
        healthy = False  # the app genuinely cannot serve without Postgres

    # --- cache ---
    cache = get_cache()
    if await cache.healthy():
        checks["cache"] = "redis" if cache.is_distributed else "in-process fallback"
    else:
        checks["cache"] = "unavailable"

    # --- model ---
    predictor = get_predictor()
    checks["ml_model"] = (
        f"loaded ({predictor.metadata()['model_name']} "
        f"v{predictor.metadata()['model_version']})"
        if predictor.is_ready
        else "not loaded — run `python -m ml.train`"
    )

    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if healthy else "degraded",
        "version": VERSION,
        "environment": settings.environment,
        "checks": checks,
    }


@router.get("/health", summary="Full service status")
async def health(response: Response) -> dict:
    """Detailed status for the operator dashboard.

    Reports which optional capabilities are active, so a deployment running on
    fallbacks is visible rather than silently degraded.
    """
    base = await readiness(response)
    retriever = get_retriever()

    return {
        **base,
        "capabilities": {
            "llm_provider": get_llm_provider().name,
            "embedding_provider": get_embedding_provider().name,
            "rag": await retriever.stats(),
            "storage": get_storage().backend_name(),
            "ocr_tesseract": tesseract_available(),
            "server_voice": bool(settings.openai_api_key),
        },
    }


@router.get("/", include_in_schema=False)
async def root() -> dict:
    """Friendly landing payload for anyone hitting the API root directly."""
    return {
        "name": settings.app_name,
        "tagline": "Your Personal AI Companion for Managing PCOS",
        "version": VERSION,
        "documentation": "/docs",
        "notice": (
            "Oviora AI provides educational information and lifestyle guidance "
            "only. It does not diagnose any medical condition."
        ),
    }
