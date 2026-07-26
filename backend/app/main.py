"""Oviora AI — FastAPI application entry point.

Run locally::

    uvicorn app.main:app --reload --port 8000

Swagger UI is at ``/docs``, ReDoc at ``/redoc``.

What happens at startup, and why in this order
----------------------------------------------
1. **Logging**, first, so every subsequent step is observable.
2. **Cache**, because rate limiting depends on it and middleware runs before
   any route.
3. **Database schema** in non-production only. Production schema changes go
   through Alembic, where they are reviewed and reversible.
4. **ML model**, loaded once into memory rather than per request.
5. **Knowledge base**, embedded and indexed once.

Steps 4 and 5 are allowed to fail without stopping the process. A missing model
should disable the prediction endpoints, not take down chat, tracking and
analytics with it — coupling every feature to one artifact is a bad
availability trade.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.router import api_router
from app.api.v1.routes import system
from app.core.config import settings
from app.core.database import dispose_engine, init_models
from app.core.exceptions import OvioraError
from app.core.logging import configure_logging, get_logger, request_id_ctx
from app.core.middleware import (
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from app.core.redis_client import close_cache, init_cache

logger = get_logger(__name__)

DESCRIPTION = """
**Oviora AI — Your Personal AI Companion for Managing PCOS**

An AI copilot for polycystic ovary syndrome: conversational guidance from eight
specialist agents grounded in a curated knowledge base, an explainable risk
model, lab-report parsing, meal-photo analysis, and a full tracking dashboard.

---

### ⚠️ Medical notice

Oviora **does not diagnose any condition**. It provides educational
information, lifestyle guidance and a screening-level risk *estimate*. Every
health-related response carries a disclaimer directing you to a qualified
clinician, and that disclaimer is enforced in code rather than left to the
model.

---

### How it fits together

* **Multi-agent chat** — a router dispatches each question to one of eight
  specialists. Routing runs a cheap lexical classifier first and only consults
  the language model when the wording is genuinely ambiguous.
* **RAG** — answers are grounded in a curated PCOS corpus using hybrid dense +
  BM25 retrieval fused by Reciprocal Rank Fusion. When retrieval finds nothing
  relevant, the assistant says so rather than inventing an answer.
* **Memory** — a verbatim window in Redis, a rolling summary in Postgres, and
  structured long-term facts on the profile, so "I'm 22" persists into a later
  "suggest my breakfast".
* **Explainable ML** — every risk score ships with a SHAP breakdown of which
  answers moved it and by how much.

### Authentication

Call `POST /api/v1/auth/login`, then send `Authorization: Bearer <access_token>`.
Access tokens last 30 minutes; refresh them at `POST /api/v1/auth/refresh`.
"""

TAGS_METADATA = [
    {"name": "System", "description": "Health checks and service capabilities."},
    {
        "name": "Authentication",
        "description": "Registration, sign-in and account lifecycle.",
    },
    {
        "name": "Profile & Settings",
        "description": "Health profile, data export and AI memory.",
    },
    {
        "name": "AI Chat",
        "description": "Multi-agent chat, streaming and conversation history.",
    },
    {
        "name": "Cycle, Symptoms & Risk",
        "description": "Cycle tracking, symptom logs and the explainable risk model.",
    },
    {
        "name": "Daily Tracking",
        "description": "Habits, meals, workouts, water, sleep, weight, mood and AI plan generation.",
    },
    {
        "name": "Dashboard & Analytics",
        "description": "Aggregated metrics, trends, insights and notifications.",
    },
    {
        "name": "Reports, Vision & Voice",
        "description": "Blood-report OCR, meal-photo analysis and speech.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop shared resources exactly once per process."""
    configure_logging()
    started = time.perf_counter()
    logger.info(
        "starting up",
        extra={"environment": settings.environment, "app": settings.app_name},
    )

    await init_cache()

    # Alembic owns the schema in production; this is for local and CI runs.
    if not settings.is_production:
        try:
            await init_models()
        except Exception as exc:
            logger.error(
                "could not create the database schema; the API will start but "
                "database-backed routes will fail",
                extra={"error": str(exc)},
            )

    # --- optional capability: risk prediction ---
    from app.ml.predictor import RiskPredictor

    if not RiskPredictor.instance().load():
        logger.warning("prediction endpoints will return 503 until the model is trained")

    # --- optional capability: retrieval ---
    try:
        from app.ai.rag.retriever import get_retriever

        count = await get_retriever().ingest()
        logger.info("knowledge base ready", extra={"chunks": count})
    except Exception as exc:
        logger.error(
            "knowledge base ingestion failed; chat will run ungrounded",
            extra={"error": str(exc)},
        )

    logger.info(
        "startup complete",
        extra={"duration_ms": round((time.perf_counter() - started) * 1000, 1)},
    )
    yield

    logger.info("shutting down")
    await close_cache()
    await dispose_engine()


app = FastAPI(
    title=settings.app_name,
    description=DESCRIPTION,
    version=system.VERSION,
    lifespan=lifespan,
    openapi_tags=TAGS_METADATA,
    # Interactive docs are disabled in production: they describe every endpoint
    # and payload shape, which is free reconnaissance for an attacker.
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
    contact={"name": "Oviora AI", "url": "https://github.com/Rahul0817/AI-Coach"},
    license_info={"name": "MIT"},
)

# ---------------------------------------------------------------- middleware
# Starlette applies middleware in reverse registration order, so the last one
# added is the outermost. Request context is registered last so that it wraps
# everything and a request id exists even for a rate-limit rejection.
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestContextMiddleware)

app.add_middleware(
    CORSMiddleware,
    # An explicit origin list, never "*". With credentials enabled a wildcard
    # is both forbidden by the spec and a genuine security hole.
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "X-Response-Time-ms"],
    max_age=3600,
)


# ------------------------------------------------------------ error handlers
@app.exception_handler(OvioraError)
async def handle_domain_error(request: Request, exc: OvioraError) -> JSONResponse:
    """Translate a domain exception into its HTTP representation.

    This single handler is why the service layer never imports HTTPException:
    services raise meaning, and exactly one place decides what status code that
    meaning maps to.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={**exc.to_dict(), "request_id": request_id_ctx.get()},
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Reshape FastAPI's validation errors into readable field messages."""
    fields: dict[str, str] = {}
    for error in exc.errors():
        # Drop the leading "body"/"query" segment — the client knows where it
        # put the field, it needs to know which one is wrong.
        location = [str(part) for part in error["loc"] if part not in {"body", "query"}]
        fields[".".join(location) or "request"] = error["msg"]

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "Some fields did not pass validation.",
            "details": fields,
            "request_id": request_id_ctx.get(),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Give framework-raised errors (404, 405) the same body shape as ours."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"http_{exc.status_code}",
            "message": str(exc.detail),
            "request_id": request_id_ctx.get(),
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Last line of defence for anything unhandled.

    The full traceback goes to the logs; the client receives a request id and
    nothing else. Leaking a stack trace to a browser exposes file paths,
    library versions and sometimes credentials.
    """
    logger.exception(
        "unhandled exception",
        extra={"path": request.url.path, "method": request.method},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "message": (
                "Something went wrong on our end. Quote the request id below "
                "if you contact support."
            ),
            "request_id": request_id_ctx.get(),
        },
    )


# --------------------------------------------------------------------- routes
app.include_router(system.router)
app.include_router(api_router, prefix=settings.api_v1_prefix)

# Locally-stored uploads are served directly; with Cloudinary configured the
# files never touch this filesystem and this mount stays empty.
_upload_dir = Path(settings.local_upload_dir)
if not settings.cloudinary_enabled:
    _upload_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=str(_upload_dir)), name="uploads")


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=not settings.is_production,
        log_config=None,  # our own logging config owns formatting
    )
