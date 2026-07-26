# syntax=docker/dockerfile:1.7
# =============================================================================
# Oviora AI — backend image
#
# Multi-stage: dependencies are built in a stage that carries compilers, and
# only the installed packages are copied into the runtime image. That keeps
# gcc, headers and build caches out of production — a smaller image and a
# smaller attack surface.
# =============================================================================

# ----------------------------------------------------------------- builder --
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is needed to compile asyncpg and a few scientific wheels;
# libpq-dev is required by psycopg2, which Alembic uses.
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Dependencies are copied and installed before application code so that a
# source change does not invalidate the (slow) dependency layer.
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install -r requirements.txt

# ----------------------------------------------------------------- runtime --
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

# Runtime-only system packages:
#   libpq5      — Postgres client library
#   tesseract   — OCR for scanned lab reports
#   poppler     — rasterises scanned PDFs so Tesseract can read them
#   curl        — used by the container healthcheck
RUN apt-get update && apt-get install --no-install-recommends -y \
        libpq5 \
        tesseract-ocr \
        tesseract-ocr-eng \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Run as an unprivileged user. A container process running as root shares the
# host's root uid, so a container escape starts from the worst possible place.
RUN groupadd --system --gid 1001 oviora \
    && useradd --system --uid 1001 --gid oviora --create-home oviora

WORKDIR /app
COPY --chown=oviora:oviora . .

# Writable paths the application needs at runtime.
RUN mkdir -p /app/uploads /app/chroma_store /app/ml/artifacts \
    && chown -R oviora:oviora /app/uploads /app/chroma_store /app/ml/artifacts

USER oviora
EXPOSE 8000

# Hits readiness, not liveness: the orchestrator should only route traffic once
# dependencies are actually reachable.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/health/live || exit 1

# One worker per container. Scaling happens by adding containers, which keeps
# each one's memory footprint predictable — the ML model and the embedded
# knowledge base are loaded per process, not per worker.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*", \
     "--log-config", "/dev/null"]
