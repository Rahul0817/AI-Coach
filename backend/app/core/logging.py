"""Structured JSON logging with request correlation.

Production logs are read by machines before humans. Emitting JSON lines means
every field is queryable in CloudWatch/Loki/Datadog without regex parsing, and
the ``request_id`` propagated through a ``ContextVar`` lets an operator pull
every log line belonging to one user request — including lines written deep
inside the agent layer, which never sees the HTTP request object.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from app.core.config import settings

#: Correlation id for the in-flight request, set by RequestContextMiddleware.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
#: Authenticated user id for the in-flight request, when there is one.
user_id_ctx: ContextVar[str] = ContextVar("user_id", default="-")

_RESERVED = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render a ``LogRecord`` as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
            "user_id": user_id_ctx.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Anything passed via `logger.info("msg", extra={...})` rides along.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Colourised single-line format — far easier to scan during development."""

    _COLOURS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLOURS.get(record.levelname, "")
        stamp = datetime.now(UTC).strftime("%H:%M:%S")
        rid = request_id_ctx.get()
        prefix = f"{stamp} {colour}{record.levelname:<8}{self._RESET} {record.name}"
        if rid != "-":
            prefix += f" [{rid[:8]}]"
        line = f"{prefix} :: {record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging() -> None:
    """Install the root handler. Idempotent — safe to call from tests."""
    root = logging.getLogger()
    root.setLevel(settings.log_level)

    for existing in list(root.handlers):
        root.removeHandler(existing)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.is_production else HumanFormatter())
    root.addHandler(handler)

    # Uvicorn installs its own noisy handlers; route them through ours instead.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(noisy)
        logger.handlers = []
        logger.propagate = True

    # SQLAlchemy echoes every statement at INFO; that is too much even in dev.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor so modules do not import ``logging`` directly."""
    return logging.getLogger(name)
