"""Cross-cutting HTTP middleware.

Three concerns are handled once here so no route handler has to think about
them: request correlation, security headers, and rate limiting.
"""

from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.config import settings
from app.core.logging import get_logger, request_id_ctx, user_id_ctx
from app.core.redis_client import get_cache
from app.core.security import decode_token

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, bind logging context, and emit an access log."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        rid_token = request_id_ctx.set(request_id)
        uid_token = user_id_ctx.set("-")
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise
        finally:
            request_id_ctx.reset(rid_token)
            user_id_ctx.reset(uid_token)

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{duration_ms:.2f}"

        # Health checks are polled constantly; logging them buries real traffic.
        if request.url.path not in {"/health", "/health/live", "/metrics"}:
            logger.info(
                "request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach defensive headers to every response."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), camera=(), microphone=(self)"
        )
        if settings.is_production:
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window rate limiter keyed by user id, falling back to client IP.

    Chat and other AI routes get a tighter budget because each call costs real
    money and GPU time; everything else shares the general budget. Keying on the
    authenticated subject when available prevents one user behind a shared NAT
    from exhausting everybody else's quota.
    """

    #: Path fragments billed against the expensive-route budget.
    AI_PATH_MARKERS = ("/chat", "/voice", "/vision", "/reports/analyze", "/predict")
    #: Never rate limited — these must stay reachable for orchestrators.
    EXEMPT_PATHS = frozenset({"/health", "/health/live", "/health/ready", "/docs",
                              "/openapi.json", "/redoc"})

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if not settings.rate_limit_enabled or request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        if request.method == "OPTIONS":  # CORS preflight
            return await call_next(request)

        identity = self._identify(request)
        is_ai_route = any(m in request.url.path for m in self.AI_PATH_MARKERS)
        if is_ai_route:
            limit = settings.chat_rate_limit_requests
            window = settings.chat_rate_limit_window_seconds
            bucket = "ai"
        else:
            limit = settings.rate_limit_requests
            window = settings.rate_limit_window_seconds
            bucket = "general"

        cache = get_cache()
        key = f"ratelimit:{bucket}:{identity}:{int(time.time() // window)}"
        allowed, remaining, retry_after = await cache.hit_rate_limit(key, limit, window)

        if not allowed:
            logger.warning(
                "rate limit exceeded",
                extra={"identity": identity, "bucket": bucket, "path": request.url.path},
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "message": (
                        "You have made too many requests. "
                        f"Please retry in {retry_after} seconds."
                    ),
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    @staticmethod
    def _identify(request: Request) -> str:
        """Prefer the JWT subject; fall back to the originating IP address."""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            try:
                payload = decode_token(auth[7:], expected_type="access")
                return f"user:{payload['sub']}"
            except Exception:
                pass  # Unauthenticated/invalid — fall through to IP keying.

        # X-Forwarded-For is only trustworthy behind our own proxy, and we take
        # the left-most entry, which the edge proxy sets.
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        return f"ip:{request.client.host if request.client else 'unknown'}"
