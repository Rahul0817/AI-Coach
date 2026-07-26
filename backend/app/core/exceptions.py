"""Domain exception hierarchy.

Services raise these instead of ``HTTPException``. Keeping HTTP semantics out
of the service layer is what lets the same services be reused by a CLI, a
worker, or a gRPC surface later — the API layer is the only place that knows
about status codes, and it translates via a single registered handler.
"""

from __future__ import annotations

from typing import Any


class OvioraError(Exception):
    """Base class for every error Oviora raises deliberately."""

    status_code: int = 500
    error_code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.__class__.message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": self.error_code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class NotFoundError(OvioraError):
    status_code = 404
    error_code = "not_found"
    message = "The requested resource does not exist."


class ConflictError(OvioraError):
    status_code = 409
    error_code = "conflict"
    message = "The resource already exists or conflicts with current state."


class ValidationError(OvioraError):
    status_code = 422
    error_code = "validation_error"
    message = "The submitted data failed validation."


class AuthenticationError(OvioraError):
    status_code = 401
    error_code = "authentication_failed"
    message = "Invalid or missing credentials."


class PermissionDeniedError(OvioraError):
    status_code = 403
    error_code = "permission_denied"
    message = "You do not have access to this resource."


class RateLimitError(OvioraError):
    status_code = 429
    error_code = "rate_limited"
    message = "Too many requests. Please slow down."


class ExternalServiceError(OvioraError):
    status_code = 502
    error_code = "external_service_error"
    message = "An upstream service failed."


class ModelUnavailableError(OvioraError):
    status_code = 503
    error_code = "model_unavailable"
    message = "The prediction model is not loaded. Train it before serving."


class UnsupportedMediaError(OvioraError):
    status_code = 415
    error_code = "unsupported_media"
    message = "That file type is not supported."
