"""Password hashing and JWT issuance/verification.

Security choices worth defending in a review:

* **bcrypt, not SHA-256.** Password hashes must be *slow*. bcrypt has a tunable
  work factor and a per-password salt, so identical passwords produce different
  hashes and offline cracking stays expensive.
* **Two token types.** Short-lived access tokens (30 min) limit the blast radius
  of a leaked token; long-lived refresh tokens (14 days) keep users signed in.
  The ``type`` claim is verified so a refresh token can never be replayed as an
  access token.
* **``jti`` on every token.** A unique token id lets us revoke individual tokens
  through a Redis denylist on logout, which plain stateless JWT cannot do.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.core.exceptions import AuthenticationError

TokenType = Literal["access", "refresh"]

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=settings.bcrypt_rounds,
)

# bcrypt silently truncates beyond 72 bytes; reject long inputs explicitly so
# "password" and "password + 200 chars" can never be treated as equal.
MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    """Return a salted bcrypt hash for ``password``."""
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValueError("Password exceeds the maximum supported length.")
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time-ish comparison of a candidate password against a hash."""
    if len(plain.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return False
    try:
        return pwd_context.verify(plain, hashed)
    except ValueError:
        # Malformed hash in the database — treat as a failed login, not a 500.
        return False


def _create_token(
    subject: str,
    token_type: TokenType,
    expires_delta: timedelta,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "jti": uuid.uuid4().hex,
        "iss": settings.app_name,
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def create_access_token(
    subject: str, extra_claims: dict[str, Any] | None = None
) -> str:
    return _create_token(
        subject,
        "access",
        timedelta(minutes=settings.access_token_expire_minutes),
        extra_claims,
    )


def create_refresh_token(subject: str) -> str:
    return _create_token(
        subject, "refresh", timedelta(days=settings.refresh_token_expire_days)
    )


def decode_token(token: str, expected_type: TokenType | None = None) -> dict[str, Any]:
    """Decode and validate a JWT.

    Raises :class:`AuthenticationError` for anything unusable — expired,
    tampered, wrong issuer, or the wrong token type — so callers never have to
    distinguish jose's exception zoo.
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
            issuer=settings.app_name,
        )
    except JWTError as exc:
        raise AuthenticationError("Token is invalid or has expired.") from exc

    if expected_type and payload.get("type") != expected_type:
        raise AuthenticationError(
            f"Expected a {expected_type} token but received "
            f"{payload.get('type', 'unknown')}."
        )
    if not payload.get("sub"):
        raise AuthenticationError("Token is missing a subject claim.")
    return payload


def token_expiry_seconds(payload: dict[str, Any]) -> int:
    """Seconds remaining before ``payload`` expires (never negative)."""
    exp = payload.get("exp")
    if not exp:
        return 0
    return max(0, int(exp - datetime.now(timezone.utc).timestamp()))


def generate_reset_token() -> tuple[str, str]:
    """Create a password-reset token: ``(plaintext, sha256_digest)``.

    Only the digest is stored, so a database leak does not hand an attacker
    working reset links.
    """
    raw = secrets.token_urlsafe(48)
    return raw, hashlib.sha256(raw.encode()).hexdigest()


def hash_reset_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()
