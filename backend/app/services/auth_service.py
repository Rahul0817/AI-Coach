"""Authentication and account lifecycle.

Security properties enforced here rather than in the route handler:

* **No user enumeration.** A wrong email and a wrong password produce the same
  error and take comparable time, so an attacker cannot harvest valid addresses
  from the login endpoint.
* **Lockout with backoff.** Five consecutive failures locks the account for
  fifteen minutes, which defeats online guessing without a CAPTCHA.
* **Token revocation.** Logout adds the token's ``jti`` to a Redis denylist for
  its remaining lifetime, giving stateless JWTs a real logout.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import UTC

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    AuthenticationError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
)
from app.core.logging import get_logger
from app.core.redis_client import CacheService, get_cache
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    token_expiry_seconds,
    verify_password,
)
from app.models.user import User
from app.repositories.user import ProfileRepository, UserRepository
from app.schemas.auth import TokenPair

logger = get_logger(__name__)

#: A pre-computed hash used to equalise timing when the email does not exist.
#: Without it, a missing account returns measurably faster than a wrong
#: password, which leaks account existence.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(24))


class AuthService:
    def __init__(self, session: AsyncSession, cache: CacheService | None = None) -> None:
        self.session = session
        self.users = UserRepository(session)
        self.profiles = ProfileRepository(session)
        self.cache = cache or get_cache()

    # ------------------------------------------------------------ registration
    async def register(self, email: str, password: str, full_name: str) -> User:
        normalised = email.lower().strip()
        if await self.users.email_taken(normalised):
            raise ConflictError("An account with that email already exists.")

        user = await self.users.create(
            email=normalised,
            hashed_password=hash_password(password),
            full_name=full_name.strip(),
            is_active=True,
            is_verified=False,
        )
        # A profile row always exists from registration, so every downstream
        # feature can assume it rather than defensively creating one.
        await self.profiles.create(user_id=user.id)
        await self.session.commit()

        logger.info("user registered", extra={"user_id": str(user.id)})
        return user

    # ------------------------------------------------------------------ login
    async def authenticate(self, email: str, password: str) -> User:
        user = await self.users.get_by_email(email)

        if user is None:
            # Verify against a dummy hash so the response time matches the
            # wrong-password path.
            verify_password(password, _DUMMY_HASH)
            raise AuthenticationError("Incorrect email or password.")

        if self.users.is_locked(user):
            remaining = 0
            if user.locked_until:
                from datetime import datetime

                remaining = max(
                    0,
                    int((user.locked_until - datetime.now(UTC)).total_seconds()),
                )
            raise AuthenticationError(
                f"This account is temporarily locked after repeated failed "
                f"sign-in attempts. Try again in {remaining // 60 + 1} minutes."
            )

        if not verify_password(password, user.hashed_password):
            await self.users.record_failed_login(user)
            await self.session.commit()
            raise AuthenticationError("Incorrect email or password.")

        if not user.is_active:
            raise PermissionDeniedError("This account has been deactivated.")

        await self.users.record_successful_login(user)
        await self.session.commit()
        return user

    def issue_tokens(self, user: User) -> TokenPair:
        return TokenPair(
            access_token=create_access_token(
                str(user.id), extra_claims={"email": user.email}
            ),
            refresh_token=create_refresh_token(str(user.id)),
            expires_in=settings.access_token_expire_minutes * 60,
        )

    async def login(self, email: str, password: str) -> tuple[User, TokenPair]:
        user = await self.authenticate(email, password)
        return user, self.issue_tokens(user)

    # ---------------------------------------------------------------- refresh
    async def refresh(self, refresh_token: str) -> TokenPair:
        payload = decode_token(refresh_token, expected_type="refresh")
        if await self.is_revoked(payload["jti"]):
            raise AuthenticationError("This session has been signed out.")

        user = await self.users.get(uuid.UUID(payload["sub"]))
        if user is None or not user.is_active:
            raise AuthenticationError("This account is no longer active.")

        # Refresh-token rotation: the presented token is revoked as soon as it
        # is exchanged, so a stolen refresh token is usable at most once and
        # its reuse is detectable.
        await self.revoke(payload["jti"], token_expiry_seconds(payload))
        return self.issue_tokens(user)

    # ----------------------------------------------------------- revocation
    @staticmethod
    def _denylist_key(jti: str) -> str:
        return f"auth:revoked:{jti}"

    async def revoke(self, jti: str, ttl_seconds: int) -> None:
        """Deny a token id for the remainder of its natural lifetime.

        The TTL matches the token's own expiry, so the denylist self-cleans and
        cannot grow without bound.
        """
        if ttl_seconds > 0:
            await self.cache.set_json(self._denylist_key(jti), True, ttl_seconds)

    async def is_revoked(self, jti: str) -> bool:
        return bool(await self.cache.get_json(self._denylist_key(jti)))

    async def logout(self, access_token: str) -> None:
        try:
            payload = decode_token(access_token, expected_type="access")
        except AuthenticationError:
            return  # Already invalid; nothing to revoke.
        await self.revoke(payload["jti"], token_expiry_seconds(payload))
        logger.info("user logged out", extra={"user_id": payload["sub"]})

    # ------------------------------------------------------- password change
    async def change_password(
        self, user_id: uuid.UUID, current_password: str, new_password: str
    ) -> None:
        user = await self.users.get(user_id)
        if user is None:
            raise NotFoundError("Account not found.")
        if not verify_password(current_password, user.hashed_password):
            raise AuthenticationError("Your current password is incorrect.")
        if verify_password(new_password, user.hashed_password):
            raise ConflictError("The new password must differ from the current one.")

        user.hashed_password = hash_password(new_password)
        await self.session.commit()
        logger.info("password changed", extra={"user_id": str(user_id)})

    # -------------------------------------------------------- account removal
    async def delete_account(self, user_id: uuid.UUID, password: str) -> None:
        """Permanently delete the account and every record attached to it.

        Deletion cascades at the database level (see ``UserOwnedMixin``), so a
        single ``DELETE`` removes all health data in one transaction with no
        chance of orphaned records surviving an application bug.
        """
        user = await self.users.get(user_id)
        if user is None:
            raise NotFoundError("Account not found.")
        if not verify_password(password, user.hashed_password):
            raise AuthenticationError("Password confirmation failed.")

        await self.users.delete(user)
        await self.session.commit()
        logger.warning("account deleted", extra={"user_id": str(user_id)})


async def resolve_current_user(
    session: AsyncSession, token: str, cache: CacheService
) -> User:
    """Decode a bearer token and load the corresponding active user.

    Shared by the HTTP dependency and the SSE handler, which cannot use the
    standard dependency because EventSource sends no Authorization header.
    """
    payload = decode_token(token, expected_type="access")

    if await cache.get_json(AuthService._denylist_key(payload["jti"])):
        raise AuthenticationError("This session has been signed out.")

    try:
        user_id = uuid.UUID(payload["sub"])
    except (ValueError, TypeError) as exc:
        raise AuthenticationError("Malformed token subject.") from exc

    users = UserRepository(session)
    user = await users.get_with_profile(user_id)
    if user is None:
        raise AuthenticationError("This account no longer exists.")
    if not user.is_active:
        raise PermissionDeniedError("This account has been deactivated.")
    return user
