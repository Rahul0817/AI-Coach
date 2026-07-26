"""User, profile and account-lifecycle queries."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models.user import Profile, User
from app.repositories.base import BaseRepository

#: Lock an account after this many consecutive failures…
MAX_FAILED_LOGINS = 5
#: …for this long. Long enough to defeat online guessing, short enough that a
#: legitimate user is not locked out for the rest of the day.
LOCKOUT_DURATION = timedelta(minutes=15)


class UserRepository(BaseRepository[User]):
    model = User

    async def get_by_email(self, email: str) -> User | None:
        """Case-insensitive lookup. Emails are stored lowercased on write."""
        stmt = select(User).where(User.email == email.lower().strip())
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_with_profile(self, user_id: uuid.UUID) -> User | None:
        stmt = (
            select(User)
            .where(User.id == user_id)
            .options(selectinload(User.profile))
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def email_taken(self, email: str) -> bool:
        return await self.exists(email=email.lower().strip())

    async def record_successful_login(self, user: User) -> User:
        user.last_login_at = datetime.now(timezone.utc)
        user.failed_login_count = 0
        user.locked_until = None
        await self.session.flush()
        return user

    async def record_failed_login(self, user: User) -> User:
        """Increment the failure counter and lock the account at the threshold."""
        user.failed_login_count += 1
        if user.failed_login_count >= MAX_FAILED_LOGINS:
            user.locked_until = datetime.now(timezone.utc) + LOCKOUT_DURATION
        await self.session.flush()
        return user

    @staticmethod
    def is_locked(user: User) -> bool:
        if user.locked_until is None:
            return False
        return user.locked_until > datetime.now(timezone.utc)


class ProfileRepository(BaseRepository[Profile]):
    model = Profile

    async def get_by_user(self, user_id: uuid.UUID) -> Profile | None:
        stmt = select(Profile).where(Profile.user_id == user_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_or_create(self, user_id: uuid.UUID) -> Profile:
        """Profiles are created lazily so registration stays a single insert."""
        profile = await self.get_by_user(user_id)
        if profile is None:
            profile = await self.create(user_id=user_id)
        return profile

    async def merge_ai_memory(
        self, profile: Profile, updates: dict
    ) -> Profile:
        """Merge newly-learned facts into long-term memory.

        A fresh dict is assigned rather than mutated in place: SQLAlchemy's
        change detection does not see in-place mutation of a JSONB dict, so
        mutating would silently drop the update.
        """
        merged = dict(profile.ai_memory or {})
        merged.update({k: v for k, v in updates.items() if v is not None})
        profile.ai_memory = merged
        await self.session.flush()
        return profile
