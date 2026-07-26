"""FastAPI dependencies: authentication, pagination and service wiring.

Every protected route depends on :func:`get_current_user`. Because that is a
single choke point, a route cannot accidentally ship unauthenticated — omitting
the dependency is visible in the signature, whereas a forgotten ``if not
user:`` check inside a handler is not.

Services are constructed per request with the request-scoped session injected.
That is dependency injection in the form FastAPI supports natively: no global
service registry, no container, and tests override a single provider.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import AuthenticationError
from app.core.logging import user_id_ctx
from app.core.redis_client import CacheService, get_cache
from app.models.user import User
from app.services.analytics_service import AnalyticsService
from app.services.auth_service import AuthService, resolve_current_user
from app.services.chat_service import ChatService
from app.services.health_service import CycleService, PredictionService, SymptomService
from app.services.notification_service import NotificationService
from app.services.plan_service import PlanService
from app.services.report_service import ReportService
from app.services.tracking_service import (
    DailyTrackerService,
    HabitService,
    MealService,
    WorkoutService,
)
from app.services.user_service import UserService

# ``auto_error=False`` so a missing header raises our own AuthenticationError
# with a consistent body, rather than FastAPI's differently-shaped 403.
bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")

DbSession = Annotated[AsyncSession, Depends(get_db)]
Cache = Annotated[CacheService, Depends(get_cache)]


async def get_current_user(
    request: Request,
    session: DbSession,
    cache: Cache,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> User:
    """Resolve the authenticated user from the bearer token."""
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Authentication credentials were not provided.")

    user = await resolve_current_user(session, credentials.credentials, cache)

    # Bind the user to the logging context so every log line emitted for the
    # rest of this request is attributable without threading the id through.
    user_id_ctx.set(str(user.id))
    request.state.user_id = user.id
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_current_user_id(user: CurrentUser) -> uuid.UUID:
    """Just the id, for routes that never touch the rest of the user row."""
    return user.id


CurrentUserId = Annotated[uuid.UUID, Depends(get_current_user_id)]


class Pagination:
    """Shared list-endpoint query parameters."""

    def __init__(
        self,
        page: Annotated[int, Query(ge=1, le=10_000, description="1-indexed page.")] = 1,
        page_size: Annotated[
            int, Query(ge=1, le=100, description="Items per page (max 100).")
        ] = 20,
    ) -> None:
        self.page = page
        self.page_size = page_size
        self.offset = (page - 1) * page_size
        self.limit = page_size


PaginationDep = Annotated[Pagination, Depends(Pagination)]


class DateWindow:
    """A bounded date range for analytics and history endpoints."""

    def __init__(
        self,
        days: Annotated[
            int,
            Query(ge=1, le=365, description="Size of the window, ending today."),
        ] = 30,
    ) -> None:
        from datetime import date, timedelta

        self.days = days
        self.end = date.today()
        self.start = self.end - timedelta(days=days - 1)


DateWindowDep = Annotated[DateWindow, Depends(DateWindow)]


# --------------------------------------------------------------- services --
# Each provider is a thin factory. Declaring them as dependencies (rather than
# constructing services inside handlers) is what lets a test swap any service
# for a fake with `app.dependency_overrides`.


def get_auth_service(session: DbSession, cache: Cache) -> AuthService:
    return AuthService(session, cache)


def get_user_service(session: DbSession) -> UserService:
    return UserService(session)


def get_chat_service(session: DbSession) -> ChatService:
    return ChatService(session)


def get_cycle_service(session: DbSession) -> CycleService:
    return CycleService(session)


def get_symptom_service(session: DbSession) -> SymptomService:
    return SymptomService(session)


def get_prediction_service(session: DbSession) -> PredictionService:
    return PredictionService(session)


def get_habit_service(session: DbSession) -> HabitService:
    return HabitService(session)


def get_meal_service(session: DbSession) -> MealService:
    return MealService(session)


def get_workout_service(session: DbSession) -> WorkoutService:
    return WorkoutService(session)


def get_daily_tracker_service(session: DbSession) -> DailyTrackerService:
    return DailyTrackerService(session)


def get_analytics_service(session: DbSession) -> AnalyticsService:
    return AnalyticsService(session)


def get_report_service(session: DbSession) -> ReportService:
    return ReportService(session)


def get_notification_service(session: DbSession) -> NotificationService:
    return NotificationService(session)


def get_plan_service(session: DbSession) -> PlanService:
    return PlanService(session)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
UserServiceDep = Annotated[UserService, Depends(get_user_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
CycleServiceDep = Annotated[CycleService, Depends(get_cycle_service)]
SymptomServiceDep = Annotated[SymptomService, Depends(get_symptom_service)]
PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
HabitServiceDep = Annotated[HabitService, Depends(get_habit_service)]
MealServiceDep = Annotated[MealService, Depends(get_meal_service)]
WorkoutServiceDep = Annotated[WorkoutService, Depends(get_workout_service)]
DailyTrackerServiceDep = Annotated[
    DailyTrackerService, Depends(get_daily_tracker_service)
]
AnalyticsServiceDep = Annotated[AnalyticsService, Depends(get_analytics_service)]
ReportServiceDep = Annotated[ReportService, Depends(get_report_service)]
NotificationServiceDep = Annotated[
    NotificationService, Depends(get_notification_service)
]
PlanServiceDep = Annotated[PlanService, Depends(get_plan_service)]


def extract_bearer_token(request: Request) -> str:
    """Pull the raw token out of the Authorization header.

    Needed by the logout route, which must revoke the *specific* token it was
    presented with rather than the decoded user.
    """
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise AuthenticationError("A bearer token is required.")
    return header[7:]
