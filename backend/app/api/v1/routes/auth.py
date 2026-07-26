"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, status

from app.api.deps import (
    AuthServiceDep,
    CurrentUser,
    UserServiceDep,
    extract_bearer_token,
)
from app.schemas.auth import (
    ChangePasswordRequest,
    DeleteAccountRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenPair,
)
from app.schemas.common import ErrorResponse, MessageResponse
from app.schemas.user import UserResponse, UserWithProfile
from app.services.notification_service import NotificationService

router = APIRouter(prefix="/auth", tags=["Authentication"])

COMMON_ERRORS = {
    401: {"model": ErrorResponse, "description": "Invalid or missing credentials"},
    422: {"model": ErrorResponse, "description": "Validation failed"},
}


@router.post(
    "/register",
    response_model=TokenPair,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account",
    responses={409: {"model": ErrorResponse, "description": "Email already registered"}},
)
async def register(payload: RegisterRequest, service: AuthServiceDep) -> TokenPair:
    """Register a new account and return a token pair.

    Tokens are issued immediately so the client does not have to follow
    registration with a second login round trip.
    """
    user = await service.register(payload.email, payload.password, payload.full_name)
    await NotificationService(service.session).welcome(user.id, user.full_name)
    return service.issue_tokens(user)


@router.post(
    "/login",
    response_model=TokenPair,
    summary="Sign in",
    responses=COMMON_ERRORS,
)
async def login(payload: LoginRequest, service: AuthServiceDep) -> TokenPair:
    """Exchange email and password for an access/refresh token pair.

    A wrong email and a wrong password return the identical error, so the
    endpoint cannot be used to discover which addresses have accounts.
    """
    _, tokens = await service.login(payload.email, payload.password)
    return tokens


@router.post(
    "/refresh",
    response_model=TokenPair,
    summary="Exchange a refresh token",
    responses=COMMON_ERRORS,
)
async def refresh(payload: RefreshRequest, service: AuthServiceDep) -> TokenPair:
    """Rotate a refresh token for a fresh pair.

    The presented refresh token is revoked as it is exchanged, so a stolen
    token is usable at most once.
    """
    return await service.refresh(payload.refresh_token)


@router.post("/logout", response_model=MessageResponse, summary="Sign out")
async def logout(
    request: Request,
    service: AuthServiceDep,
    _: CurrentUser,
) -> MessageResponse:
    """Revoke the presented access token for its remaining lifetime."""
    await service.logout(extract_bearer_token(request))
    return MessageResponse(message="Signed out successfully.")


@router.get(
    "/me",
    response_model=UserWithProfile,
    summary="Current account and profile",
    responses=COMMON_ERRORS,
)
async def me(user: CurrentUser, service: UserServiceDep) -> UserWithProfile:
    """Return the signed-in user together with their health profile."""
    profile = await service.get_profile(user.id)
    return UserWithProfile(
        **UserResponse.model_validate(user).model_dump(),
        profile=service.serialise_profile(profile),  # type: ignore[arg-type]
    )


@router.post(
    "/change-password",
    response_model=MessageResponse,
    summary="Change password",
    responses=COMMON_ERRORS,
)
async def change_password(
    payload: ChangePasswordRequest, user: CurrentUser, service: AuthServiceDep
) -> MessageResponse:
    await service.change_password(user.id, payload.current_password, payload.new_password)
    return MessageResponse(
        message="Password updated. Existing sessions on other devices remain "
        "signed in until their tokens expire."
    )


@router.post(
    "/delete-account",
    response_model=MessageResponse,
    summary="Permanently delete this account",
    responses=COMMON_ERRORS,
)
async def delete_account(
    payload: DeleteAccountRequest, user: CurrentUser, service: AuthServiceDep
) -> MessageResponse:
    """Irreversibly delete the account and every record attached to it.

    Requires the current password plus an explicit typed confirmation, because
    there is no undo — the delete cascades through every health table.
    """
    await service.delete_account(user.id, payload.password)
    return MessageResponse(
        message="Your account and all associated health data have been "
        "permanently deleted."
    )
