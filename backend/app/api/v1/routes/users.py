"""Profile, settings and data-portability endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Response, status

from app.api.deps import CurrentUser, CurrentUserId, UserServiceDep
from app.schemas.common import ErrorResponse, MessageResponse
from app.schemas.user import ProfileResponse, ProfileUpdate, UserResponse, UserUpdate

router = APIRouter(prefix="/users", tags=["Profile & Settings"])


@router.get(
    "/profile",
    response_model=ProfileResponse,
    summary="Get health profile",
    responses={401: {"model": ErrorResponse}},
)
async def get_profile(
    user_id: CurrentUserId, service: UserServiceDep
) -> ProfileResponse:
    """Return the health profile, creating an empty one if none exists."""
    profile = await service.get_profile(user_id)
    return ProfileResponse(**service.serialise_profile(profile))


@router.patch(
    "/profile",
    response_model=ProfileResponse,
    summary="Update health profile",
    responses={422: {"model": ErrorResponse}},
)
async def update_profile(
    payload: ProfileUpdate, user_id: CurrentUserId, service: UserServiceDep
) -> ProfileResponse:
    """Partially update the profile.

    PATCH rather than PUT: the client sends only what changed, so an onboarding
    step that collects height cannot accidentally clear a stored allergy list.
    """
    profile = await service.update_profile(user_id, payload)
    return ProfileResponse(**service.serialise_profile(profile))


@router.patch(
    "/account",
    response_model=UserResponse,
    summary="Update name or email",
    responses={409: {"model": ErrorResponse, "description": "Email already in use"}},
)
async def update_account(
    payload: UserUpdate, user_id: CurrentUserId, service: UserServiceDep
) -> UserResponse:
    user = await service.update_account(user_id, payload)
    return UserResponse.model_validate(user)


@router.get(
    "/export",
    summary="Export all personal data",
    response_class=Response,
    responses={
        200: {
            "content": {"application/json": {}},
            "description": "A downloadable JSON archive of every stored record.",
        }
    },
)
async def export_data(
    user: CurrentUser, service: UserServiceDep
) -> Response:
    """Download a complete copy of the account's data.

    Served as a file attachment rather than a JSON body so the browser saves it
    directly. This is the GDPR data-portability path and deliberately walks
    every table rather than a curated subset.
    """
    payload = await service.export_data(user.id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    filename = f"oviora-export-{stamp}.json"

    return Response(
        content=json.dumps(payload, indent=2, default=str),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete(
    "/memory",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear what the AI remembers",
)
async def clear_ai_memory(
    user_id: CurrentUserId, service: UserServiceDep
) -> MessageResponse:
    """Wipe the free-form facts the assistant has learned from conversation.

    Structured profile fields are untouched — those the user entered
    deliberately. This clears only what was inferred, which is the part they
    may not have realised was being stored.
    """
    from app.ai.memory.manager import ConversationMemory
    from app.ai.llm.factory import get_llm_provider

    memory = ConversationMemory(service.session, get_llm_provider())
    await memory.clear_long_term(user_id)
    await service.session.commit()
    return MessageResponse(
        message="Cleared. The assistant no longer remembers facts inferred from "
                "your conversations."
    )
