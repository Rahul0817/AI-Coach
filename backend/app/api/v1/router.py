"""Aggregates every v1 route module into one router.

Versioning the API from day one costs nothing now and is the only way to make a
breaking change later without stranding an old mobile client.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes import (
    analytics,
    auth,
    chat,
    health_tracking,
    multimodal,
    tracking,
    users,
)

api_router = APIRouter()

api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(chat.router)
api_router.include_router(health_tracking.router)
api_router.include_router(tracking.router)
api_router.include_router(analytics.router)
api_router.include_router(multimodal.router)

__all__ = ["api_router"]
