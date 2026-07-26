"""Shared pytest fixtures.

Testing strategy
----------------
The integration and API suites run against **real SQLite**, not mocks. Mocking
the database would leave the layer most likely to break — queries, constraints,
cascade behaviour — completely unexercised, and the bugs that actually shipped
in this project were all in that layer.

Each test gets a fresh schema in a file-backed temporary database. In-memory
SQLite would be faster but gives each connection its own private database,
which breaks the moment the pool opens a second connection.

The AI layer is pointed at its local providers, so the whole suite runs
offline, deterministically, and for free.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import tempfile
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

# Configure the environment *before* importing anything from `app`, since
# settings are read at import time and cached.
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault(
    # Generated per run rather than written as a literal. A hard-coded
    # high-entropy string is indistinguishable from a leaked key to a secret
    # scanner, and generating it also guarantees no test can quietly come to
    # depend on a fixed signing key.
    "SECRET_KEY",
    secrets.token_urlsafe(48),
)
os.environ.setdefault("LLM_PROVIDER", "local")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """One event loop for the whole session.

    Without this, pytest-asyncio creates a loop per test while the SQLAlchemy
    engine is bound to the loop that created it, producing confusing
    "attached to a different loop" failures.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_path() -> AsyncIterator[Path]:
    """A throwaway SQLite file, removed after the test."""
    handle, name = tempfile.mkstemp(suffix=".db", prefix="oviora-test-")
    os.close(handle)
    path = Path(name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
async def session(db_path: Path):
    """A database session against a freshly-created schema."""
    from app.core.database import Base, configure_engine, get_engine, get_session_factory

    configure_engine(f"sqlite+aiosqlite:///{db_path}")
    import app.models  # noqa: F401 - registers tables on the metadata

    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with get_session_factory()() as s:
        yield s

    from app.core.database import dispose_engine

    await dispose_engine()


@pytest.fixture
async def client(db_path: Path) -> AsyncIterator:
    """A TestClient wired to an isolated database, with lifespan run."""
    from fastapi.testclient import TestClient

    import app.core.redis_client as redis_module
    from app.core.database import configure_engine
    from app.core.redis_client import CacheService, InMemoryBackend

    configure_engine(f"sqlite+aiosqlite:///{db_path}")
    # Force the in-process cache so the suite never depends on a Redis daemon
    # and cannot leak rate-limit counters between tests.
    redis_module._cache = CacheService(InMemoryBackend())

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client

    from app.core.database import dispose_engine

    await dispose_engine()


@pytest.fixture
def user_payload() -> dict:
    """Registration payload with a unique email per test."""
    return {
        "email": f"asha-{uuid.uuid4().hex[:8]}@example.com",
        "password": "Oviora!Secure24",
        "full_name": "Asha Rao",
    }


@pytest.fixture
def auth_headers(client, user_payload: dict) -> dict[str, str]:
    """Register a user and return ready-to-use Authorization headers."""
    response = client.post("/api/v1/auth/register", json=user_payload)
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def sample_risk_payload() -> dict:
    """A high-risk feature vector for prediction tests."""
    return {
        "age": 24,
        "bmi": 31.2,
        "cycle_length_days": 48,
        "cycle_irregularity": 1,
        "weight_gain": 1,
        "hair_growth": 1,
        "skin_darkening": 1,
        "hair_loss": 1,
        "pimples": 1,
        "fast_food": 1,
        "exercise_hours_per_week": 0.5,
        "sleep_hours": 5.5,
        "stress_level": 5,
        "family_history": 1,
        "activity_level": "sedentary",
    }


@pytest.fixture
def low_risk_payload() -> dict:
    """A low-risk feature vector, used to assert the model discriminates."""
    return {
        "age": 23,
        "bmi": 21.5,
        "cycle_length_days": 28,
        "cycle_irregularity": 0,
        "weight_gain": 0,
        "hair_growth": 0,
        "skin_darkening": 0,
        "hair_loss": 0,
        "pimples": 0,
        "fast_food": 0,
        "exercise_hours_per_week": 6.0,
        "sleep_hours": 8.0,
        "stress_level": 2,
        "family_history": 0,
        "activity_level": "active",
    }
