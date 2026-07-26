"""Async SQLAlchemy engine, session factory and declarative base.

Design notes
------------
* **Async all the way down.** Oviora spends most of its request time waiting on
  the LLM and on Postgres. An async driver (``asyncpg``) lets one worker hold
  thousands of in-flight requests instead of one per thread.
* **Session-per-request.** ``get_db`` yields a session scoped to a single HTTP
  request and guarantees rollback on failure, so a half-applied write can never
  leak into the next request.
* **``expire_on_commit=False``.** Without it, accessing an ORM attribute after
  ``commit()`` triggers a lazy refresh — which raises in async code. Turning it
  off is what makes "commit then serialise the object" work.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Explicit constraint naming makes Alembic autogenerate produce stable,
# reviewable migration names instead of database-assigned gibberish.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model.

    ``eager_defaults`` is essential here, not an optimisation. Columns with a
    server-side default or ``onupdate`` — ``created_at`` and ``updated_at`` on
    every table — are *expired* by SQLAlchemy after an INSERT or UPDATE, since
    only the database knows their new value. Reading one then triggers an
    implicit refresh, and implicit IO is illegal under asyncio: it raises
    ``MissingGreenlet``.

    That would break every PATCH/PUT endpoint that returns the updated row.
    With ``eager_defaults`` SQLAlchemy fetches those values inline via
    RETURNING (supported by both Postgres and SQLite), so the attributes are
    already populated and no lazy refresh is ever attempted.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    __mapper_args__ = {"eager_defaults": True}

    def to_dict(self) -> dict[str, Any]:
        """Shallow column dump — handy in tests and debug endpoints."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


def create_engine(uri: str | None = None) -> AsyncEngine:
    """Build the engine with a pool sized for a typical container.

    SQLite (used by the test suite) rejects the pool arguments below, so they
    are applied only for server-backed databases.
    """
    target = uri or settings.sqlalchemy_uri
    if target.startswith("sqlite"):
        return create_async_engine(target, echo=False, future=True)
    return create_async_engine(
        target,
        echo=False,
        future=True,
        pool_pre_ping=True,   # transparently drops connections killed by the DB
        pool_size=10,
        max_overflow=20,
        pool_recycle=1800,    # stay under most managed-Postgres idle timeouts
    )


# The engine is built lazily rather than at import time. Constructing it eagerly
# would make `import app.core.database` fail on any machine without the asyncpg
# driver installed — which breaks unit tests, tooling, and anything that merely
# wants to import a model class. Nothing should pay for a database connection
# it never uses.
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory, creating it on first use."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


def configure_engine(uri: str) -> None:
    """Point the process at a different database. Used by the test suite."""
    global _engine, _session_factory
    _engine = create_engine(uri)
    _session_factory = async_sessionmaker(
        bind=_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a request-scoped session."""
    async with get_session_factory()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_models() -> None:
    """Create tables directly from metadata.

    Alembic owns schema changes in every real environment; this exists for
    local bootstrapping and for the integration test suite, which builds a
    throwaway schema rather than replaying the full migration history.
    """
    # Importing the package registers every model on Base.metadata.
    import app.models  # noqa: F401

    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database schema ensured")


async def dispose_engine() -> None:
    """Close pooled connections on shutdown so containers exit cleanly."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("database engine disposed")
