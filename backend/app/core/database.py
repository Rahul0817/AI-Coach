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
    """Declarative base shared by every ORM model."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    def to_dict(self) -> dict[str, Any]:
        """Shallow column dump — handy in tests and debug endpoints."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


def create_engine() -> AsyncEngine:
    """Build the engine with a pool sized for a typical container."""
    return create_async_engine(
        settings.sqlalchemy_uri,
        echo=False,
        future=True,
        pool_pre_ping=True,   # transparently drops connections killed by the DB
        pool_size=10,
        max_overflow=20,
        pool_recycle=1800,    # stay under most managed-Postgres idle timeouts
    )


engine: AsyncEngine = create_engine()

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a request-scoped session."""
    async with SessionLocal() as session:
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

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database schema ensured")


async def dispose_engine() -> None:
    """Close pooled connections on shutdown so containers exit cleanly."""
    await engine.dispose()
    logger.info("database engine disposed")
