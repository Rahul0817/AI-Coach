"""Generic async repository.

Why a repository layer at all, when SQLAlchemy is already an abstraction?

* **Testability.** Services depend on a repository interface, so a unit test can
  substitute a fake without a database.
* **Query reuse.** ``list_for_user_between_dates`` is written once instead of
  being re-derived (subtly differently) in six route handlers.
* **A seam for change.** Adding read replicas, caching, or a switch to another
  store touches this layer only.

The typing is deliberately strict: ``BaseRepository[MealLog]`` gives full
autocomplete and catches "wrong model" mistakes at type-check time.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date
from typing import Any, Generic, TypeVar

from sqlalchemy import Select, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import Base
from app.core.exceptions import NotFoundError

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """CRUD primitives shared by every concrete repository."""

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ------------------------------------------------------------- creation
    async def create(self, **kwargs: Any) -> ModelT:
        """Insert a row and flush so the caller can read server defaults.

        ``flush`` rather than ``commit``: the unit of work is owned by the
        service layer, which decides where the transaction boundary sits.
        """
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def bulk_create(self, rows: Sequence[dict[str, Any]]) -> list[ModelT]:
        instances = [self.model(**row) for row in rows]
        self.session.add_all(instances)
        await self.session.flush()
        return instances

    # -------------------------------------------------------------- reading
    async def get(self, entity_id: uuid.UUID) -> ModelT | None:
        return await self.session.get(self.model, entity_id)

    async def get_or_404(self, entity_id: uuid.UUID) -> ModelT:
        instance = await self.get(entity_id)
        if instance is None:
            raise NotFoundError(f"{self.model.__name__} {entity_id} was not found.")
        return instance

    async def get_for_user(self, entity_id: uuid.UUID, user_id: uuid.UUID) -> ModelT:
        """Fetch a row *and* assert ownership in one query.

        Doing the ownership check in SQL rather than after loading closes the
        IDOR hole where a handler forgets to compare ``row.user_id``.
        """
        stmt = select(self.model).where(
            self.model.id == entity_id,  # type: ignore[attr-defined]
            self.model.user_id == user_id,  # type: ignore[attr-defined]
        )
        result = await self.session.execute(stmt)
        instance = result.scalar_one_or_none()
        if instance is None:
            raise NotFoundError(f"{self.model.__name__} {entity_id} was not found.")
        return instance

    async def list_all(self, *, limit: int = 100, offset: int = 0) -> list[ModelT]:
        stmt = select(self.model).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 100,
        offset: int = 0,
        order_desc: bool = True,
    ) -> list[ModelT]:
        stmt = select(self.model).where(
            self.model.user_id == user_id  # type: ignore[attr-defined]
        )
        stmt = self._apply_default_order(stmt, order_desc)
        result = await self.session.execute(stmt.limit(limit).offset(offset))
        return list(result.scalars().all())

    async def count_for_user(self, user_id: uuid.UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(self.model)
            .where(
                self.model.user_id == user_id  # type: ignore[attr-defined]
            )
        )
        return int((await self.session.execute(stmt)).scalar_one())

    async def list_between(
        self,
        user_id: uuid.UUID,
        start: date,
        end: date,
        *,
        date_column: str = "logged_on",
        limit: int = 1000,
    ) -> list[ModelT]:
        """Inclusive date-window query used by every analytics rollup."""
        column = getattr(self.model, date_column)
        stmt = (
            select(self.model)
            .where(
                self.model.user_id == user_id,  # type: ignore[attr-defined]
                column >= start,
                column <= end,
            )
            .order_by(column.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def find_one_by(self, **filters: Any) -> ModelT | None:
        stmt = select(self.model).filter_by(**filters)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def exists(self, **filters: Any) -> bool:
        stmt = select(func.count()).select_from(self.model).filter_by(**filters)
        return int((await self.session.execute(stmt)).scalar_one()) > 0

    # ------------------------------------------------------------- mutation
    async def update(self, instance: ModelT, **kwargs: Any) -> ModelT:
        """Apply a partial update, ignoring ``None`` (PATCH semantics)."""
        for key, value in kwargs.items():
            if value is not None and hasattr(instance, key):
                setattr(instance, key, value)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def set_fields(self, instance: ModelT, **kwargs: Any) -> ModelT:
        """Apply an update that *may* set fields to ``None`` (explicit clear)."""
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def delete(self, instance: ModelT) -> None:
        await self.session.delete(instance)
        await self.session.flush()

    async def delete_for_user(self, entity_id: uuid.UUID, user_id: uuid.UUID) -> None:
        instance = await self.get_for_user(entity_id, user_id)
        await self.delete(instance)

    async def delete_all_for_user(self, user_id: uuid.UUID) -> int:
        stmt = delete(self.model).where(
            self.model.user_id == user_id  # type: ignore[attr-defined]
        )
        result = await self.session.execute(stmt)
        return int(result.rowcount or 0)

    # -------------------------------------------------------------- helpers
    def _apply_default_order(self, stmt: Select[Any], desc: bool) -> Select[Any]:
        """Order by the most natural date column the model exposes."""
        for candidate in ("logged_on", "start_date", "created_at"):
            column = getattr(self.model, candidate, None)
            if column is not None:
                return stmt.order_by(column.desc() if desc else column.asc())
        return stmt
