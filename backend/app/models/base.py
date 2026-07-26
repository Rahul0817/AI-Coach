"""Shared ORM building blocks: primary keys, timestamps, soft ownership.

Every table in Oviora uses a UUID primary key rather than a serial integer:

* UUIDs can be generated client-side, which lets us write a parent and its
  children in one flush without a round trip per row.
* They do not leak business volume. ``/api/v1/users/1043`` tells a competitor
  how many users you have; a UUID does not.
* They make merging data across environments (and future sharding) safe.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column


class UUIDPrimaryKeyMixin:
    """Adds a client-generatable UUID primary key."""

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )


class TimestampMixin:
    """Adds ``created_at`` / ``updated_at`` maintained by the database.

    ``server_default=func.now()`` means correctness does not depend on the
    application clock, and rows inserted by a migration or an admin SQL session
    still get sane timestamps.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class UserOwnedMixin:
    """Adds the tenant key used by every per-user health record.

    ``ondelete="CASCADE"`` implements the right-to-erasure requirement at the
    database level: deleting a user removes every derived health record in one
    transaction, with no chance of orphaned PHI surviving an application bug.
    """

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
