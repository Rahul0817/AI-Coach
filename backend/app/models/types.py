"""Dialect-aware column types.

Postgres is the production database, and JSONB and native UUID are the right
choices there — JSONB is binary, indexable and queryable; a native UUID column
is 16 bytes rather than a 36-character string.

But the test suite runs against in-memory SQLite, which has neither type. Using
the ``postgresql`` dialect types directly makes ``CREATE TABLE`` fail there, so
the integration tests could only run against a real Postgres server — which
means they mostly do not get run.

``with_variant`` resolves this properly: SQLAlchemy emits the Postgres type when
connected to Postgres and the portable fallback everywhere else. One model
definition, correct DDL on both. This is not a testing hack — it is how
SQLAlchemy is designed to express "this type, specialised per backend".
"""

from __future__ import annotations

from sqlalchemy import JSON, Uuid
from sqlalchemy.dialects.postgresql import JSONB

#: UUID primary/foreign keys. ``sqlalchemy.Uuid`` is dialect-aware by design:
#: native ``UUID`` on Postgres, ``CHAR(32)`` elsewhere, with Python-side
#: conversion to and from ``uuid.UUID`` in both cases.
GUID = Uuid(as_uuid=True)

#: JSON document columns. JSONB on Postgres for binary storage and GIN
#: indexing; plain JSON elsewhere.
JSONDict = JSON().with_variant(JSONB(), "postgresql")

__all__ = ["GUID", "JSONDict"]
