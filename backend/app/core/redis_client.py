"""Redis access layer with a transparent in-process fallback.

Redis backs three concerns in Oviora:

1. **Rate limiting** — a sliding counter per identity per route class.
2. **Response caching** — analytics rollups and RAG lookups are expensive and
   change slowly, so they are cached with a short TTL.
3. **Conversation working memory** — the last N turns of a chat thread live in
   Redis for fast prompt assembly, with Postgres as the durable record.

The fallback matters: a developer running ``pytest`` or evaluating the project
from a fresh clone should not need a Redis daemon. ``InMemoryBackend``
implements the same narrow interface, so the app degrades to single-process
behaviour instead of crashing. In production the real client is always used and
a failed connection is logged loudly.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CacheBackend(Protocol):
    """The subset of Redis that Oviora actually depends on."""

    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl: int | None = None) -> None: ...
    async def delete(self, *keys: str) -> None: ...
    async def incr_with_ttl(self, key: str, ttl: int) -> int: ...
    async def ttl(self, key: str) -> int: ...
    async def list_push(self, key: str, value: str, max_len: int, ttl: int) -> None: ...
    async def list_range(self, key: str, start: int = 0, end: int = -1) -> list[str]: ...
    async def ping(self) -> bool: ...


class InMemoryBackend:
    """Process-local stand-in used when Redis is unreachable.

    Deliberately simple: a dict with expiry timestamps. It is correct for a
    single process, which is exactly the situation it exists to serve (tests,
    local demos). It is *not* correct across workers, hence the warning.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}

    def _expired(self, key: str) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return True
        _, expires_at = entry
        if expires_at is not None and expires_at < time.time():
            self._store.pop(key, None)
            return True
        return False

    async def get(self, key: str) -> str | None:
        if self._expired(key):
            return None
        return self._store[key][0]

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        expires = time.time() + ttl if ttl else None
        self._store[key] = (value, expires)

    async def delete(self, *keys: str) -> None:
        for key in keys:
            self._store.pop(key, None)

    async def incr_with_ttl(self, key: str, ttl: int) -> int:
        if self._expired(key):
            self._store[key] = (0, time.time() + ttl)
        current, expires = self._store[key]
        current = int(current) + 1
        self._store[key] = (current, expires)
        return current

    async def ttl(self, key: str) -> int:
        if self._expired(key):
            return -2
        _, expires = self._store[key]
        return int(expires - time.time()) if expires else -1

    async def list_push(self, key: str, value: str, max_len: int, ttl: int) -> None:
        if self._expired(key):
            self._store[key] = ([], time.time() + ttl)
        items, expires = self._store[key]
        if not isinstance(items, list):
            items = []
        items.append(value)
        self._store[key] = (items[-max_len:], expires)

    async def list_range(self, key: str, start: int = 0, end: int = -1) -> list[str]:
        if self._expired(key):
            return []
        items = self._store[key][0]
        if not isinstance(items, list):
            return []
        return items[start:] if end == -1 else items[start : end + 1]

    async def ping(self) -> bool:
        return True


class RedisBackend:
    """Thin adapter over ``redis.asyncio`` implementing :class:`CacheBackend`."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def get(self, key: str) -> str | None:
        return await self._client.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        await self._client.set(key, value, ex=ttl)

    async def delete(self, *keys: str) -> None:
        if keys:
            await self._client.delete(*keys)

    async def incr_with_ttl(self, key: str, ttl: int) -> int:
        # Pipelined so the INCR and its EXPIRE cannot be interleaved by another
        # worker — otherwise a key can end up with no TTL and never reset.
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, ttl, nx=True)
            count, _ = await pipe.execute()
        return int(count)

    async def ttl(self, key: str) -> int:
        return int(await self._client.ttl(key))

    async def list_push(self, key: str, value: str, max_len: int, ttl: int) -> None:
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, value)
            pipe.ltrim(key, -max_len, -1)
            pipe.expire(key, ttl)
            await pipe.execute()

    async def list_range(self, key: str, start: int = 0, end: int = -1) -> list[str]:
        return list(await self._client.lrange(key, start, end))

    async def ping(self) -> bool:
        return bool(await self._client.ping())


class CacheService:
    """Application-facing cache API with JSON helpers and namespacing."""

    def __init__(self, backend: CacheBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> CacheBackend:
        return self._backend

    @property
    def is_distributed(self) -> bool:
        return isinstance(self._backend, RedisBackend)

    async def get_json(self, key: str) -> Any | None:
        raw = await self._backend.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # A poisoned key should never break a request; drop and miss.
            await self._backend.delete(key)
            return None

    async def set_json(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self._backend.set(key, json.dumps(value, default=str), ttl)

    async def delete(self, *keys: str) -> None:
        await self._backend.delete(*keys)

    async def hit_rate_limit(
        self, key: str, limit: int, window: int
    ) -> tuple[bool, int, int]:
        """Increment a counter and report whether the caller is over budget.

        Returns ``(allowed, remaining, retry_after_seconds)``.
        """
        count = await self._backend.incr_with_ttl(key, window)
        remaining = max(0, limit - count)
        if count > limit:
            retry_after = await self._backend.ttl(key)
            return False, 0, max(1, retry_after)
        return True, remaining, 0

    async def push_recent(self, key: str, value: Any, max_len: int, ttl: int) -> None:
        await self._backend.list_push(key, json.dumps(value, default=str), max_len, ttl)

    async def recent(self, key: str, limit: int) -> list[Any]:
        raw_items = await self._backend.list_range(key, -limit, -1)
        out: list[Any] = []
        for item in raw_items:
            try:
                out.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return out

    async def healthy(self) -> bool:
        try:
            return await self._backend.ping()
        except Exception:  # pragma: no cover - network dependent
            return False


_cache: CacheService | None = None


async def init_cache() -> CacheService:
    """Connect to Redis, falling back to in-process storage on failure."""
    global _cache
    backend: CacheBackend
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            settings.redis_uri,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        await client.ping()
        backend = RedisBackend(client)
        logger.info("redis connected", extra={"uri": settings.redis_uri})
    except Exception as exc:
        if settings.is_production:
            logger.error("redis unavailable in production", extra={"error": str(exc)})
        else:
            logger.warning(
                "redis unavailable, using in-process cache (single worker only)",
                extra={"error": str(exc)},
            )
        backend = InMemoryBackend()

    _cache = CacheService(backend)
    return _cache


def get_cache() -> CacheService:
    """FastAPI dependency returning the shared cache service."""
    global _cache
    if _cache is None:
        # Reached in unit tests that never ran the lifespan hook.
        _cache = CacheService(InMemoryBackend())
    return _cache


async def close_cache() -> None:
    global _cache
    if _cache is not None and isinstance(_cache.backend, RedisBackend):
        await _cache.backend._client.aclose()
    _cache = None
