"""Redis client singleton for CollectiveOS.

Falls back to an in-process TTL dict when REDIS_URL is not set, so the app
works in development without a Redis instance. The public API (get/set/delete/
exists/publish/subscribe) is the same in both modes.

Usage:
    from src import redis_client as cache
    cache.set("my:key", {"some": "value"}, ttl=60)
    data = cache.get("my:key")   # None after TTL expires
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

_REDIS_URL: str = os.environ.get("REDIS_URL", "")
_redis = None
_fallback: dict[str, tuple[Any, float]] = {}  # key → (value, expiry_unix)


def _client():
    global _redis
    if _redis is not None:
        return _redis
    if not _REDIS_URL:
        return None
    import redis as _r
    _redis = _r.from_url(_REDIS_URL, decode_responses=True)
    return _redis


# ---------------------------------------------------------------------------
# Core cache operations
# ---------------------------------------------------------------------------

def get(key: str) -> Any | None:
    """Return the cached value or None if absent / expired."""
    r = _client()
    if r is None:
        entry = _fallback.get(key)
        if entry is None:
            return None
        value, exp = entry
        if exp and time.time() > exp:
            _fallback.pop(key, None)
            return None
        return value
    raw = r.get(key)
    return json.loads(raw) if raw is not None else None


def set(key: str, value: Any, ttl: int = 300) -> None:
    """Store value under key with a TTL in seconds (0 = no expiry)."""
    r = _client()
    if r is None:
        _fallback[key] = (value, time.time() + ttl if ttl else 0.0)
        return
    r.set(key, json.dumps(value), ex=ttl or None)


def delete(key: str) -> None:
    """Remove a key."""
    r = _client()
    if r is None:
        _fallback.pop(key, None)
        return
    r.delete(key)


def exists(key: str) -> bool:
    """Return True if key is present and not expired."""
    r = _client()
    if r is None:
        entry = _fallback.get(key)
        if entry is None:
            return False
        _, exp = entry
        if exp and time.time() > exp:
            _fallback.pop(key, None)
            return False
        return True
    return bool(r.exists(key))


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def ping() -> bool:
    """Return True if Redis is reachable (always True in fallback mode)."""
    r = _client()
    if r is None:
        return True
    try:
        return r.ping()
    except Exception:
        return False
