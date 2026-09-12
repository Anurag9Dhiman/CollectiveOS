"""
personalization.py
------------------
Lightweight user-profile store.  Tracks tool usage, active hours, and
common query patterns so the assistant can:

  1. Inject a short "user habits" summary into the system prompt.
  2. Surface proactive routine suggestions based on behaviour.
  3. Expose a /profile endpoint for the dashboard.

Storage: a single JSON file at ~/.collectiveos_profile.json (no new DB
table needed; the data is small and append-only writes are safe).

Public API (called from assistant_starter and api.py):
  record_interaction(tool_name, query_text)  — call after every tool exec
  get_summary()                              — returns dict for /dashboard
  get_profile_context()                      — returns short str for system prompt
  suggest_routines()                         — returns list of suggestion dicts
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_PROFILE_PATH = Path(os.environ.get(
    "COLLECTIVEOS_PROFILE_PATH",
    Path.home() / ".collectiveos_profile.json",
))
_lock = threading.Lock()

# In-memory cache, loaded once at import and flushed on every write.
_cache: dict = {}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load() -> dict:
    global _cache
    if _cache:
        return _cache
    try:
        if _PROFILE_PATH.exists():
            _cache = json.loads(_PROFILE_PATH.read_text())
    except Exception as exc:
        log.warning("personalization: load failed: %s", exc)
        _cache = {}
    _cache.setdefault("tool_counts", {})
    _cache.setdefault("hour_counts", {})   # "HH" → count
    _cache.setdefault("weekday_counts", {})  # "Mon" → count
    _cache.setdefault("queries", [])        # last 100 query texts
    _cache.setdefault("last_updated", None)
    return _cache


def _save(data: dict) -> None:
    global _cache
    try:
        _PROFILE_PATH.write_text(json.dumps(data, indent=2))
        _cache = data
    except Exception as exc:
        log.warning("personalization: save failed: %s", exc)


# ---------------------------------------------------------------------------
# Public write API
# ---------------------------------------------------------------------------

def record_interaction(tool_name: str, query_text: str = "") -> None:
    """Record one tool call.  Thread-safe; silently ignores errors."""
    try:
        now = datetime.now(timezone.utc)
        hour    = now.strftime("%H")
        weekday = now.strftime("%a")

        with _lock:
            data = _load()
            tc = data["tool_counts"]
            tc[tool_name] = tc.get(tool_name, 0) + 1

            hc = data["hour_counts"]
            hc[hour] = hc.get(hour, 0) + 1

            wc = data["weekday_counts"]
            wc[weekday] = wc.get(weekday, 0) + 1

            if query_text:
                data["queries"].append(query_text[:120])
                data["queries"] = data["queries"][-100:]  # keep last 100

            data["last_updated"] = now.isoformat()
            total = sum(data["tool_counts"].values())
            _save(data)

        # Every 10 interactions push a proactive routine suggestion
        if total > 0 and total % 10 == 0:
            try:
                suggestions = suggest_routines()
                if suggestions:
                    s = suggestions[0]
                    from src import proactive as _proactive
                    _proactive.push(
                        f"💡 Suggestion: {s['title']} — {s['description']}",
                        trigger="personalization",
                        icon="💡",
                        action_prompt=f"Set up a routine for me: {s['title']}. Schedule it at {s['suggested_schedule']}.",
                    )
            except Exception:
                pass
    except Exception as exc:
        log.debug("personalization.record_interaction: %s", exc)


# ---------------------------------------------------------------------------
# Public read API
# ---------------------------------------------------------------------------

def get_summary() -> dict:
    """Return a dict suitable for the /dashboard endpoint."""
    with _lock:
        data = _load()

    tc = data.get("tool_counts", {})
    if not tc:
        return {"top_tool": None, "top_tool_count": 0, "active_hour": None, "total_calls": 0}

    top_tool = max(tc, key=tc.__getitem__)
    hc = data.get("hour_counts", {})
    active_hour = max(hc, key=hc.__getitem__) if hc else None

    return {
        "top_tool":       top_tool,
        "top_tool_count": tc[top_tool],
        "active_hour":    f"{active_hour}:00" if active_hour else None,
        "total_calls":    sum(tc.values()),
        "last_updated":   data.get("last_updated"),
    }


def get_profile_context() -> str:
    """Return a short paragraph injected into the system prompt."""
    with _lock:
        data = _load()

    tc = data.get("tool_counts", {})
    if not tc:
        return ""

    top = Counter(tc).most_common(3)
    top_str = ", ".join(f"{t}({c})" for t, c in top)

    hc = data.get("hour_counts", {})
    active_hour = max(hc, key=hc.__getitem__) if hc else None
    hour_str = f" Most active around {active_hour}:00 UTC." if active_hour else ""

    return (
        f"[User profile] Most-used tools: {top_str}.{hour_str} "
        "Tailor suggestions and proactive tips to these patterns."
    )


def suggest_routines() -> list[dict]:
    """
    Return up to 3 routine suggestions based on tool usage patterns.
    Each suggestion: {title, description, suggested_schedule}.
    """
    with _lock:
        data = _load()

    tc = data.get("tool_counts", {})
    suggestions = []

    if tc.get("health_get_sleep", 0) >= 3:
        suggestions.append({
            "title": "Daily health check",
            "description": "Summarise sleep and readiness scores every morning.",
            "suggested_schedule": "0 8 * * *",
        })
    if tc.get("web_search", 0) >= 5:
        suggestions.append({
            "title": "Morning news briefing",
            "description": "Search for top news and send a Slack summary.",
            "suggested_schedule": "0 7 * * 1-5",
        })
    if tc.get("navigate_computer", 0) >= 3:
        suggestions.append({
            "title": "Daily computer routine",
            "description": "Automate your most common computer tasks each morning.",
            "suggested_schedule": "0 9 * * 1-5",
        })
    if tc.get("get_devices", 0) >= 3 or tc.get("control_device", 0) >= 2:
        suggestions.append({
            "title": "Evening home check",
            "description": "Check all smart-home devices and turn off lights at 10 pm.",
            "suggested_schedule": "0 22 * * *",
        })

    return suggestions[:3]
