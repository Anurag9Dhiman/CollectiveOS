"""
Activity event log — a chronological record of what the assistant did proactively.

Records three event types:
  routine      — a scheduled routine ran (success or error)
  watcher      — a condition watcher triggered and notified
  notification — a notification was delivered via output_bus

The table is append-only and intentionally lightweight. It is not a
replacement for the existing tool_calls / api_usage observability tables;
those track cost and latency. This table answers "what did my assistant do?"
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.db import connect

log = logging.getLogger(__name__)

_CREATE = """
CREATE TABLE IF NOT EXISTS activity_events (
    id          SERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type  TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL DEFAULT ''
);
"""

_ICONS = {
    "routine":      "⏱",
    "watcher":      "👁",
    "notification": "🔔",
}


def bootstrap() -> None:
    """Create the activity_events table if it doesn't exist."""
    try:
        conn = connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE)
        conn.close()
    except Exception as exc:
        log.debug("activity bootstrap skipped: %s", exc)


def log_event(event_type: str, title: str, body: str = "") -> None:
    """
    Append one activity event. Fire-and-forget — swallows all errors so
    a DB hiccup never breaks a routine or watcher.
    """
    try:
        conn = connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO activity_events (event_type, title, body) VALUES (%s, %s, %s)",
                    (event_type, title[:255], body[:1000]),
                )
        conn.close()
    except Exception as exc:
        log.debug("activity log_event failed (%s): %s", event_type, exc)


def list_events(limit: int = 100, days: int = 7) -> list[dict]:
    """Return recent activity events, newest first."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        conn = connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, created_at, event_type, title, body "
                "FROM activity_events WHERE created_at >= %s "
                "ORDER BY created_at DESC LIMIT %s",
                (since, limit),
            )
            rows = cur.fetchall()
        conn.close()
        return [
            {
                "id":         r[0],
                "created_at": r[1].isoformat() if r[1] else "",
                "event_type": r[2],
                "icon":       _ICONS.get(r[2], "•"),
                "title":      r[3],
                "body":       r[4],
            }
            for r in rows
        ]
    except Exception as exc:
        log.debug("activity list_events failed: %s", exc)
        return []
