"""
Wearable connector — reads events stored by POST /wearable/ingest.

Any wearable device (Garmin, Frame glasses, Apple Watch via Shortcuts,
custom hardware) can push events to /wearable/ingest. This connector
lets the agent query those events.

Ingest payload shape (all fields except device_id and event_type optional):
  {
    "device_id":  "garmin-forerunner-265",
    "event_type": "gesture" | "sensor" | "location" | "button" | "voice",
    "payload":    { ... device-specific fields ... }
  }
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from src import db as _db


def wearable_get_events(
    limit: int = 20,
    device_id: str | None = None,
    event_type: str | None = None,
) -> str:
    """Return recent wearable events, optionally filtered by device or type."""
    conn = _db.connect()
    try:
        uid = _db.default_user_id(conn)
        clauses = ["user_id = %s"]
        params: list = [uid]
        if device_id:
            clauses.append("device_id = %s")
            params.append(device_id)
        if event_type:
            clauses.append("event_type = %s")
            params.append(event_type)
        where = " AND ".join(clauses)
        params.append(max(1, min(limit, 100)))
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT device_id, event_type, payload, created_at "
                f"FROM wearable_events WHERE {where} "
                f"ORDER BY created_at DESC LIMIT %s",
                params,
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return "No wearable events found."

    lines = []
    for device, etype, payload, ts in rows:
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
        payload_str = json.dumps(payload)[:120] if payload else "{}"
        lines.append(f"[{ts_str}] {device} / {etype}: {payload_str}")
    return "\n".join(lines)


def wearable_list_devices() -> str:
    """List distinct wearable devices that have sent events."""
    conn = _db.connect()
    try:
        uid = _db.default_user_id(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT device_id, COUNT(*) AS events, MAX(created_at) AS last_seen "
                "FROM wearable_events WHERE user_id = %s "
                "GROUP BY device_id ORDER BY last_seen DESC",
                (uid,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return "No wearable devices have sent data yet."

    lines = []
    for device_id, count, last_seen in rows:
        ts = last_seen.strftime("%Y-%m-%d %H:%M") if hasattr(last_seen, "strftime") else str(last_seen)
        lines.append(f"• {device_id} — {count} event(s), last seen {ts}")
    return "\n".join(lines)
