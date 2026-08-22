"""
Health connector — sleep, activity, and readiness metrics.

Sources (in priority order):
  1. Oura Ring REST API  — if OURA_TOKEN is set, reads directly from Oura Cloud
  2. Shortcuts bridge    — reads from health_snapshots table populated by an iOS
                           Shortcut that POSTs to POST /health-ingest

Env vars:
  OURA_TOKEN — Personal access token from https://cloud.ouraring.com/personal-access-tokens
"""

import datetime
import json
import os

import requests

_OURA_BASE = "https://api.ouraring.com/v2"


def _oura_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {os.environ.get('OURA_TOKEN', '')}"}


def _has_oura() -> bool:
    return bool(os.environ.get("OURA_TOKEN"))


def _date_range(days: int) -> tuple[str, str]:
    today = datetime.date.today()
    start = today - datetime.timedelta(days=days - 1)
    return start.isoformat(), today.isoformat()


def _oura_get(path: str, params: dict) -> list[dict]:
    resp = requests.get(
        f"{_OURA_BASE}{path}", headers=_oura_headers(), params=params, timeout=15
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def _fmt_dur(seconds) -> str:
    if seconds is None:
        return "—"
    try:
        s = int(seconds)
        return f"{s // 3600}h {(s % 3600) // 60}m"
    except Exception:
        return str(seconds)


# ---------------------------------------------------------------------------
# Oura-backed readers
# ---------------------------------------------------------------------------


def _oura_sleep(days: int) -> str:
    start, end = _date_range(days)
    data = _oura_get("/usercollection/daily_sleep", {"start_date": start, "end_date": end})
    if not data:
        return f"No Oura sleep data for the last {days} days."
    lines = [f"Sleep — last {days} days (Oura Ring):"]
    for d in sorted(data, key=lambda x: x.get("day", "")):
        lines.append(
            f"  {d['day']}  score={d.get('score', '—')}  "
            f"total={_fmt_dur(d.get('total_sleep_duration'))}  "
            f"deep={_fmt_dur(d.get('deep_sleep_duration'))}  "
            f"REM={_fmt_dur(d.get('rem_sleep_duration'))}  "
            f"HRV={d.get('average_hrv', '—')}ms  "
            f"RHR={d.get('lowest_heart_rate', '—')}bpm  "
            f"efficiency={d.get('efficiency', '—')}%"
        )
    return "\n".join(lines)


def _oura_activity(days: int) -> str:
    start, end = _date_range(days)
    data = _oura_get("/usercollection/daily_activity", {"start_date": start, "end_date": end})
    if not data:
        return f"No Oura activity data for the last {days} days."
    lines = [f"Activity — last {days} days (Oura Ring):"]
    for d in sorted(data, key=lambda x: x.get("day", "")):
        steps = d.get("steps", 0)
        steps_str = f"{steps:,}" if isinstance(steps, int) else str(steps)
        lines.append(
            f"  {d['day']}  score={d.get('score', '—')}  steps={steps_str}  "
            f"active={d.get('active_calories', '—')}kcal  "
            f"total={d.get('total_calories', '—')}kcal"
        )
    return "\n".join(lines)


def _oura_readiness(days: int) -> str:
    start, end = _date_range(days)
    data = _oura_get("/usercollection/daily_readiness", {"start_date": start, "end_date": end})
    if not data:
        return f"No Oura readiness data for the last {days} days."
    lines = [f"Readiness — last {days} days (Oura Ring):"]
    for d in sorted(data, key=lambda x: x.get("day", "")):
        c = d.get("contributors", {})
        lines.append(
            f"  {d['day']}  score={d.get('score', '—')}  "
            f"HRV balance={c.get('hrv_balance', '—')}  "
            f"RHR score={c.get('resting_heart_rate', '—')}  "
            f"sleep balance={c.get('sleep_balance', '—')}  "
            f"activity balance={c.get('activity_balance', '—')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shortcuts-cache readers (Postgres health_snapshots)
# ---------------------------------------------------------------------------

_CACHE_HINT = (
    "\n\nTip: set OURA_TOKEN for automatic sync, or push data from an iOS Shortcut "
    "via POST /health-ingest with {\"date\": \"YYYY-MM-DD\", \"metrics\": {...}}."
)


def _cache_read(keys: list[str], days: int, label: str) -> str:
    try:
        from collectiveos.db import connect
        start, end = _date_range(days)
        conn = connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT date, metrics FROM health_snapshots "
                "WHERE date BETWEEN %s AND %s ORDER BY date ASC",
                (start, end),
            )
            rows = cur.fetchall()
        conn.close()
        if not rows:
            return f"No {label.lower()} data for the last {days} days." + _CACHE_HINT
        lines = [f"{label} — last {days} days (Apple Health cache):"]
        for date, metrics in rows:
            parts = [str(date)]
            for k in keys:
                if k in metrics:
                    parts.append(f"{k}={metrics[k]}")
            lines.append("  " + "  ".join(parts))
        return "\n".join(lines)
    except Exception as exc:
        return f"Health cache read error: {exc}" + _CACHE_HINT


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def health_get_sleep(days: int = 7) -> str:
    """Get sleep duration, efficiency, HRV, and heart rate for the last N days."""
    if _has_oura():
        try:
            return _oura_sleep(days)
        except Exception as exc:
            return f"Oura sleep error: {exc}"
    return _cache_read(["sleep_hours", "deep_sleep_hours", "rem_sleep_hours",
                         "hrv", "resting_heart_rate", "sleep_efficiency"], days, "Sleep")


def health_get_activity(days: int = 7) -> str:
    """Get steps and active calories for the last N days."""
    if _has_oura():
        try:
            return _oura_activity(days)
        except Exception as exc:
            return f"Oura activity error: {exc}"
    return _cache_read(["steps", "active_calories", "total_calories", "workouts"], days, "Activity")


def health_get_readiness(days: int = 7) -> str:
    """Get readiness score and HRV balance for the last N days."""
    if _has_oura():
        try:
            return _oura_readiness(days)
        except Exception as exc:
            return f"Oura readiness error: {exc}"
    return _cache_read(["readiness_score", "hrv", "resting_heart_rate"], days, "Readiness")
