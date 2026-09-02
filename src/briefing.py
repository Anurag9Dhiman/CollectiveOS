"""
Morning briefing engine.

Assembles a personalised daily brief from all connected data sources (calendar,
email, tasks, weather, memory) and synthesises it into a short spoken summary
using Gemini Flash.  Every section is best-effort: if a connector is not
configured the section is silently omitted rather than crashing.

Entry points
  generate()             → dict  (sections + synthesised "briefing" text)
  schedule_enabled()     → bool
  get_config()           → dict
  set_config(patch)      → dict  (merged config)
  register_job()                 (call once, at scheduler start-up)
  deliver()                      (generate + push via output_bus; called by APScheduler)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".briefing_config.json",
)

_DEFAULT_CONFIG: dict = {
    "enabled":    False,
    "hour":       8,
    "minute":     0,
    "timezone":   "UTC",
    "notify_via": "notification",
}

_SYNTHESIS_MODEL = os.environ.get("GEMINI_ROUTER_MODEL", "gemini-2.0-flash-lite")

_SYNTHESIS_SYSTEM = (
    "You are a personal AI assistant delivering a morning briefing. "
    "Given the sections below, write a warm, concise 3-5 sentence briefing "
    "that the user will hear spoken aloud. "
    "Lead with the date and any urgent calendar events. "
    "Mention key tasks and emails briefly. "
    "Close with the weather or a useful heads-up if available. "
    "Be conversational, not robotic. Omit sections that are marked UNAVAILABLE."
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_config() -> dict:
    try:
        with open(_CONFIG_FILE) as f:
            saved = json.load(f)
        return {**_DEFAULT_CONFIG, **saved}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(_DEFAULT_CONFIG)


def set_config(patch: dict) -> dict:
    cfg = get_config()
    # Validate hour/minute
    if "hour" in patch:
        patch["hour"] = max(0, min(23, int(patch["hour"])))
    if "minute" in patch:
        patch["minute"] = max(0, min(59, int(patch["minute"])))
    cfg.update(patch)
    try:
        with open(_CONFIG_FILE, "w") as f:
            json.dump(cfg, f)
    except OSError as exc:
        log.warning("Could not save briefing config: %s", exc)
    return cfg


def schedule_enabled() -> bool:
    return bool(get_config().get("enabled", False))


# ---------------------------------------------------------------------------
# Data gathering — each section is best-effort
# ---------------------------------------------------------------------------

def _get_weather() -> str | None:
    try:
        from src.connectors.web_search import search
        results = search("today weather forecast")
        if results and len(results) > 20:
            return results[:600]
    except Exception as exc:
        log.debug("Weather fetch failed: %s", exc)
    return None


def _get_calendar() -> str | None:
    try:
        from src.connectors.google_calendar import get_calendar_events
        events = get_calendar_events(days_ahead=1)
        if events and "no events" not in events.lower():
            return events[:800]
    except Exception as exc:
        log.debug("Calendar fetch failed: %s", exc)
    return None


def _get_tasks() -> str | None:
    try:
        from src.connectors.todoist import get_tasks
        tasks = get_tasks()
        if tasks and "no tasks" not in tasks.lower():
            return tasks[:600]
    except Exception as exc:
        log.debug("Tasks fetch failed: %s", exc)
    return None


def _get_emails() -> str | None:
    try:
        from src.connectors.gmail import get_recent_emails
        emails = get_recent_emails(max_results=5)
        if emails and "no email" not in emails.lower():
            return emails[:800]
    except Exception as exc:
        log.debug("Email fetch failed: %s", exc)
    return None


def _get_memory_context() -> str | None:
    try:
        from src import memory
        ctx = memory.search("morning priorities today goals")
        if ctx and len(ctx.strip()) > 20:
            return ctx[:500]
    except Exception as exc:
        log.debug("Memory context failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def _synthesize(sections: dict) -> str:
    """Call Gemini Flash to turn the raw sections into a spoken briefing."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return _fallback_text(sections)

    try:
        from google import genai
        from google.genai import types as _gt

        lines = [f"Today is {sections['date']}."]
        for key, label in [
            ("calendar", "CALENDAR"),
            ("tasks",    "TASKS"),
            ("emails",   "EMAILS"),
            ("weather",  "WEATHER"),
            ("memory",   "PERSONAL CONTEXT"),
        ]:
            val = sections.get(key)
            lines.append(f"\n[{label}]\n{val if val else 'UNAVAILABLE'}")

        prompt = "\n".join(lines)
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=_SYNTHESIS_MODEL,
            contents=prompt,
            config=_gt.GenerateContentConfig(
                system_instruction=_SYNTHESIS_SYSTEM,
                temperature=0.4,
                max_output_tokens=350,
            ),
        )
        return (resp.text or "").strip() or _fallback_text(sections)
    except Exception as exc:
        log.warning("Briefing synthesis failed: %s", exc)
        return _fallback_text(sections)


def _fallback_text(sections: dict) -> str:
    """Plain-text fallback when Gemini is unavailable."""
    parts = [f"Good morning! Today is {sections.get('date', 'today')}."]
    if sections.get("calendar"):
        parts.append("Your calendar: " + sections["calendar"][:200])
    if sections.get("tasks"):
        parts.append("Tasks due: " + sections["tasks"][:200])
    if sections.get("emails"):
        parts.append("Recent emails: " + sections["emails"][:200])
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public: generate
# ---------------------------------------------------------------------------

def generate() -> dict:
    """
    Collect all data sections and synthesise a spoken briefing.
    Returns a dict:
      { "date": "...", "sections": {...}, "briefing": "...", "generated_at": "..." }
    """
    now = datetime.now()
    date_str = now.strftime("%A, %B %-d, %Y")

    sections = {
        "date":     date_str,
        "weather":  _get_weather(),
        "calendar": _get_calendar(),
        "tasks":    _get_tasks(),
        "emails":   _get_emails(),
        "memory":   _get_memory_context(),
    }

    briefing_text = _synthesize(sections)
    log.info("Morning briefing generated (%d chars)", len(briefing_text))

    return {
        "date":         date_str,
        "sections":     sections,
        "briefing":     briefing_text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Scheduled delivery
# ---------------------------------------------------------------------------

def deliver() -> None:
    """Generate a briefing and push it via the configured output channel."""
    try:
        from src import output_bus
        result = generate()
        text = result["briefing"]
        cfg = get_config()
        output_bus.deliver(
            title="Morning Briefing",
            body=text,
            channel=cfg.get("notify_via", "notification"),
        )
        log.info("Morning briefing delivered via %s", cfg.get("notify_via"))
    except Exception as exc:
        log.error("Briefing delivery failed: %s", exc)


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------

def register_job(scheduler) -> None:
    """
    Register (or remove) the daily briefing job with the given APScheduler instance.
    Called once at scheduler start and again after config changes.
    """
    JOB_ID = "morning_briefing"
    cfg = get_config()

    # Always remove the existing job first
    if scheduler.get_job(JOB_ID):
        scheduler.remove_job(JOB_ID)

    if not cfg.get("enabled", False):
        log.info("Morning briefing disabled — job removed")
        return

    from apscheduler.triggers.cron import CronTrigger
    tz = cfg.get("timezone", "UTC")
    try:
        trigger = CronTrigger(
            hour=cfg["hour"],
            minute=cfg["minute"],
            timezone=tz,
        )
    except Exception as exc:
        log.warning("Invalid briefing timezone %r: %s — falling back to UTC", tz, exc)
        trigger = CronTrigger(hour=cfg["hour"], minute=cfg["minute"])

    scheduler.add_job(
        deliver,
        trigger=trigger,
        id=JOB_ID,
        replace_existing=True,
    )
    log.info(
        "Morning briefing scheduled: %02d:%02d %s",
        cfg["hour"], cfg["minute"], tz,
    )
