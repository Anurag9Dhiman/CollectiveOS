"""
Morning briefing engine.

Assembles a personalised daily brief from available data sources (health,
memory/graph context) and synthesises it into a short spoken summary using
Gemini Flash.

Note: calendar, email, tasks, and weather sections have been removed — those
are now handled by the navigation agent when the user explicitly asks. The
briefing focuses on what the assistant already knows: health metrics and
personal context from memory.

Entry points
  generate()             → dict
  schedule_enabled()     → bool
  get_config()           → dict
  set_config(patch)      → dict
  register_job()
  deliver()
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
    "Lead with the date. Mention relevant health or recovery data if available. "
    "Close with a relevant memory or personal context note. "
    "Be conversational, not robotic. Omit sections marked UNAVAILABLE."
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

def _get_health() -> str | None:
    try:
        from src.connectors.health import health_get_readiness
        result = health_get_readiness()
        if result and len(result.strip()) > 10:
            return result[:600]
    except Exception as exc:
        log.debug("Health fetch failed: %s", exc)
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
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return _fallback_text(sections)

    try:
        from google import genai
        from google.genai import types as _gt

        lines = [f"Today is {sections['date']}."]
        for key, label in [
            ("health",  "HEALTH / RECOVERY"),
            ("memory",  "PERSONAL CONTEXT"),
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
                max_output_tokens=300,
            ),
        )
        return (resp.text or "").strip() or _fallback_text(sections)
    except Exception as exc:
        log.warning("Briefing synthesis failed: %s", exc)
        return _fallback_text(sections)


def _fallback_text(sections: dict) -> str:
    parts = [f"Good morning! Today is {sections.get('date', 'today')}."]
    if sections.get("health"):
        parts.append("Health: " + sections["health"][:200])
    if sections.get("memory"):
        parts.append("Context: " + sections["memory"][:200])
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public: generate
# ---------------------------------------------------------------------------

def generate() -> dict:
    """
    Collect available data sections and synthesise a spoken briefing.
    Returns:
      { "date": "...", "sections": {...}, "briefing": "...", "generated_at": "..." }
    """
    now = datetime.now()
    date_str = now.strftime("%A, %B %-d, %Y")

    sections = {
        "date":   date_str,
        "health": _get_health(),
        "memory": _get_memory_context(),
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
    JOB_ID = "morning_briefing"
    cfg = get_config()

    if scheduler.get_job(JOB_ID):
        scheduler.remove_job(JOB_ID)

    if not cfg.get("enabled", False):
        log.info("Morning briefing disabled — job removed")
        return

    from apscheduler.triggers.cron import CronTrigger
    tz = cfg.get("timezone", "UTC")
    try:
        trigger = CronTrigger(hour=cfg["hour"], minute=cfg["minute"], timezone=tz)
    except Exception as exc:
        log.warning("Invalid briefing timezone %r: %s — falling back to UTC", tz, exc)
        trigger = CronTrigger(hour=cfg["hour"], minute=cfg["minute"])

    scheduler.add_job(deliver, trigger=trigger, id=JOB_ID, replace_existing=True)
    log.info("Morning briefing scheduled: %02d:%02d %s", cfg["hour"], cfg["minute"], tz)
