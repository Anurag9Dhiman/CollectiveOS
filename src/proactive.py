"""
proactive.py
------------
Ambient proactive push queue.

Server-side components (scheduler jobs, screen watcher, personalization)
call push() to enqueue a message. The UI polls GET /proactive every 30 s
and renders any pending messages as assistant-initiated chat bubbles.

Messages expire after PROACTIVE_EXPIRE_S seconds if not consumed (default 10 min).

Public API:
    push(message, trigger, icon, action_prompt)   — enqueue a message
    pop_all()                                      — return + clear pending
    register_jobs(scheduler)                       — add APScheduler time jobs
"""

from __future__ import annotations

import logging
import os
import threading
import time

log = logging.getLogger(__name__)

_EXPIRE_S = int(os.environ.get("PROACTIVE_EXPIRE_S", 600))
_MORNING_HOUR = int(os.environ.get("PROACTIVE_MORNING_HOUR", 9))
_EVENING_HOUR = int(os.environ.get("PROACTIVE_EVENING_HOUR", 18))

_lock = threading.Lock()
_queue: list[dict] = []


def push(
    message: str,
    trigger: str = "system",
    icon: str = "🤖",
    action_prompt: str | None = None,
) -> None:
    """Enqueue a proactive message for the next UI poll."""
    with _lock:
        _queue.append({
            "message":      message,
            "trigger":      trigger,
            "icon":         icon,
            "action_prompt": action_prompt,
            "at":           time.time(),
        })
    log.info("proactive.push [%s]: %s", trigger, message[:80])


def pop_all() -> list[dict]:
    """Return and clear all non-expired pending messages."""
    now = time.time()
    with _lock:
        fresh = [m for m in _queue if now - m["at"] < _EXPIRE_S]
        _queue.clear()
    return [
        {
            "message":      m["message"],
            "trigger":      m["trigger"],
            "icon":         m["icon"],
            "action_prompt": m["action_prompt"],
        }
        for m in fresh
    ]


# ---------------------------------------------------------------------------
# Scheduled triggers
# ---------------------------------------------------------------------------

def _morning_trigger() -> None:
    push(
        "Good morning! It's time for your daily briefing — I can check your calendar, "
        "summarise overnight emails, and report the weather.",
        trigger="schedule",
        icon="☀️",
        action_prompt=(
            "Give me a morning briefing: list today's calendar events, summarise any "
            "important emails received overnight, and check the weather."
        ),
    )


def _evening_trigger() -> None:
    push(
        "It's evening — want me to run your end-of-day shutdown? "
        "I can close work apps and send a quick day summary.",
        trigger="schedule",
        icon="🌙",
        action_prompt=(
            "Run my end-of-day shutdown: save any open documents, close work apps, "
            "and send a brief day-summary to Slack."
        ),
    )


def register_jobs(scheduler) -> None:
    """Register APScheduler time-of-day proactive jobs."""
    try:
        scheduler.add_job(
            _morning_trigger,
            "cron",
            hour=_MORNING_HOUR, minute=0,
            day_of_week="mon-fri",
            id="proactive_morning",
            replace_existing=True,
        )
        scheduler.add_job(
            _evening_trigger,
            "cron",
            hour=_EVENING_HOUR, minute=0,
            day_of_week="mon-fri",
            id="proactive_evening",
            replace_existing=True,
        )
        log.info(
            "proactive: registered morning (%s:00) and evening (%s:00) weekday jobs",
            _MORNING_HOUR, _EVENING_HOUR,
        )
    except Exception as exc:
        log.warning("proactive.register_jobs failed: %s", exc)
