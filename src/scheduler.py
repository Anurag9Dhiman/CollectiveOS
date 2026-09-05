"""
Scheduler — runs saved routines automatically on their cron schedule.

Uses APScheduler's BackgroundScheduler so it lives inside the same process
as the FastAPI server. On startup, all enabled routines are loaded from
Postgres and scheduled. When a routine fires:

  1. Calls the full agent loop (run()) with the routine's saved prompt.
  2. Sends the result as a Mac notification (title = routine name).
  3. Records last_run_at + last_result in the DB.

When you create, update, or delete a routine via the API, call
reload_routine() / remove_job() so the live scheduler reflects the change
without a server restart.
"""

import logging
import os

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src import routines as _db
from src import output_bus

log = logging.getLogger(__name__)

_scheduler = BackgroundScheduler(timezone="UTC")


# ---------------------------------------------------------------------------
# Internal job function
# ---------------------------------------------------------------------------

def _run_routine(routine_id: int, name: str, prompt: str, notify_via: str) -> None:
    """Executed by APScheduler in a background thread."""
    log.info("Routine %d (%s) firing", routine_id, name)

    # Import here to avoid circular imports at module load
    from src.assistant_starter import run
    from src.api import _system_prompt
    from src import memory

    try:
        past = memory.search(prompt)
        result = run(prompt, system=_system_prompt(past))
    except Exception as exc:
        log.error("Routine %d failed: %s", routine_id, exc)
        result = f"Error: {exc}"

    _db.record_run(routine_id, result)
    output_bus.deliver(name, result, channel=notify_via)

    if notify_via in ("notification", "both"):
        _send_notification(name, result)
    if notify_via in ("telegram", "both"):
        _send_telegram(name, result)

    try:
        from src.activity import log_event
        status = "error" if result.startswith("Error:") else "ok"
        log_event("routine", name, f"[{status}] {result[:300]}")
    except Exception:
        pass


def _send_notification(title: str, body: str) -> None:
    import platform
    import subprocess

    if platform.system() != "Darwin":
        return

    safe_title = title.replace('"', "'")
    # Trim body for the notification banner (250 chars is a practical limit)
    safe_body = body[:250].replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{safe_body}" with title "{safe_title}"'
    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, timeout=10,
        )
    except Exception as exc:
        log.warning("Notification failed: %s", exc)


def _send_telegram(title: str, body: str) -> None:
    """Send routine result to the configured Telegram chat."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        log.warning("Telegram notify skipped — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        return

    import requests

    # Telegram messages: 4096 char limit; trim body so title + separator fits
    max_body = 4000 - len(title) - 10
    text = f"*{title}*\n\n{body[:max_body]}"
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
    except Exception as exc:
        log.warning("Telegram send failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all() -> None:
    """Load all enabled routines from the DB and schedule them."""
    try:
        rows = _db.list_all(enabled_only=True)
    except Exception as exc:
        log.warning("Could not load routines from DB: %s", exc)
        return

    for r in rows:
        _schedule(r)
    log.info("Scheduler: loaded %d routine(s)", len(rows))


def _schedule(r: dict) -> None:
    job_id = f"routine_{r['id']}"
    try:
        trigger = CronTrigger.from_crontab(r["schedule"], timezone="UTC")
    except Exception as exc:
        log.warning("Routine %d has invalid schedule %r: %s", r["id"], r["schedule"], exc)
        return

    _scheduler.add_job(
        _run_routine,
        trigger=trigger,
        id=job_id,
        replace_existing=True,
        kwargs={
            "routine_id": r["id"],
            "name":       r["name"],
            "prompt":     r["prompt"],
            "notify_via": r["notify_via"],
        },
    )
    log.info("Scheduled routine %d (%s) → %s", r["id"], r["name"], r["schedule"])


def reload_routine(routine_id: int) -> None:
    """Re-read a single routine from DB and reschedule (or remove if disabled)."""
    r = _db.get(routine_id)
    if r is None or not r["enabled"]:
        remove_job(routine_id)
    else:
        _schedule(r)


def remove_job(routine_id: int) -> None:
    job_id = f"routine_{routine_id}"
    if _scheduler.get_job(job_id):
        _scheduler.remove_job(job_id)


def start() -> None:
    if not _scheduler.running:
        _scheduler.start()
        # Watcher polling — runs every 60 s, then each watcher decides if it's due
        _scheduler.add_job(
            _run_watchers,
            "interval",
            seconds=60,
            id="watcher_poll",
            replace_existing=True,
        )
        # Morning briefing — reads .briefing_config.json and schedules if enabled
        try:
            from src import briefing as _briefing
            _briefing.register_job(_scheduler)
        except Exception as exc:
            log.warning("Could not register briefing job: %s", exc)
        log.info("APScheduler started")


def _run_watchers() -> None:
    try:
        from src import watchers as _watchers
        _watchers.check_due()
    except Exception as exc:
        log.warning("Watcher poll error: %s", exc)


def shutdown() -> None:
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
