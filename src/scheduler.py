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

def _run_routine(routine_id: int, name: str, prompt: str,
                 notify_via: str, nav_task: str | None = None) -> None:
    """Executed by APScheduler in a background thread.

    Execution modes (in priority order):
      1. nav_task set → navigate_computer() controls the Mac autonomously.
      2. prompt set   → full agent loop answers the prompt.
      3. both set     → nav_task runs first; result is appended to prompt for
                        the agent loop so the AI can summarise / act on it.
    """
    log.info("Routine %d (%s) firing — mode=%s", routine_id, name,
             "nav" if nav_task else "chat")

    from src.api import _system_prompt
    from src import memory

    result = ""

    # ── Nav task (computer automation) ───────────────────────────────────────
    if nav_task:
        try:
            import asyncio
            from src.agents.nav_agent import navigate_computer as _nav
            result = asyncio.run(_nav(nav_task.strip()))
        except Exception as exc:
            log.error("Routine %d nav_task failed: %s", routine_id, exc)
            result = f"[nav error] {exc}"

    # ── Chat prompt (agent loop) ──────────────────────────────────────────────
    if prompt:
        try:
            from src.assistant_starter import run
            combined = prompt
            if result:
                combined = f"{prompt}\n\n[Computer task result: {result}]"
            past = memory.search(combined)
            result = run(combined, system=_system_prompt(past))
        except Exception as exc:
            log.error("Routine %d prompt failed: %s", routine_id, exc)
            result = result or f"Error: {exc}"

    if not result:
        result = "Routine fired but produced no output."

    _db.record_run(routine_id, result)
    # output_bus handles all channel dispatch (notification / slack / push / etc.)
    output_bus.deliver(name, result, channel=notify_via)

    try:
        from src.activity import log_event
        status = "error" if result.startswith(("Error:", "[nav error]")) else "ok"
        log_event("routine", name, f"[{status}] {result[:300]}")
    except Exception:
        pass


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
            "prompt":     r["prompt"] or "",
            "notify_via": r["notify_via"],
            "nav_task":   r.get("nav_task"),
        },
    )
    mode = "nav+chat" if r.get("nav_task") and r.get("prompt") else \
           "nav" if r.get("nav_task") else "chat"
    log.info("Scheduled routine %d (%s) → %s [%s]", r["id"], r["name"], r["schedule"], mode)


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

        # Proactive screen awareness (opt-in via SCREEN_WATCHER_ENABLED=1)
        try:
            from src import screen_watcher as _sw
            _sw.register_job(_scheduler)
        except Exception as exc:
            log.warning("Could not register screen watcher: %s", exc)

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
