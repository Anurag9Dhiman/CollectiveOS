"""
Proactive condition watchers — run a prompt on an interval, fire when condition is met.

Flow per watcher tick:
  1. Run `prompt` through the agent loop (same as routines).
  2. Ask Gemini Flash to evaluate `condition` against the result → YES / NO.
  3. If YES and not triggered within the last interval, deliver via output_bus
     and update last_triggered.
  4. Always update last_checked.
"""

import logging
import os
from datetime import datetime, timezone

from src.db import connect, default_user_id

log = logging.getLogger(__name__)

_CONDITION_MODEL = os.environ.get("GEMINI_ROUTER_MODEL", "gemini-2.0-flash-lite")


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def list_all(enabled_only: bool = False) -> list[dict]:
    conn = connect()
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT id, name, prompt, condition, interval_min, enabled,
                       notify_via, last_checked, last_triggered, last_result, created_at
                FROM watchers
            """
            if enabled_only:
                sql += " WHERE enabled = TRUE"
            sql += " ORDER BY created_at DESC"
            cur.execute(sql)
            return [_row(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get(watcher_id: int) -> dict | None:
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, name, prompt, condition, interval_min, enabled,
                          notify_via, last_checked, last_triggered, last_result, created_at
                   FROM watchers WHERE id = %s""",
                (watcher_id,),
            )
            row = cur.fetchone()
            return _row(row) if row else None
    finally:
        conn.close()


def create(name: str, prompt: str, condition: str,
           interval_min: int = 60, notify_via: str = "notification") -> dict:
    conn = connect()
    try:
        user_id = default_user_id(conn)
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO watchers (user_id, name, prompt, condition, interval_min, notify_via)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING id, name, prompt, condition, interval_min, enabled,
                             notify_via, last_checked, last_triggered, last_result, created_at""",
                (user_id, name, prompt, condition, interval_min, notify_via),
            )
            row = cur.fetchone()
        conn.commit()
        return _row(row)
    finally:
        conn.close()


def update(watcher_id: int, **fields) -> dict | None:
    allowed = {"name", "prompt", "condition", "interval_min", "enabled", "notify_via"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return get(watcher_id)
    conn = connect()
    try:
        sets = ", ".join(f"{k} = %s" for k in updates)
        vals = list(updates.values()) + [watcher_id]
        with conn.cursor() as cur:
            cur.execute(
                f"""UPDATE watchers SET {sets}
                    WHERE id = %s
                    RETURNING id, name, prompt, condition, interval_min, enabled,
                              notify_via, last_checked, last_triggered, last_result, created_at""",
                vals,
            )
            row = cur.fetchone()
        conn.commit()
        return _row(row) if row else None
    finally:
        conn.close()


def delete(watcher_id: int) -> bool:
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM watchers WHERE id = %s RETURNING id", (watcher_id,))
            deleted = cur.fetchone() is not None
        conn.commit()
        return deleted
    finally:
        conn.close()


def _record_check(watcher_id: int, result: str, triggered: bool) -> None:
    conn = connect()
    try:
        now = datetime.now(timezone.utc)
        with conn.cursor() as cur:
            if triggered:
                cur.execute(
                    "UPDATE watchers SET last_checked=%s, last_triggered=%s, last_result=%s WHERE id=%s",
                    (now, now, result[:2000], watcher_id),
                )
            else:
                cur.execute(
                    "UPDATE watchers SET last_checked=%s, last_result=%s WHERE id=%s",
                    (now, result[:2000], watcher_id),
                )
        conn.commit()
    finally:
        conn.close()


def _row(r) -> dict:
    keys = ["id", "name", "prompt", "condition", "interval_min", "enabled",
            "notify_via", "last_checked", "last_triggered", "last_result", "created_at"]
    d = dict(zip(keys, r))
    for k in ("last_checked", "last_triggered", "created_at"):
        if d.get(k):
            d[k] = d[k].isoformat()
    return d


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

def _check_condition(condition: str, result: str) -> bool:
    """Ask a cheap Gemini call whether `condition` is satisfied given `result`."""
    import google.genai as genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return False
    try:
        client = genai.Client(api_key=api_key)
        prompt = (
            f"Given the following information:\n{result}\n\n"
            f"Is this condition true? {condition}\n"
            "Reply with only YES or NO."
        )
        resp = client.models.generate_content(
            model=_CONDITION_MODEL,
            contents=prompt,
        )
        answer = (resp.text or "").strip().upper()
        return answer.startswith("YES")
    except Exception as exc:
        log.warning("Condition check failed: %s", exc)
        return False


def evaluate(watcher: dict) -> None:
    """Run one watcher: prompt → condition check → notify if met."""
    watcher_id = watcher["id"]
    name       = watcher["name"]
    prompt     = watcher["prompt"]
    condition  = watcher["condition"]
    notify_via = watcher["notify_via"]

    log.info("Watcher %d (%s) evaluating", watcher_id, name)

    # Run the prompt through the agent
    from src.assistant_starter import run
    from src.api import _system_prompt
    from src import memory, output_bus

    try:
        past   = memory.search(prompt)
        result = run(prompt, system=_system_prompt(past))
    except Exception as exc:
        log.error("Watcher %d prompt failed: %s", watcher_id, exc)
        _record_check(watcher_id, f"[Error] {exc}", triggered=False)
        return

    # Check condition
    triggered = _check_condition(condition, result)
    _record_check(watcher_id, result, triggered=triggered)

    if triggered:
        log.info("Watcher %d (%s) condition met — notifying via %s", watcher_id, name, notify_via)
        output_bus.deliver(f"🔔 {name}", result, channel=notify_via)


def check_due() -> None:
    """Called by the scheduler every minute — finds due watchers and evaluates them."""
    import threading
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    try:
        all_w = list_all(enabled_only=True)
    except Exception as exc:
        log.warning("Could not load watchers: %s", exc)
        return

    for w in all_w:
        last = w.get("last_checked")
        interval = timedelta(minutes=max(1, w.get("interval_min", 60)))
        if last is None:
            due = True
        else:
            from dateutil.parser import isoparse
            due = (now - isoparse(last)) >= interval

        if due:
            threading.Thread(target=evaluate, args=(w,), daemon=True).start()
