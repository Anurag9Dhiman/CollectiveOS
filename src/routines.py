"""
Routines — DB CRUD for scheduled agent tasks.

A routine is a saved (name, prompt, cron-schedule) triple. The scheduler
in scheduler.py reads all enabled routines on startup and re-fires them
whenever the cron expression matches.

Two execution modes per routine:
  • prompt-only  — runs the full agent loop with the saved prompt text.
  • nav_task     — runs navigate_computer() on the Mac instead (or in addition).
    If nav_task is set, the computer-control result is delivered; prompt is
    used as fallback summary context when nav_task is empty.

Table is created here on first import so no manual migration is needed.
"""

from src.db import connect

_CREATE = """
CREATE TABLE IF NOT EXISTS routines (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    prompt      TEXT NOT NULL DEFAULT '',
    schedule    TEXT NOT NULL,
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    notify_via  TEXT NOT NULL DEFAULT 'slack'
                CHECK (notify_via IN ('notification', 'slack', 'telegram', 'push', 'both', 'none')),
    nav_task    TEXT,
    last_run_at TIMESTAMPTZ,
    last_result TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_MIGRATE = """
DO $$
BEGIN
    -- Widen notify_via constraint to include all current channels
    ALTER TABLE routines DROP CONSTRAINT IF EXISTS routines_notify_via_check;
    ALTER TABLE routines ADD CONSTRAINT routines_notify_via_check
        CHECK (notify_via IN ('notification', 'slack', 'telegram', 'push', 'both', 'none'));
    -- Add nav_task column for computer-control routines
    ALTER TABLE routines ADD COLUMN IF NOT EXISTS nav_task TEXT;
    -- Allow prompt to be empty (nav-only routines don't need a prompt)
    ALTER TABLE routines ALTER COLUMN prompt SET DEFAULT '';
EXCEPTION WHEN others THEN NULL;
END $$;
"""


def _bootstrap() -> None:
    try:
        conn = connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE)
                cur.execute(_MIGRATE)
        conn.close()
    except Exception:
        pass


_bootstrap()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def create(name: str, prompt: str, schedule: str,
           notify_via: str = "slack", nav_task: str | None = None) -> dict:
    conn = connect()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO routines (name, prompt, schedule, notify_via, nav_task)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, name, prompt, schedule, enabled, notify_via,
                          nav_task, last_run_at, last_result, created_at
                """,
                (name, prompt or "", schedule, notify_via, nav_task),
            )
            row = cur.fetchone()
    conn.close()
    return _row_to_dict(row)


def list_all(enabled_only: bool = False) -> list[dict]:
    conn = connect()
    with conn.cursor() as cur:
        q = (
            "SELECT id, name, prompt, schedule, enabled, notify_via, "
            "nav_task, last_run_at, last_result, created_at "
            "FROM routines"
        )
        if enabled_only:
            q += " WHERE enabled = TRUE"
        q += " ORDER BY id"
        cur.execute(q)
        rows = cur.fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get(routine_id: int) -> dict | None:
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, prompt, schedule, enabled, notify_via, "
            "nav_task, last_run_at, last_result, created_at "
            "FROM routines WHERE id = %s",
            (routine_id,),
        )
        row = cur.fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def update(routine_id: int, **fields) -> dict | None:
    """Update any subset of: name, prompt, schedule, enabled, notify_via, nav_task."""
    allowed = {"name", "prompt", "schedule", "enabled", "notify_via", "nav_task"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return get(routine_id)

    set_clause = ", ".join(f"{k} = %s" for k in updates)
    values = list(updates.values()) + [routine_id]

    conn = connect()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE routines SET {set_clause} "
                f"WHERE id = %s "
                f"RETURNING id, name, prompt, schedule, enabled, notify_via, "
                f"nav_task, last_run_at, last_result, created_at",
                values,
            )
            row = cur.fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def delete(routine_id: int) -> bool:
    conn = connect()
    with conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM routines WHERE id = %s RETURNING id", (routine_id,))
            deleted = cur.fetchone() is not None
    conn.close()
    return deleted


def record_run(routine_id: int, result: str) -> None:
    conn = connect()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE routines SET last_run_at = NOW(), last_result = %s WHERE id = %s",
                (result[:4000], routine_id),
            )
    conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row) -> dict:
    keys = ("id", "name", "prompt", "schedule", "enabled", "notify_via",
            "nav_task", "last_run_at", "last_result", "created_at")
    d = dict(zip(keys, row))
    for k in ("last_run_at", "created_at"):
        if d[k] is not None:
            d[k] = d[k].isoformat()
    return d
