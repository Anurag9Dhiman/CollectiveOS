"""
Routines — DB CRUD for scheduled agent tasks.

A routine is a saved (name, prompt, cron-schedule) triple. The scheduler
in scheduler.py reads all enabled routines on startup and re-fires them
whenever the cron expression matches.

Table is created here on first import so no manual migration is needed.
"""

from src.db import connect

_CREATE = """
CREATE TABLE IF NOT EXISTS routines (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    schedule    TEXT NOT NULL,
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    notify_via  TEXT NOT NULL DEFAULT 'notification'
                CHECK (notify_via IN ('notification', 'none', 'telegram', 'both', 'push')),
    last_run_at TIMESTAMPTZ,
    last_result TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# Widens the notify_via CHECK to include all output-bus channels on existing DBs.
_MIGRATE_NOTIFY_VIA = """
DO $$
BEGIN
    ALTER TABLE routines DROP CONSTRAINT IF EXISTS routines_notify_via_check;
    ALTER TABLE routines ADD CONSTRAINT routines_notify_via_check
        CHECK (notify_via IN ('notification', 'none', 'telegram', 'both', 'push'));
EXCEPTION WHEN others THEN NULL;
END $$;
"""


def _bootstrap() -> None:
    try:
        conn = connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE)
                cur.execute(_MIGRATE_NOTIFY_VIA)
        conn.close()
    except Exception:
        pass


_bootstrap()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def create(name: str, prompt: str, schedule: str,
           notify_via: str = "notification") -> dict:
    conn = connect()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO routines (name, prompt, schedule, notify_via)
                VALUES (%s, %s, %s, %s)
                RETURNING id, name, prompt, schedule, enabled, notify_via,
                          last_run_at, last_result, created_at
                """,
                (name, prompt, schedule, notify_via),
            )
            row = cur.fetchone()
    conn.close()
    return _row_to_dict(row)


def list_all(enabled_only: bool = False) -> list[dict]:
    conn = connect()
    with conn.cursor() as cur:
        if enabled_only:
            cur.execute(
                "SELECT id, name, prompt, schedule, enabled, notify_via, "
                "last_run_at, last_result, created_at "
                "FROM routines WHERE enabled = TRUE ORDER BY id"
            )
        else:
            cur.execute(
                "SELECT id, name, prompt, schedule, enabled, notify_via, "
                "last_run_at, last_result, created_at "
                "FROM routines ORDER BY id"
            )
        rows = cur.fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get(routine_id: int) -> dict | None:
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, prompt, schedule, enabled, notify_via, "
            "last_run_at, last_result, created_at "
            "FROM routines WHERE id = %s",
            (routine_id,),
        )
        row = cur.fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def update(routine_id: int, **fields) -> dict | None:
    """
    Update any subset of: name, prompt, schedule, enabled, notify_via.
    Returns the updated row, or None if not found.
    """
    allowed = {"name", "prompt", "schedule", "enabled", "notify_via"}
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
                f"last_run_at, last_result, created_at",
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
            "last_run_at", "last_result", "created_at")
    d = dict(zip(keys, row))
    # Make timestamps JSON-serialisable
    for k in ("last_run_at", "created_at"):
        if d[k] is not None:
            d[k] = d[k].isoformat()
    return d
