"""
nav_queue.py
------------
Thread-safe in-memory registry for background nav tasks.

Every call to navigate_computer that exceeds the 30-second sync timeout
registers here so the UI and REST API can show live status.

States: pending → running → done | error | cancelled
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

_lock = threading.Lock()
_tasks: dict[str, dict] = {}
_MAX_HISTORY = 200  # keep the last N tasks to bound memory


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def submit(task: str) -> str:
    """Register a task as pending and return its short id."""
    tid = uuid.uuid4().hex[:8]
    entry = {
        "id": tid,
        "task": task,
        "label": task[:80],
        "status": "pending",
        "submitted_at": _now(),
        "started_at": None,
        "finished_at": None,
        "result": None,
    }
    with _lock:
        _tasks[tid] = entry
        _trim()
    return tid


def mark_running(tid: str) -> None:
    with _lock:
        if tid in _tasks:
            _tasks[tid]["status"] = "running"
            _tasks[tid]["started_at"] = _now()


def mark_done(tid: str, result: str, error: bool = False) -> None:
    with _lock:
        if tid in _tasks:
            _tasks[tid]["status"] = "error" if error else "done"
            _tasks[tid]["finished_at"] = _now()
            _tasks[tid]["result"] = result[:500] if result else ""


def cancel(tid: str) -> bool:
    """Cancel a pending or running task. Returns True if the state changed."""
    with _lock:
        t = _tasks.get(tid)
        if t and t["status"] in ("pending", "running"):
            t["status"] = "cancelled"
            t["finished_at"] = _now()
            return True
    return False


def get(tid: str) -> Optional[dict]:
    with _lock:
        t = _tasks.get(tid)
        return dict(t) if t else None


def list_all(limit: int = 50) -> list[dict]:
    with _lock:
        items = sorted(_tasks.values(), key=lambda t: t["submitted_at"], reverse=True)
    return [dict(t) for t in items[:limit]]


def is_cancelled(tid: str) -> bool:
    with _lock:
        t = _tasks.get(tid)
        return bool(t and t["status"] == "cancelled")


def _trim() -> None:
    if len(_tasks) > _MAX_HISTORY:
        oldest = sorted(_tasks.keys(),
                        key=lambda k: _tasks[k]["submitted_at"])
        for k in oldest[:len(_tasks) - _MAX_HISTORY]:
            del _tasks[k]
