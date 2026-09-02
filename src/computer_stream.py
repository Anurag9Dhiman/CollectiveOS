"""
Computer-use progress bus — a thread-safe event queue that computer.py pushes
to while running and GET /computer/stream drains as Server-Sent Events.

Single global queue because the assistant is single-user: at most one
computer_use run happens at a time.  Clients that connect mid-run receive
all buffered events that haven't been consumed yet; events older than the
current run are automatically discarded when a new run starts.

Event shapes (JSON-serialisable dicts):

  {"event": "start",  "task": "...", "run_id": "..."}
  {"event": "action", "run_id": "...", "iteration": N,
   "action": {...},   "screenshot_b64": "...",
   "verify": "PROCEED|STUCK|ERROR|DONE|None"}
  {"event": "done",   "run_id": "...", "result": "...", "iterations": N}
  {"event": "stop"}   — emitted when the run is aborted via /computer/stop
"""

from __future__ import annotations

import queue
import threading
import uuid

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_q: queue.Queue = queue.Queue(maxsize=200)
_stop_event   = threading.Event()
_current_run: str | None = None          # run_id of the active task, or None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Producer API  (called from computer.py)
# ---------------------------------------------------------------------------

def begin_run(task: str) -> str:
    """Signal the start of a new computer-use run. Returns the run_id."""
    global _current_run
    run_id = uuid.uuid4().hex[:8]
    with _lock:
        _current_run = run_id
        _stop_event.clear()
        # Drain stale events from any previous run
        while not _q.empty():
            try:
                _q.get_nowait()
            except queue.Empty:
                break
    _put({"event": "start", "task": task, "run_id": run_id})
    return run_id


def emit_action(run_id: str, iteration: int,
                action: dict, screenshot_b64: str | None,
                verify: str | None = None) -> None:
    """Emit one action step.  screenshot_b64 may be None if capture failed."""
    _put({
        "event": "action",
        "run_id": run_id,
        "iteration": iteration,
        "action": action,
        "screenshot_b64": screenshot_b64 or "",
        "verify": verify,
    })


def end_run(run_id: str, result: str, iterations: int) -> None:
    """Signal successful or partial completion."""
    global _current_run
    with _lock:
        _current_run = None
    _put({"event": "done", "run_id": run_id,
          "result": result, "iterations": iterations})


def should_stop() -> bool:
    """computer.py polls this each iteration to honour a stop request."""
    return _stop_event.is_set()


def request_stop() -> bool:
    """Called by POST /computer/stop.  Returns True if a run was active."""
    with _lock:
        active = _current_run is not None
    if active:
        _stop_event.set()
        _put({"event": "stop"})
    return active


def _put(event: dict) -> None:
    try:
        _q.put_nowait(event)
    except queue.Full:
        # Drop oldest event to make room
        try:
            _q.get_nowait()
        except queue.Empty:
            pass
        try:
            _q.put_nowait(event)
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Consumer API  (called from the SSE endpoint in api.py)
# ---------------------------------------------------------------------------

def event_stream(timeout: float = 30.0):
    """
    Generator that yields JSON-serialisable dicts from the queue.
    Yields None on timeout (caller should send a keep-alive comment).
    Stops after a 'done' or 'stop' event so the SSE response closes cleanly.
    """
    import json
    while True:
        try:
            event = _q.get(timeout=timeout)
            yield event
            if event.get("event") in ("done", "stop"):
                return
        except queue.Empty:
            yield None  # keep-alive
