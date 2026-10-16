"""
Robot learning stream — WebSocket handler for robot camera feed.

The robot connects here to stream its camera view and receive tasks.
The nav agent runs with `robot_camera_frame=<jpeg>` and `record=True`
so every step (before-screenshot + action) is written to
data/demonstrations/ for later imitation learning.

Wire protocol (JSON over WebSocket, both directions):

  Client → Server
    {"type": "frame",  "image_b64": "...", "device_id": "robot-1",
                       "image_mime": "image/jpeg"}
    {"type": "task",   "text": "open the fridge door",
                       "device_id": "robot-1",  "image_b64": "...",
                       "image_mime": "image/jpeg"}
    {"type": "ping"}

  Server → Client
    {"type": "ack",   "message": "..."}
    {"type": "reply", "text": "...", "demo_path": "...", "triggered": true}
    {"type": "pong"}
    {"type": "error", "message": "..."}

Task mode:
  A "task" message triggers the nav agent synchronously in a thread pool.
  The attached image_b64 (if present) is used as the robot's point-of-view
  context for that run.  Recording is always on.

Frame mode:
  A "frame" message updates the stored camera view without triggering a run.
  Useful for streaming continuous frames that the next task will pick up.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect

log = logging.getLogger(__name__)

_MAX_FRAME_KB = 2048   # same limit as wearable stream


# ---------------------------------------------------------------------------
# Nav agent invocation (blocking — run in thread pool)
# ---------------------------------------------------------------------------

def _run_robot_task(task: str, context_lines: list[str],
                    robot_frame: Optional[bytes]) -> dict:
    """
    Run the nav agent with robot camera context and recording enabled.
    Returns a dict with status, result, steps, and demo_path.
    """
    from src.agents.nav_agent import _get_agent

    async def _async_run():
        return await _get_agent().run(
            task,
            context="\n".join(context_lines) if context_lines else "",
            robot_camera_frame=robot_frame,
            record=True,
        )

    result = asyncio.run(_async_run())

    demo_path = ""
    if result.steps:
        # _save_demos writes the file; reconstruct path for the reply
        import time
        from pathlib import Path
        demo_dir = Path("data/demonstrations")
        if demo_dir.exists():
            files = sorted(demo_dir.glob("demo_*.json"), key=lambda p: p.stat().st_mtime)
            if files:
                demo_path = str(files[-1])

    return {
        "status": result.status,
        "result": result.result,
        "steps": len(result.steps),
        "demo_path": demo_path,
    }


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def handle_robot_ws(websocket: WebSocket, token: str) -> None:
    """
    Main entry point — called from api.py for each connecting robot client.
    *token* has already been validated by the caller.
    """
    await websocket.accept()
    await websocket.send_text(json.dumps({"type": "ack", "message": "Robot stream connected."}))

    # Per-connection state
    context_lines: list[str] = []
    last_frame: Optional[bytes] = None
    last_mime: str = "image/jpeg"

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg: dict[str, Any] = json.loads(raw)
            except Exception:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON."}))
                continue

            msg_type = msg.get("type", "")

            # ── ping / pong ─────────────────────────────────────────────────
            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            # ── frame: update stored camera view ────────────────────────────
            if msg_type == "frame":
                b64 = msg.get("image_b64", "")
                if b64 and len(b64) <= _MAX_FRAME_KB * 1024:
                    last_frame = base64.b64decode(b64)
                    last_mime = msg.get("image_mime", "image/jpeg")
                await websocket.send_text(json.dumps({"type": "ack", "message": "Frame stored."}))
                continue

            # ── task: run nav agent with recording ──────────────────────────
            if msg_type == "task":
                text = (msg.get("text") or "").strip()
                if not text:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Empty task."}))
                    continue

                # Task message may also include a fresh frame
                b64 = msg.get("image_b64", "")
                if b64 and len(b64) <= _MAX_FRAME_KB * 1024:
                    last_frame = base64.b64decode(b64)
                    last_mime = msg.get("image_mime", "image/jpeg")

                log.info("Robot task received: %s", text[:80])
                context_lines.append(text)

                loop = asyncio.get_event_loop()
                outcome = await loop.run_in_executor(
                    None,
                    _run_robot_task,
                    text,
                    list(context_lines),
                    last_frame,
                )

                await websocket.send_text(json.dumps({
                    "type": "reply",
                    "text": outcome["result"],
                    "status": outcome["status"],
                    "steps": outcome["steps"],
                    "demo_path": outcome["demo_path"],
                    "triggered": True,
                }))

                # Clear frame after use — stale frames shouldn't persist
                last_frame = None
                continue

            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Unknown message type: {msg_type!r}",
            }))

    except WebSocketDisconnect:
        log.debug("Robot WebSocket disconnected.")
    except Exception as exc:
        log.warning("Robot stream error: %s", exc)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
