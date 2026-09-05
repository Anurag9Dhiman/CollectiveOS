"""
Wearable streaming layer — always-on WebSocket handler.

Implements the VisionClaw architecture: a continuous stream of transcripts and
optional camera frames from wearable hardware (glasses, phone, watch) is
processed by a lightweight intent classifier before the full agent is invoked.

Wire protocol (JSON over WebSocket, both directions):

  Client → Server
    {"type": "transcript", "text": "...", "device_id": "glasses-1"}
    {"type": "frame",      "image_b64": "...", "device_id": "glasses-1",
                           "image_mime": "image/jpeg"}
    {"type": "context",    "text": "...", "image_b64": "...",
                           "image_mime": "image/jpeg", "device_id": "..."}
    {"type": "ping"}

  Server → Client
    {"type": "ack",   "message": "..."}
    {"type": "reply", "text": "...", "triggered": true}
    {"type": "pong"}
    {"type": "error", "message": "..."}

Intent detection:
  A cheap Gemini Flash call decides YES/NO: does this transcript indicate the
  user wants assistance from their AI assistant?  Runs only when a transcript
  arrives — frame-only events skip classification and update visual context.

  The threshold can be tuned via the WEARABLE_INTENT_THRESHOLD env var
  (default: permissive — any YES triggers the agent).

Session state:
  Each WebSocket connection maintains a short rolling context window:
  - last few transcript lines (for multi-turn awareness)
  - last image frame (attached to the agent call if present)
  This context resets when the connection closes.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

log = logging.getLogger(__name__)

_INTENT_MODEL  = os.environ.get("GEMINI_ROUTER_MODEL", "gemini-2.0-flash-lite")
_AGENT_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
_MAX_CONTEXT   = 6      # rolling transcript lines kept for multi-turn context
_MAX_FRAME_KB  = 2048   # max base64 bytes for an attached frame (~1.5 MB raw)

_INTENT_SYSTEM = (
    "You are a trigger classifier for a wearable AI assistant. "
    "The user is going about their day wearing a device that transcribes "
    "ambient speech. Decide whether the latest transcript snippet is "
    "a direct request or question addressed to the AI assistant. "
    "Reply with exactly one word: YES or NO. "
    "Examples that are YES: 'Hey assistant what time is my meeting', "
    "'remind me to call Alice', 'what's the weather like', "
    "'add milk to my shopping list', 'play some music'. "
    "Examples that are NO: background conversation, TV audio, "
    "the user talking to someone else, monologues with no clear ask."
)


# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------

def _classify_intent(transcript: str, context_lines: list[str]) -> bool:
    """
    Returns True if the transcript looks like a request to the assistant.
    Uses Gemini Flash (cheap) — falls back to False on any error.
    """
    if not transcript.strip():
        return False
    try:
        from google import genai
        from google.genai import types as _gtypes

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return False

        ctx = ""
        if context_lines:
            ctx = "Recent context:\n" + "\n".join(context_lines[-4:]) + "\n\n"

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=_INTENT_MODEL,
            contents=f"{ctx}Latest: {transcript.strip()}",
            config=_gtypes.GenerateContentConfig(
                system_instruction=_INTENT_SYSTEM,
                temperature=0.0,
            ),
        )
        answer = (resp.text or "").strip().upper()
        return answer.startswith("YES")
    except Exception as exc:
        log.debug("Intent classifier error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------

def _run_agent(text: str, context_lines: list[str],
               image_b64: str | None, image_mime: str) -> str:
    """
    Run the main agent with the transcript + rolling context as the user
    message, and return the reply string.

    Phase 4: when image_b64 is present (Frame glasses camera frame), store it
    via set_first_person_frame() so that navigate_computer picks it up as
    first_person_frame — the agent can then see what the user physically sees.
    """
    import base64
    from src import memory, conversations
    from src.agent import run as agent_run
    from src.api import _system_prompt  # reuse the same system prompt builder

    # Phase 4: forward Frame camera frame to the nav agent
    if image_b64:
        try:
            from src.agents.nav_agent import set_first_person_frame
            set_first_person_frame(base64.b64decode(image_b64))
        except Exception:
            pass
    else:
        try:
            from src.agents.nav_agent import set_first_person_frame
            set_first_person_frame(None)
        except Exception:
            pass

    # Build a composite user message from rolling context + latest transcript
    if context_lines:
        user_message = "[Wearable context]\n" + "\n".join(context_lines) + f"\n\n{text}"
    else:
        user_message = text

    past = memory.search_with_graph(text)
    system = _system_prompt(past)

    conv_id = conversations.create()
    conversations.save_message(conv_id, "user", user_message)

    reply, _interrupted, _destructive = agent_run(
        user_message,
        system_prompt=system,
        thread_id=f"wearable-{conv_id}",
        image_b64=image_b64 or None,
        image_mime=image_mime or "image/jpeg",
    )

    conversations.save_message(conv_id, "assistant", reply)
    memory.save_smart(user_message, reply)
    return reply


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def handle_wearable_ws(websocket: WebSocket, token: str) -> None:
    """
    Main entry point — called from api.py for each connecting wearable client.
    *token* has already been validated by the caller.
    """
    import asyncio
    import json

    await websocket.accept()
    await websocket.send_text(json.dumps({"type": "ack", "message": "Connected to wearable stream."}))

    # Per-connection state
    context_lines: deque[str] = deque(maxlen=_MAX_CONTEXT)
    last_image_b64: str | None = None
    last_image_mime: str = "image/jpeg"

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

            # ── frame: update visual context only, no intent check ──────────
            if msg_type == "frame":
                b64 = msg.get("image_b64", "")
                if b64 and len(b64) <= _MAX_FRAME_KB * 1024:
                    last_image_b64 = b64
                    last_image_mime = msg.get("image_mime", "image/jpeg")
                await websocket.send_text(json.dumps({"type": "ack", "message": "Frame received."}))
                continue

            # ── transcript: classify intent, run agent if triggered ──────────
            if msg_type in ("transcript", "context"):
                text = (msg.get("text") or "").strip()
                if not text:
                    await websocket.send_text(json.dumps({"type": "ack", "message": "Empty transcript."}))
                    continue

                # For "context" type, also update the image
                if msg_type == "context":
                    b64 = msg.get("image_b64", "")
                    if b64 and len(b64) <= _MAX_FRAME_KB * 1024:
                        last_image_b64 = b64
                        last_image_mime = msg.get("image_mime", "image/jpeg")

                triggered = _classify_intent(text, list(context_lines))
                context_lines.append(text)

                if not triggered:
                    await websocket.send_text(json.dumps({
                        "type": "ack",
                        "message": "Transcript received (no intent detected).",
                        "triggered": False,
                    }))
                    continue

                # Intent detected — run the agent in a thread so we don't
                # block the event loop
                loop = asyncio.get_event_loop()
                reply = await loop.run_in_executor(
                    None,
                    _run_agent,
                    text,
                    list(context_lines),
                    last_image_b64,
                    last_image_mime,
                )

                # Phase 4: format reply for Frame in-lens display
                try:
                    from src.connectors.frame_wearable import format_for_frame
                    frame_text = format_for_frame(reply)
                except Exception:
                    frame_text = reply[:150]

                # Deliver via WebSocket AND output_bus (notification/push)
                await websocket.send_text(json.dumps({
                    "type": "reply",
                    "text": reply,
                    "frame_display": frame_text,   # pre-formatted for Frame glasses
                    "triggered": True,
                }))

                try:
                    from src.output_bus import deliver
                    deliver("Wearable", reply[:300], channel="notification")
                except Exception:
                    pass

                # Clear image after use (stale frames shouldn't persist)
                last_image_b64 = None
                continue

            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Unknown message type: {msg_type!r}",
            }))

    except WebSocketDisconnect:
        log.debug("Wearable WebSocket disconnected.")
    except Exception as exc:
        log.warning("Wearable stream error: %s", exc)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
