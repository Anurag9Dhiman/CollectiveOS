"""Voice gateway — WebSocket handler implementing the VoiceOS wire contract.

VoiceOS connects to ws://localhost:8000/v1/ws and speaks the voice_contract
JSON protocol (type-discriminated events). This module is the server side.

Wire format (both directions): JSON-serialised Pydantic models.
  VoiceOS → CollectiveOS: session_start | user_utterance | interrupt |
                          confirmation_response | session_query | session_end
  CollectiveOS → VoiceOS: ack | progress | confirmation_request | speak |
                          task_update | done | error

The gateway intentionally does NOT import voice_contract from the VoiceOS
package — CollectiveOS is a separate repo and the shared contract is small
enough to speak as plain JSON. The type strings are stable; if the contract
ever adds a new event, the worst case is an unrecognised-type log line.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("collectiveos.voice_gateway")


async def handle_voice_ws(ws: WebSocket) -> None:
    await ws.accept()
    session_id: str | None = None
    user_id: str = "unknown"

    try:
        while True:
            raw = await ws.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Voice gateway: invalid JSON, skipping")
                continue

            event_type = event.get("type")
            session_id = event.get("session_id", session_id)

            if event_type == "session_start":
                user_id = event.get("user_id", "unknown")
                resume = event.get("resume", False)
                logger.info("Voice session start: session=%s user=%s resume=%s", session_id, user_id, resume)
                await _send(ws, {
                    "type": "ack",
                    "task_id": str(uuid.uuid4()),
                    "text": "Connected to CollectiveOS. What can I help with?",
                    "ts": _now(),
                })

            elif event_type == "user_utterance":
                text = event.get("text", "").strip()
                entity_refs = event.get("entity_refs") or {}
                if not text:
                    continue
                task_id = str(uuid.uuid4())
                await _send(ws, {"type": "ack", "task_id": task_id, "text": "On it.", "ts": _now()})
                asyncio.create_task(_run_task(ws, task_id, text, entity_refs, user_id))

            elif event_type == "interrupt":
                target = event.get("target_task_id")
                mod_text = event.get("text", "")
                logger.info("Interrupt: target=%s text=%r", target, mod_text)
                # Task cancellation is not yet wired into the LangGraph checkpointer;
                # acknowledge receipt and let the inflight task finish naturally.
                await _send(ws, {
                    "type": "task_update",
                    "task_id": target or str(uuid.uuid4()),
                    "status": "running",
                    "waiting_reason": None,
                    "ts": _now(),
                })

            elif event_type == "confirmation_response":
                task_id = event.get("task_id", "")
                decision = event.get("decision")
                logger.info("Confirmation response: task=%s decision=%s", task_id, decision)
                # Wired into interrupt_before HITL in a future milestone.

            elif event_type == "session_query":
                query = event.get("query", "")
                logger.info("Session query: %r", query)
                await _send(ws, {
                    "type": "speak",
                    "task_id": str(uuid.uuid4()),
                    "text": "No tasks currently waiting on your input.",
                    "priority": "low",
                    "ts": _now(),
                })

            elif event_type == "session_end":
                logger.info("Voice session end: session=%s", session_id)
                break

            else:
                logger.warning("Voice gateway: unrecognised event type %r", event_type)

    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected: session=%s", session_id)
    except Exception:
        logger.exception("Voice gateway error: session=%s", session_id)
        try:
            await _send(ws, {
                "type": "error",
                "task_id": str(uuid.uuid4()),
                "code": "internal_error",
                "speak": "Something went wrong. Please try again.",
                "recoverable": True,
                "ts": _now(),
            })
        except Exception:
            pass


async def _run_task(
    ws: WebSocket,
    task_id: str,
    text: str,
    entity_refs: dict,
    user_id: str,
) -> None:
    from src.agent import run as agent_run
    from src import memory
    from src.api import _system_prompt

    try:
        prefix = ""
        if entity_refs:
            prefix = "Visual context: " + json.dumps(entity_refs) + "\n\n"

        past = memory.search(text)
        system = _system_prompt(past)

        loop = asyncio.get_event_loop()
        reply, _interrupted = await loop.run_in_executor(
            None, agent_run, prefix + text, system, f"voice_{user_id}"
        )

        memory.save(text, reply)

        await _send(ws, {
            "type": "speak",
            "task_id": task_id,
            "text": reply,
            "priority": "high",
            "ts": _now(),
        })
        await _send(ws, {
            "type": "done",
            "task_id": task_id,
            "summary_speak": "Done.",
            "ts": _now(),
        })

    except Exception as exc:
        logger.exception("Agent task failed: task=%s", task_id)
        await _send(ws, {
            "type": "error",
            "task_id": task_id,
            "code": "agent_error",
            "speak": f"I ran into a problem: {exc}",
            "recoverable": True,
            "ts": _now(),
        })


async def _send(ws: WebSocket, payload: dict) -> None:
    await ws.send_text(json.dumps(payload))


def _now() -> str:
    return datetime.now(UTC).isoformat()
