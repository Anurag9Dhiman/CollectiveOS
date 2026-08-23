"""
Voice gateway — WebSocket handler implementing the VoiceOS contract.

Implements the event contract defined in:
  https://github.com/Anurag9Dhiman/VoiceOS/tree/main/contract/schemas

VoiceToAgent events received:   session_start, user_utterance, interrupt,
                                  confirmation_response, session_query, session_end
AgentToVoice events sent:        ack, progress, confirmation_request,
                                  clarification_request, speak, done, error, task_update

Multi-step flow:
  1. Voice service sends user_utterance (e.g. "send a slack message then summarise email")
  2. Gateway ACKs immediately, runs agent.run() in a background thread
  3. If run() wants to call a write tool, it sends confirmation_request to voice
  4. Voice layer speaks the action and listens for user "yes"/"no"
  5. confirmation_response comes back → task resumes or cancels
  6. Final reply sent as speak + done
"""

import asyncio
import datetime
import json
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from collectiveos.db import connect, default_user_id


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class VoiceSession:
    """Tracks one live voice WebSocket connection."""

    def __init__(self, ws: WebSocket, session_id: str, user_id: str):
        self.ws = ws
        self.session_id = session_id
        self.user_id = user_id
        self.history: list[dict] = []

        # task_id → asyncio.Task (in-flight agent calls)
        self._tasks: dict[str, asyncio.Task] = {}

        # task_id → asyncio.Event (set when confirmation arrives)
        self._confirm_events: dict[str, asyncio.Event] = {}
        # task_id → decision dict
        self._confirm_decisions: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    async def _send(self, event: dict):
        try:
            await self.ws.send_json(event)
        except Exception:
            pass

    async def ack(self, task_id: str, text: str = "On it"):
        await self._send({"type": "ack", "task_id": task_id, "text": text})

    async def speak(self, task_id: str, text: str, priority: str = "low"):
        await self._send({"type": "speak", "task_id": task_id, "text": text, "priority": priority})

    async def done(self, task_id: str, outcome: str = "completed", summary: str = ""):
        await self._send({
            "type": "done",
            "task_id": task_id,
            "outcome": outcome,
            "summary_speak": summary,
        })

    async def error(self, task_id: str, text: str, recoverable: bool = True):
        await self._send({"type": "error", "task_id": task_id, "speak": text, "recoverable": recoverable})

    async def progress(self, task_id: str, text: str, priority: str = "low"):
        await self._send({"type": "progress", "task_id": task_id, "text": text, "priority": priority})

    async def confirmation_request(self, task_id: str, text: str, risk_class: str = "write"):
        await self._send({
            "type": "confirmation_request",
            "task_id": task_id,
            "speak": text,
            "priority": "high",
            "options": ["approve", "reject", "modify"],
            "risk_class": risk_class,
        })

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def handle(self, event: dict):
        t = event.get("type")
        if t == "session_start":
            await self._on_session_start(event)
        elif t == "user_utterance":
            await self._on_utterance(event)
        elif t == "interrupt":
            await self._on_interrupt(event)
        elif t == "confirmation_response":
            await self._on_confirmation(event)
        elif t == "session_query":
            await self._on_query(event)
        elif t == "session_end":
            await self._on_session_end(event)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _on_session_start(self, event: dict):
        resuming = event.get("resume", False)
        try:
            conn = connect()
            try:
                uid = _resolve_user_id(conn, self.user_id)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO voice_sessions (id, user_id, active_task_ids, entity_stack)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (self.session_id, uid, json.dumps([]), json.dumps([])),
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass  # DB not required for voice to work

        greeting = (
            "Resuming your session. What would you like to do?"
            if resuming
            else "Voice assistant ready. What can I help you with?"
        )
        task_id = _new_id()
        await self.ack(task_id, greeting)

    async def _on_utterance(self, event: dict):
        task_id = _new_id()
        text = event.get("text", "")
        router_class = event.get("router_class", "new_intent")

        # Small talk / lookup → quick inline reply without a background task
        if router_class in ("small_talk", "simple_lookup"):
            await self.ack(task_id, "Let me check…")
            asyncio.create_task(self._run_agent(task_id, text, quick=True))
            return

        # New intent or modify in-flight → full agent run
        await self.ack(task_id)
        t = asyncio.create_task(self._run_agent(task_id, text))
        self._tasks[task_id] = t
        t.add_done_callback(lambda _: self._tasks.pop(task_id, None))

    async def _on_interrupt(self, event: dict):
        target = event.get("target_task_id")
        if target and target in self._tasks:
            self._tasks[target].cancel()
            await self._send({"type": "task_update", "task_id": target, "status": "cancelled"})
        elif not target:
            # Cancel all in-flight tasks
            for tid, t in list(self._tasks.items()):
                t.cancel()
            self._tasks.clear()

    async def _on_confirmation(self, event: dict):
        task_id = event.get("task_id")
        if task_id and task_id in self._confirm_events:
            self._confirm_decisions[task_id] = {
                "decision": event.get("decision", "reject"),
                "modification": event.get("modification"),
            }
            self._confirm_events[task_id].set()

    async def _on_query(self, event: dict):
        task_id = _new_id()
        n = len(self._tasks)
        if n == 0:
            text = "No tasks are currently running."
        elif n == 1:
            text = "There is one task in progress."
        else:
            text = f"There are {n} tasks in progress."
        await self.speak(task_id, text, priority="high")
        await self.done(task_id, summary=text)

    async def _on_session_end(self, event: dict):
        for t in self._tasks.values():
            t.cancel()
        self._tasks.clear()
        try:
            conn = connect()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE voice_sessions SET ended_at = now() WHERE id = %s",
                        (self.session_id,),
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    async def _run_agent(self, task_id: str, text: str, quick: bool = False):
        """Run the CollectiveOS agent loop and stream events back."""
        from collectiveos import assistant_starter as agent
        from collectiveos import memory

        system_prompt = self._system_prompt(quick)

        # Save memory context async
        past = await asyncio.to_thread(memory.search, text)
        if past:
            system_prompt += "\n\nRelevant context from past conversations:\n" + past

        try:
            await self.progress(task_id, "Working on it…")

            # Run agent in thread so WebSocket isn't blocked
            reply = await asyncio.to_thread(
                agent.run, text,
                system=system_prompt,
                history=self.history,
            )

            # If reply contains a confirmation prompt (write tool), ask voice layer
            if _needs_voice_confirmation(reply):
                confirmed = await self._ask_confirmation(task_id, reply)
                if not confirmed:
                    msg = "Okay, I've cancelled that."
                    await self.speak(task_id, msg)
                    await self.done(task_id, outcome="failed", summary=msg)
                    return
                # Re-run with explicit approval in context
                system_prompt += "\n\nThe user has approved this action verbally. Proceed."
                reply = await asyncio.to_thread(
                    agent.run, text,
                    system=system_prompt,
                    history=self.history,
                )

            await self.speak(task_id, reply)
            await self.done(task_id, outcome="completed", summary=reply[:200])

            # Update in-session history
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})
            self.history = self.history[-20:]

            # Persist to memory
            await asyncio.to_thread(memory.save, text, reply)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            await self.error(task_id, f"Something went wrong: {exc}")

    async def _ask_confirmation(self, task_id: str, prompt: str) -> bool:
        """Send a confirmation_request, wait for the voice layer's response."""
        event = asyncio.Event()
        self._confirm_events[task_id] = event
        await self.confirmation_request(task_id, prompt)
        try:
            await asyncio.wait_for(event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            return False
        finally:
            self._confirm_events.pop(task_id, None)

        decision = self._confirm_decisions.pop(task_id, {}).get("decision", "reject")
        return decision == "approve"

    def _system_prompt(self, quick: bool = False) -> str:
        now = datetime.datetime.now(datetime.timezone.utc)
        date_str = now.strftime("%A, %B %d, %Y, %H:%M UTC")
        base = (
            f"You are a helpful personal voice assistant. Today is {date_str}.\n\n"

            "LANGUAGE RULE: The user may speak Hindi or English, or switch between them. "
            "Always reply in the SAME language the user just used in this turn. "
            "If they spoke Hindi, reply fully in Hindi. If English, reply in English. "
            "Never mix languages in a single reply unless the user explicitly does so.\n\n"

            "VOICE FORMAT: Responses must be short — 1 to 3 spoken sentences. "
            "No bullet points, no numbered lists, no markdown, no special characters.\n\n"

            "MULTI-STEP TASKS: When the user asks you to do several things, "
            "identify each step. Execute ONLY the first step. After you complete it, "
            "tell the user what you just did and ask whether to continue to the next step. "
            "Wait for their explicit confirmation before moving to any next step. "
            "If they say yes (or 'haan', 'theek hai', 'chalao', 'proceed', 'ok'), do the next step and repeat. "
            "If they say no (or 'nahi', 'ruko', 'stop', 'cancel'), stop and confirm you have stopped.\n\n"

            "WRITE/ACTION CONFIRMATION: Before calling any tool that sends a message, "
            "creates, deletes, or controls anything, describe exactly what you are about to do "
            "and ask a single yes/no question. Only call the tool after the user confirms. "
            "In Hindi ask: 'Kya main yeh kar doon?' or equivalent. "
            "In English ask: 'Shall I go ahead?' or equivalent."
        )
        if quick:
            base += "\n\nThis is a quick lookup — answer in one sentence in the user's language."
        return base


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return str(uuid.uuid4())


def _resolve_user_id(conn, voice_user_id: str) -> int:
    """Map the voice-layer's UUID user_id to the Postgres integer user id."""
    return default_user_id(conn)


def _needs_voice_confirmation(reply: str) -> bool:
    """
    Detect when the agent is pausing to ask the user for confirmation or
    permission before proceeding — including Hindi phrases. In voice sessions
    the confirmation goes through the contract, not the text channel.
    """
    lower = reply.lower()
    confirmation_phrases = [
        # English — write/action confirmation
        "shall i", "should i", "do you want me to", "would you like me to",
        "confirm", "are you sure", "proceed?", "go ahead?",
        "want me to", "shall i go ahead",
        # English — step transition
        "next step", "continue to the next", "move on to", "proceed to",
        "ready for the next", "shall i continue",
        # Hindi — write/action confirmation
        "kya main", "kar doon", "kar dun", "bhej doon", "bhej dun",
        "aage badhu", "aage badhun", "chahte hain", "chahte ho",
        "theek hai?", "confirm kar", "jaari rakhu",
        # Hindi — step transition
        "agle step", "agli step", "aage jaun", "continue karun",
        "next karna", "aage karna",
    ]
    return any(p in lower for p in confirmation_phrases)


# ---------------------------------------------------------------------------
# FastAPI WebSocket entry point
# ---------------------------------------------------------------------------

async def voice_ws_endpoint(websocket: WebSocket):
    """Mount this on /voice/ws in api.py."""
    await websocket.accept()
    session: VoiceSession | None = None

    try:
        # First message must be session_start
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        event = json.loads(raw)

        if event.get("type") != "session_start":
            await websocket.send_json({"type": "error", "task_id": _new_id(),
                                       "text": "First message must be session_start",
                                       "recoverable": False})
            return

        session = VoiceSession(
            ws=websocket,
            session_id=event.get("session_id", _new_id()),
            user_id=event.get("user_id", "default"),
        )
        await session.handle(event)

        # Main receive loop
        while True:
            raw = await websocket.receive_text()
            event = json.loads(raw)
            await session.handle(event)

    except asyncio.TimeoutError:
        if session:
            await session.error(_new_id(), "Connection timed out waiting for session_start",
                                recoverable=False)
    except WebSocketDisconnect:
        if session:
            await session._on_session_end({"type": "session_end",
                                           "session_id": session.session_id})
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "task_id": _new_id(),
                                       "text": str(exc), "recoverable": False})
        except Exception:
            pass
