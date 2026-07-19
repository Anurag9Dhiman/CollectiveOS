"""
FastAPI layer — exposes the assistant over HTTP.

Endpoints
---------
  GET  /                         Web chat UI (static/index.html)
  GET  /ask?q=<message>&token=<token>  Plain-text reply (iOS Shortcuts / curl).
  POST /chat                     Send a message, get a full reply (JSON).
  POST /chat/stream              Send a message, stream reply tokens (SSE).
  GET  /history/{conversation_id} Return stored messages for a conversation.
  GET  /health                   Liveness check.

Auth
----
  /ask:   pass token as ?token= query param OR Authorization: Bearer header.
  /chat*: Authorization: Bearer <API_TOKEN> header only.
  API_TOKEN is set in your .env file.

iOS Shortcuts setup
-------------------
  1. Create a new Shortcut.
  2. Add action: "Ask for Input" (Text) → name it "Message".
  3. Add action: "Get Contents of URL"
       URL: http://<your-mac-ip>:8000/ask?token=<API_TOKEN>&q=[Shortcut Input]
       Method: GET
  4. Add action: "Show Result" (or "Speak Text" for voice).
  5. Run it from the Home Screen, Siri, or the share sheet.

Run
---
  uvicorn src.api:app --reload --port 8000
  Then open http://localhost:8000 in your browser.
"""

import asyncio
import datetime
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

from src.assistant_starter import run, run_stream
from src import conversations, memory, permissions

_TZ_NAME = os.environ.get("TIMEZONE", "UTC")


_READ_TOOLS = (
    "get_calendar_events, get_recent_emails, search_emails, list_drive_files, "
    "read_drive_file, get_tasks, get_projects, get_devices, get_device_state, "
    "spotify_now_playing, spotify_get_devices, get_system_info, get_wifi_info"
)
_WRITE_TOOLS = (
    "create_event, create_draft, send_email, add_task, complete_task, update_task, "
    "control_device, set_light, spotify_control, spotify_set_volume, spotify_search_play, "
    "show_notification, open_application, set_system_volume"
)


def _system_prompt(past: str = "") -> str:
    """Build the system prompt, stamped with the current date and time."""
    now = datetime.datetime.now(datetime.timezone.utc)
    date_str = now.strftime(f"%A, %B %-d, %Y, %H:%M {_TZ_NAME}")
    prompt = (
        f"You are a helpful personal assistant.\n"
        f"Today is {date_str}.\n\n"
        f"PERMISSION RULES — follow exactly:\n"
        f"- Read tools ({_READ_TOOLS}): call freely without asking.\n"
        f"- Write/action tools ({_WRITE_TOOLS}): you MUST describe exactly what you "
        f"are about to do and only call the tool after the user gives an explicit "
        f"go-ahead such as 'yes', 'ok', 'do it', 'send it', 'confirm', or 'proceed'. "
        f"Never call a write/action tool without explicit user approval in this turn."
    )
    if past:
        prompt += "\n\nRelevant context from past conversations:\n" + past
    return prompt

_HERE   = os.path.dirname(os.path.abspath(__file__))
_STATIC = os.path.join(_HERE, "..", "static")

app = FastAPI(title="Personal Assistant API", version="0.1.0")
app.mount("/static", StaticFiles(directory=_STATIC), name="static")
bearer = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    expected = os.environ.get("API_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=500, detail="API_TOKEN not configured.")
    if not credentials or credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid token.")
    return credentials.credentials


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None

class ChatResponse(BaseModel):
    reply: str
    conversation_id: int

class PermissionUpdate(BaseModel):
    enabled: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ask", response_class=PlainTextResponse)
def ask(
    q: str = Query(..., description="The question or command for the assistant."),
    token: Optional[str] = Query(None, description="API token (alternative to Bearer header)."),
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
):
    """
    Plain-text endpoint for iOS Shortcuts, curl, and other simple clients.

    Auth: pass ?token=<API_TOKEN> in the URL, or Authorization: Bearer header.
    Returns: plain text — no JSON wrapper, no markdown characters stripped.

    Example:
      curl "http://localhost:8000/ask?token=secret&q=what+is+on+my+calendar+today"
    """
    expected = os.environ.get("API_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=500, detail="API_TOKEN not configured.")

    provided = token or (credentials.credentials if credentials else None)
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid token.")

    user_message = q.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="q must not be empty.")

    past = memory.search(user_message)
    system_prompt = _system_prompt(past)

    reply = run(user_message, system=system_prompt)
    memory.save(user_message, reply)
    return reply


@app.get("/history/{conversation_id}")
def get_history(conversation_id: int, _token: str = Depends(_verify_token)):
    """Return the stored messages for a conversation."""
    msgs = conversations.load_history(conversation_id)
    return {"messages": msgs}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, _token: str = Depends(_verify_token)):
    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message must not be empty.")

    conv_id = body.conversation_id or conversations.create()

    # Load recent turns so the model has conversational context.
    history = conversations.load_history(conv_id, limit=20)

    past = memory.search(user_message)
    system_prompt = _system_prompt(past)

    conversations.save_message(conv_id, "user", user_message)
    reply = run(user_message, system=system_prompt, history=history)
    conversations.save_message(conv_id, "assistant", reply)
    memory.save(user_message, reply)

    return ChatResponse(reply=reply, conversation_id=conv_id)


@app.get("/permissions")
def get_permissions(_token: str = Depends(_verify_token)):
    """Return all connectors with their current enabled/disabled state."""
    return {"permissions": permissions.list_all()}


@app.patch("/permissions/{connector}")
def update_permission(
    connector: str,
    body: PermissionUpdate,
    _token: str = Depends(_verify_token),
):
    """Enable or disable a connector by name."""
    try:
        permissions.set_permission(connector, body.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    label = permissions.CONNECTOR_LABELS.get(connector, connector)
    state = "enabled" if body.enabled else "disabled"
    return {"connector": connector, "label": label, "enabled": body.enabled,
            "message": f"{label} {state}."}


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, _token: str = Depends(_verify_token)):
    """
    Stream reply tokens as Server-Sent Events.
    First event:  data: {"meta": {"conversation_id": <int>}}\n\n
    Each chunk:   data: {"chunk": "..."}\n\n
    Final event:  data: {"done": true}\n\n
    """
    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message must not be empty.")

    conv_id = body.conversation_id or conversations.create()

    # Load recent turns so the model has conversational context.
    history = conversations.load_history(conv_id, limit=20)
    conversations.save_message(conv_id, "user", user_message)

    past = memory.search(user_message)
    system_prompt = _system_prompt(past)

    # run_stream is a sync generator; bridge it to this async endpoint via a queue.
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _producer():
        try:
            for chunk in run_stream(user_message, system=system_prompt, history=history):
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    threading.Thread(target=_producer, daemon=True).start()

    async def _event_generator():
        # Send conversation_id first so the client can persist it before tokens arrive.
        yield f"data: {json.dumps({'meta': {'conversation_id': conv_id}})}\n\n"

        collected = []
        while True:
            chunk = await queue.get()
            if chunk is None:
                full_reply = "".join(collected)
                conversations.save_message(conv_id, "assistant", full_reply)
                memory.save(user_message, full_reply)
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            collected.append(chunk)
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
