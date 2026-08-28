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
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Depends, Query, Request, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

from src.assistant_starter import run_stream
from src import conversations, memory, permissions, routines as _routines
from src import scheduler as _scheduler
from src import observability as _obs
from src.agent import run as agent_run, approve as agent_approve
from src.voice_gateway import handle_voice_ws

_TZ_NAME = os.environ.get("TIMEZONE", "UTC")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _obs.configure_logging()
    _scheduler.load_all()
    _scheduler.start()
    yield
    _scheduler.shutdown()


_READ_TOOLS = (
    "memory_list, "
    "get_calendar_events, get_recent_emails, search_emails, list_drive_files, "
    "read_drive_file, get_tasks, get_projects, get_devices, get_device_state, "
    "spotify_now_playing, spotify_get_devices, get_system_info, get_wifi_info, "
    "web_search, imessage_get_messages, capture_screen, list_directory, read_local_file, "
    "browser_get_active_tab, browser_list_tabs, contacts_search, reminders_list, "
    "notes_list, notes_read, clipboard_read, telegram_get_messages"
    "notes_list, notes_read, clipboard_read, notion_search, notion_read_page, "
    "github_list_repos, github_list_prs, github_list_issues, github_get_ci_status, "
    "slack_list_channels, slack_read_messages, "
    "health_get_sleep, health_get_activity, health_get_readiness, "
    "car_get_status, appliances_list, appliances_get_status"
    "finance_get_accounts, finance_get_transactions, finance_get_spending_summary"
)
_WRITE_TOOLS = (
    "memory_remember, memory_forget, "
    "create_event, create_draft, send_email, add_task, complete_task, update_task, "
    "control_device, set_light, spotify_control, spotify_set_volume, spotify_search_play, "
    "show_notification, open_application, set_system_volume, imessage_send, "
    "write_local_file, browser_open_url, reminders_add, reminders_complete, "
    "notes_create, notes_append, clipboard_write, telegram_send"
    "notes_create, notes_append, clipboard_write, notion_create_page, notion_append_to_page, "
    "github_create_issue, slack_send_message, "
    "car_lock, car_climate, appliances_control"
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
    facts = memory.get_all_facts_str()
    if facts:
        prompt += f"\n\nThings you know about the user (always keep in mind):\n{facts}"
    if past:
        prompt += "\n\nRelevant context from past conversations:\n" + past
    return prompt

_HERE   = os.path.dirname(os.path.abspath(__file__))
_STATIC = os.path.join(_HERE, "..", "static")

app = FastAPI(title="Personal Assistant API", version="0.1.0", lifespan=_lifespan)
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
    interrupted: bool = False  # True when write-tool HITL approval is needed


class ApproveRequest(BaseModel):
    conversation_id: int
    approved: bool  # True = proceed, False = cancel

class PermissionUpdate(BaseModel):
    enabled: bool

_NOTIFY_VIA_OPTIONS = {"notification", "none", "telegram"}

class RoutineCreate(BaseModel):
    name: str
    prompt: str
    schedule: str          # cron expression, e.g. "0 8 * * *"
    notify_via: str = "notification"

class RoutineUpdate(BaseModel):
    name: Optional[str]    = None
    prompt: Optional[str]  = None
    schedule: Optional[str]= None
    enabled: Optional[bool]= None
    notify_via: Optional[str] = None

class HealthIngest(BaseModel):
    date: str                    # YYYY-MM-DD
    source: str = "apple_health"
    metrics: dict                # steps, sleep_hours, hrv, resting_heart_rate, etc.

class ConnectorConfig(BaseModel):
    brand: Optional[str] = None


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

    reply, _interrupted = agent_run(user_message, system_prompt=system_prompt, thread_id="ask")
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
    thread_id = str(conv_id)

    past = memory.search(user_message)
    system_prompt = _system_prompt(past)

    conversations.save_message(conv_id, "user", user_message)

    # The LangGraph agent carries full history via PostgresSaver — no need to
    # load and pass history manually; it's restored from the checkpoint.
    reply, interrupted = agent_run(
        user_message,
        system_prompt=system_prompt,
        thread_id=thread_id,
    )

    conversations.save_message(conv_id, "assistant", reply)
    if not interrupted:
        memory.save(user_message, reply)

    return ChatResponse(reply=reply, conversation_id=conv_id, interrupted=interrupted)


@app.post("/chat/approve", response_model=ChatResponse)
def chat_approve(body: ApproveRequest, _token: str = Depends(_verify_token)):
    """Resume a paused graph after a write-tool HITL decision.

    Send approved=true to execute the pending action, false to cancel it.
    The conversation_id must match the one returned by the interrupted /chat call.
    """
    conv_id = body.conversation_id
    thread_id = str(conv_id)
    reply = agent_approve(thread_id=thread_id, approved=body.approved)
    conversations.save_message(conv_id, "assistant", reply)
    if body.approved:
        memory.save("", reply)
    return ChatResponse(reply=reply, conversation_id=conv_id, interrupted=False)


@app.get("/routines")
def get_routines(_token: str = Depends(_verify_token)):
    """List all scheduled routines."""
    return {"routines": _routines.list_all()}


@app.post("/routines", status_code=201)
def create_routine(body: RoutineCreate, _token: str = Depends(_verify_token)):
    """Create a new scheduled routine."""
    from apscheduler.triggers.cron import CronTrigger
    try:
        CronTrigger.from_crontab(body.schedule)
    except Exception:
        raise HTTPException(status_code=400,
                            detail=f"Invalid cron expression: {body.schedule!r}")
    if body.notify_via not in _NOTIFY_VIA_OPTIONS:
        raise HTTPException(status_code=400,
                            detail=f"notify_via must be one of: {sorted(_NOTIFY_VIA_OPTIONS)}")
    row = _routines.create(body.name, body.prompt, body.schedule, body.notify_via)
    _scheduler.reload_routine(row["id"])
    return row


@app.patch("/routines/{routine_id}")
def update_routine(routine_id: int, body: RoutineUpdate,
                   _token: str = Depends(_verify_token)):
    """Update fields on an existing routine."""
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if "schedule" in updates:
        from apscheduler.triggers.cron import CronTrigger
        try:
            CronTrigger.from_crontab(updates["schedule"])
        except Exception:
            raise HTTPException(status_code=400,
                                detail=f"Invalid cron: {updates['schedule']!r}")
    if "notify_via" in updates and updates["notify_via"] not in _NOTIFY_VIA_OPTIONS:
        raise HTTPException(status_code=400,
                            detail=f"notify_via must be one of: {sorted(_NOTIFY_VIA_OPTIONS)}")
    row = _routines.update(routine_id, **updates)
    if row is None:
        raise HTTPException(status_code=404, detail="Routine not found.")
    _scheduler.reload_routine(routine_id)
    return row


@app.delete("/routines/{routine_id}", status_code=204)
def delete_routine(routine_id: int, _token: str = Depends(_verify_token)):
    """Delete a routine and remove it from the live scheduler."""
    if not _routines.delete(routine_id):
        raise HTTPException(status_code=404, detail="Routine not found.")
    _scheduler.remove_job(routine_id)


@app.post("/routines/{routine_id}/run")
def run_routine_now(routine_id: int, _token: str = Depends(_verify_token)):
    """Trigger a routine immediately (runs in background thread)."""
    r = _routines.get(routine_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Routine not found.")
    import threading
    threading.Thread(
        target=_scheduler._run_routine,
        kwargs={"routine_id": r["id"], "name": r["name"],
                "prompt": r["prompt"], "notify_via": r["notify_via"]},
        daemon=True,
    ).start()
    return {"message": f"Routine '{r['name']}' triggered."}


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


@app.patch("/permissions/{connector}/config")
def update_connector_config(
    connector: str,
    body: ConnectorConfig,
    _token: str = Depends(_verify_token),
):
    """Update the brand/config for a connector."""
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        permissions.set_config(connector, updates)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"connector": connector, "config": updates}


@app.post("/health-ingest", status_code=201)
def health_ingest(body: HealthIngest, _token: str = Depends(_verify_token)):
    """
    Receive health metrics from an iOS Shortcut or any external source.

    iOS Shortcut setup:
      1. "Get Health Samples" actions for steps, sleep, HRV, heart rate, etc.
      2. "Get Contents of URL" — POST to http://<mac-ip>:8000/health-ingest
         Headers: Authorization: Bearer <API_TOKEN>, Content-Type: application/json
         Body: {"date": "<today>", "source": "apple_health",
                "metrics": {"steps": ..., "sleep_hours": ..., "hrv": ...,
                            "resting_heart_rate": ..., "active_calories": ...}}
    """
    import json as _json
    try:
        from src.db import connect
        conn = connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO health_snapshots (date, source, metrics)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (date, source) DO UPDATE
                        SET metrics    = health_snapshots.metrics || EXCLUDED.metrics,
                            created_at = NOW()
                    """,
                    (body.date, body.source, _json.dumps(body.metrics)),
                )
        conn.close()
        return {"message": f"Health data saved for {body.date} ({body.source})."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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


# ---------------------------------------------------------------------------
# Telegram webhook — two-way Telegram interface
# ---------------------------------------------------------------------------

def _tg_send(chat_id: str, text: str) -> None:
    """Send a message to a Telegram chat via the Bot API."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return
    import requests as _req
    try:
        _req.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4096], "parse_mode": "Markdown"},
            timeout=15,
        )
    except Exception:
        pass


def _handle_tg_message(chat_id: str, text: str) -> None:
    """Run the agent and reply — executed in a background thread."""
    past = memory.search(text)
    try:
        reply = run(text, system=_system_prompt(past))
    except Exception as exc:
        reply = f"Sorry, something went wrong: {exc}"
    memory.save(text, reply)
    _tg_send(chat_id, reply)


@app.websocket("/v1/ws")
async def voice_websocket(ws: WebSocket) -> None:
    """Voice gateway — VoiceOS connects here and speaks the voice_contract protocol."""
    await handle_voice_ws(ws)


@app.post("/telegram/webhook/{secret}", include_in_schema=False)
async def telegram_webhook(secret: str, request: Request):
    """
    Receive messages from Telegram and reply via the agent.

    Setup (one time):
      1. Add to .env:
           TELEGRAM_BOT_TOKEN=<token from @BotFather>
           TELEGRAM_CHAT_ID=<your personal chat ID>
           TELEGRAM_WEBHOOK_SECRET=<any random string you choose>
      2. Register the webhook (replace placeholders):
           curl "https://api.telegram.org/bot<TOKEN>/setWebhook\\
                ?url=https://<your-domain>/telegram/webhook/<SECRET>"
      3. Send your bot a message — it will reply using the full agent.
    """
    expected = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
    if not expected or secret != expected:
        raise HTTPException(status_code=403)

    body = await request.json()
    message = body.get("message", {})
    chat_id = str(message.get("chat", {}).get("id", ""))
    text = (message.get("text") or "").strip()

    if not text or not chat_id:
        return {"ok": True}

    # Only respond to the configured chat (personal-use guard)
    allowed = os.environ.get("TELEGRAM_CHAT_ID", "")
    if allowed and chat_id != allowed:
        return {"ok": True}

    if text == "/start":
        _tg_send(chat_id, "Hi! I'm your personal assistant. Send me a message.")
        return {"ok": True}

    # Run agent in background so Telegram doesn't time out waiting
    threading.Thread(
        target=_handle_tg_message,
        args=(chat_id, text),
        daemon=True,
    ).start()
    return {"ok": True}
