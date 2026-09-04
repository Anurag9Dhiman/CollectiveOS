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
from src import conversations, memory, permissions, routines as _routines, watchers as _watchers
from src import titler as _titler
from src import scheduler as _scheduler
from src import observability as _obs
from src import redis_client as _cache
from src import multi_agent as _orchestrator
from src.agent import approve as agent_approve
from src.voice_gateway import handle_voice_ws
from src.wearable_stream import handle_wearable_ws
from src import computer_stream as _cs

_REPLY_CACHE_TTL = 3600  # keep last chat reply in Redis for 1 hour

_TZ_NAME = os.environ.get("TIMEZONE", "UTC")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _obs.configure_logging()
    _scheduler.load_all()
    _scheduler.start()
    _orchestrator.setup()   # register built-in agents + reload any DB-persisted ones
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
    image_b64: Optional[str] = None        # base64-encoded image for this turn only
    image_mime: str = "image/jpeg"         # MIME type: image/jpeg, image/png, image/webp, image/gif
    entity_refs: Optional[dict] = None     # agent context: scan_session_id, region, etc.

class ChatResponse(BaseModel):
    reply: str
    conversation_id: int
    interrupted: bool = False    # True when write-tool HITL approval is needed
    destructive: bool = False    # True when pending action is DESTRUCTIVE tier (sends msg / controls hardware)


class ApproveRequest(BaseModel):
    conversation_id: int
    approved: bool  # True = proceed, False = cancel

class PermissionUpdate(BaseModel):
    enabled: bool

_NOTIFY_VIA_OPTIONS = {"notification", "slack", "telegram", "push", "both", "none"}

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
    return {"status": "ok", "redis": "ok" if _cache.ping() else "unavailable"}


@app.get("/metrics")
def metrics(days: int = Query(1, ge=1, le=90), _token: str = Depends(_verify_token)):
    """Return token usage and tool latency metrics for the last N days."""
    return _obs.usage_data(days)


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

    past = memory.search_with_graph(user_message)
    system_prompt = _system_prompt(past)

    reply, _interrupted, _destructive = agent_run(user_message, system_prompt=system_prompt, thread_id="ask")
    memory.save_smart(user_message, reply)
    return reply


@app.get("/conversations")
def list_conversations(limit: int = Query(50, ge=1, le=200), _token: str = Depends(_verify_token)):
    """List past conversations, newest first, with a snippet and message count."""
    return {"conversations": conversations.list_conversations(limit)}


@app.get("/conversations/search")
def search_conversations(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    _token: str = Depends(_verify_token),
):
    """
    Full-text search across all message content (user + assistant turns).

    Returns up to *limit* ranked hits, each with conversation_id, started_at,
    role, and a highlighted snippet (<mark>…</mark> around matching terms).
    Falls back to ILIKE when the query cannot be parsed as tsquery.
    """
    hits = conversations.search_messages(q.strip(), limit=limit)
    return {"query": q, "hits": hits}


@app.get("/history/{conversation_id}")
def get_history(conversation_id: int, _token: str = Depends(_verify_token)):
    """Return the stored messages for a conversation."""
    msgs = conversations.load_history(conversation_id)
    return {"messages": msgs}


_ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_MAX_IMAGE_B64_BYTES = 5 * 1024 * 1024  # 5 MB base64 ≈ 3.75 MB raw

def _validate_image(body: ChatRequest) -> None:
    if not body.image_b64:
        return
    if body.image_mime not in _ALLOWED_IMAGE_MIMES:
        raise HTTPException(status_code=400, detail=f"image_mime must be one of {sorted(_ALLOWED_IMAGE_MIMES)}")
    if len(body.image_b64) > _MAX_IMAGE_B64_BYTES:
        raise HTTPException(status_code=413, detail="Image too large; max 5 MB base64.")


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, _token: str = Depends(_verify_token)):
    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message must not be empty.")
    _validate_image(body)

    conv_id = body.conversation_id or conversations.create()
    thread_id = str(conv_id)

    past = memory.search_with_graph(user_message)
    system_prompt = _system_prompt(past)

    conversations.save_message(conv_id, "user", user_message)

    # Route through the multi-agent orchestrator.
    # The orchestrator enriches with VisualOS context (if scan_session_id present),
    # then delegates to the task agent (LangGraph loop) with PostgresSaver checkpointing.
    result = _orchestrator.run(
        user_message,
        entity_refs=body.entity_refs or {},
        image_b64=body.image_b64 or None,
        image_mime=body.image_mime or "image/jpeg",
        system_prompt=system_prompt,
        thread_id=thread_id,
    )
    reply = result.text
    interrupted = result.metadata.get("interrupted", False)
    destructive = result.metadata.get("destructive", False)

    conversations.save_message(conv_id, "assistant", reply)
    if not interrupted:
        memory.save_smart(user_message, reply)
        _titler.title_async(conv_id)   # generate title in background (skips if already set)

    _cache.set(
        f"chat:status:{thread_id}",
        {"reply": reply, "interrupted": interrupted, "conversation_id": conv_id},
        ttl=_REPLY_CACHE_TTL,
    )
    return ChatResponse(
        reply=reply,
        conversation_id=conv_id,
        interrupted=interrupted,
        destructive=destructive,
    )


@app.get("/chat/status/{conversation_id}", response_model=ChatResponse)
def chat_status(conversation_id: int, _token: str = Depends(_verify_token)):
    """Return the last cached reply for a conversation.

    Useful for clients that want to poll after receiving interrupted=true,
    or for reconnecting voice/mobile clients that lost their response.
    Returns 404 if the conversation has no cached entry (expired or never started).
    """
    cached = _cache.get(f"chat:status:{conversation_id}")
    if cached is None:
        raise HTTPException(status_code=404, detail="No cached status for this conversation.")
    return ChatResponse(**cached)


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

    _cache.set(
        f"chat:status:{thread_id}",
        {"reply": reply, "interrupted": False, "conversation_id": conv_id},
        ttl=_REPLY_CACHE_TTL,
    )

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


# ---------------------------------------------------------------------------
# Proactive condition watchers
# ---------------------------------------------------------------------------

class WatcherCreate(BaseModel):
    name: str
    prompt: str
    condition: str
    interval_min: int = 60
    notify_via: str = "notification"

class WatcherUpdate(BaseModel):
    name: Optional[str]        = None
    prompt: Optional[str]      = None
    condition: Optional[str]   = None
    interval_min: Optional[int]= None
    enabled: Optional[bool]    = None
    notify_via: Optional[str]  = None


@app.get("/watchers")
def list_watchers(_token: str = Depends(_verify_token)):
    return {"watchers": _watchers.list_all()}


@app.post("/watchers", status_code=201)
def create_watcher(body: WatcherCreate, _token: str = Depends(_verify_token)):
    valid_channels = {"notification", "slack", "telegram", "push", "both", "none"}
    if body.notify_via not in valid_channels:
        raise HTTPException(status_code=400, detail=f"notify_via must be one of {sorted(valid_channels)}")
    if body.interval_min < 1:
        raise HTTPException(status_code=400, detail="interval_min must be ≥ 1")
    row = _watchers.create(
        name=body.name,
        prompt=body.prompt,
        condition=body.condition,
        interval_min=body.interval_min,
        notify_via=body.notify_via,
    )
    return row


@app.patch("/watchers/{watcher_id}")
def update_watcher(watcher_id: int, body: WatcherUpdate, _token: str = Depends(_verify_token)):
    row = _watchers.update(watcher_id, **body.model_dump(exclude_none=True))
    if row is None:
        raise HTTPException(status_code=404, detail="Watcher not found.")
    return row


@app.delete("/watchers/{watcher_id}", status_code=204)
def delete_watcher(watcher_id: int, _token: str = Depends(_verify_token)):
    if not _watchers.delete(watcher_id):
        raise HTTPException(status_code=404, detail="Watcher not found.")


@app.post("/watchers/{watcher_id}/check")
def check_watcher_now(watcher_id: int, _token: str = Depends(_verify_token)):
    """Evaluate a watcher immediately (runs in background thread)."""
    w = _watchers.get(watcher_id)
    if w is None:
        raise HTTPException(status_code=404, detail="Watcher not found.")
    import threading
    threading.Thread(target=_watchers.evaluate, args=(w,), daemon=True).start()
    return {"message": f"Watcher '{w['name']}' evaluation triggered."}


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


# ---------------------------------------------------------------------------
# Multi-agent registry — register, list, and remove peer agents
# ---------------------------------------------------------------------------

class AgentRegisterRequest(BaseModel):
    name: str                          # logical name, e.g. "visual", "calendar"
    url: str                           # base URL of the agent service
    protocol: str = "rest"             # "rest" | "a2a" | "ws"
    capabilities: List[str]            # what tasks this agent handles
    api_key: Optional[str] = None      # passed as X-API-Key on outbound calls
    health_url: Optional[str] = None   # GET this to check liveness


@app.get("/agents")
def list_agents(_token: str = Depends(_verify_token)):
    """Return all registered peer agents with their capabilities and liveness."""
    from src.agents.base import AgentRegistry
    return {"agents": AgentRegistry.list_all()}


@app.post("/agents/register", status_code=201)
def register_agent(body: AgentRegisterRequest, _token: str = Depends(_verify_token)):
    """Register (or update) a peer agent in the registry.

    Remote services (VisualOS, VoiceOS, future agents) call this on startup
    so CollectiveOS can delegate tasks to them.  Built-in agents are also
    registered here on CollectiveOS startup (via multi_agent.setup()).
    """
    if body.protocol not in ("rest", "a2a", "ws"):
        raise HTTPException(status_code=400, detail="protocol must be rest, a2a, or ws")
    if not body.capabilities:
        raise HTTPException(status_code=400, detail="capabilities must not be empty")

    client = _orchestrator._build_client(
        body.name, body.url, body.protocol, body.capabilities, body.api_key or ""
    )
    if client is None:
        raise HTTPException(
            status_code=400,
            detail=f"No client class available for protocol {body.protocol!r} with these capabilities."
        )

    from src.agents.base import AgentRegistry
    AgentRegistry.register(body.name, client)
    _orchestrator.persist_agent(
        body.name, body.url, body.protocol, body.capabilities,
        body.api_key or "", body.health_url or "",
    )
    return {"message": f"Agent '{body.name}' registered.", "capabilities": body.capabilities}


@app.delete("/agents/{name}", status_code=204)
def unregister_agent(name: str, _token: str = Depends(_verify_token)):
    """Unregister a peer agent by name."""
    from src.agents.base import AgentRegistry
    if not AgentRegistry.unregister(name):
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found.")
    _orchestrator.remove_agent(name)


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


# ---------------------------------------------------------------------------
# Wearable ingest — generic endpoint for any device
# ---------------------------------------------------------------------------

class WearableIngest(BaseModel):
    device_id: str
    event_type: str
    payload: dict = {}


@app.post("/wearable/ingest", status_code=201)
def wearable_ingest(body: WearableIngest, _token: str = Depends(_verify_token)):
    """
    Receive sensor/event data from any wearable device.

    Works with Garmin Connect IQ, Frame glasses (Brilliant Labs), Apple Watch
    via Shortcuts, or any custom hardware that can make an HTTP POST.

    Device setup (generic):
      URL: http://<host>:8000/wearable/ingest
      Method: POST
      Headers: Authorization: Bearer <API_TOKEN>, Content-Type: application/json
      Body: {"device_id": "my-device", "event_type": "gesture",
             "payload": { ...device-specific fields... }}

    event_type examples: gesture, sensor, location, button, voice, heartrate
    """
    import json as _json
    try:
        from src.db import connect, default_user_id
        conn = connect()
        uid = default_user_id(conn)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO wearable_events (user_id, device_id, event_type, payload) "
                    "VALUES (%s, %s, %s, %s::jsonb)",
                    (uid, body.device_id, body.event_type, _json.dumps(body.payload)),
                )
        conn.close()
        return {"message": f"Event saved from {body.device_id} ({body.event_type})."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Memory facts — direct CRUD for the facts injected into every system prompt
# ---------------------------------------------------------------------------

class FactCreate(BaseModel):
    content: str


@app.get("/memory/facts")
def list_memory_facts(_token: str = Depends(_verify_token)):
    """Return all saved facts, newest first."""
    return {"facts": memory.list_facts()}


@app.post("/memory/facts", status_code=201)
def create_memory_fact(body: FactCreate, _token: str = Depends(_verify_token)):
    """Save a new fact into persistent memory."""
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="content must not be empty.")
    memory.save_fact(content)
    return {"message": "Fact saved.", "facts": memory.list_facts()}


@app.delete("/memory/facts/{fact_id}", status_code=204)
def delete_memory_fact(fact_id: int, _token: str = Depends(_verify_token)):
    """Delete a specific fact by ID."""
    deleted = memory.delete_fact_by_id(fact_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Fact not found.")


@app.get("/export")
def export_data(
    sections: Optional[str] = Query(None, description="Comma-separated sections: conversations,facts,entities,routines,watchers"),
    _token: str = Depends(_verify_token),
):
    """
    Export all user data as a single JSON bundle for backup or migration.

    By default all sections are included.  Pass ?sections= to limit:
      conversations, facts, entities, routines, watchers

    The response carries Content-Disposition: attachment so browsers download
    it directly as collectiveos-YYYY-MM-DD.json.
    """
    from fastapi.responses import JSONResponse
    from src import exporter

    include = None
    if sections:
        include = {s.strip() for s in sections.split(",") if s.strip()}
        unknown = include - exporter.ALL_SECTIONS
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown sections: {sorted(unknown)}. Valid: {sorted(exporter.ALL_SECTIONS)}",
            )

    try:
        payload = exporter.build(include)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"collectiveos-{date_str}.json"
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/memory/graph")
def get_memory_graph(_token: str = Depends(_verify_token)):
    """
    Return the knowledge graph as nodes + links for visualisation.

    nodes: [{id, name, type, mention_count}]
    links: [{source, target, relation}]
    """
    from src.db import connect
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.id, e.name, e.entity_type,
                       COUNT(em.chunk_id) AS mention_count
                FROM entities e
                LEFT JOIN entity_mentions em ON em.entity_id = e.id
                GROUP BY e.id, e.name, e.entity_type
                ORDER BY mention_count DESC, e.name
            """)
            nodes = [
                {"id": r[0], "name": r[1], "type": r[2], "mention_count": r[3]}
                for r in cur.fetchall()
            ]
            cur.execute("""
                SELECT er.entity_a_id, er.entity_b_id, er.relation
                FROM entity_relations er
                JOIN entities ea ON ea.id = er.entity_a_id
                JOIN entities eb ON eb.id = er.entity_b_id
            """)
            links = [
                {"source": r[0], "target": r[1], "relation": r[2]}
                for r in cur.fetchall()
            ]
    finally:
        conn.close()
    return {"nodes": nodes, "links": links}


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, _token: str = Depends(_verify_token)):
    """
    Stream reply via Server-Sent Events backed by the LangGraph agent.

    Event sequence:
      {"meta": {"conversation_id": N}}          — first, always
      {"progress": "tool name"}                 — per tool call, 0 or more
      {"chunk": "text fragment"}                — word-by-word reply, 0 or more
      {"done": true, "interrupted": false}      — final, always
        or
      {"done": true, "interrupted": true}       — if HITL approval needed
    """
    import queue as _queue

    from src.agent import get_graph
    from src.assistant_starter import TOOLS
    from src import router as _router

    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message must not be empty.")
    _validate_image(body)

    conv_id = body.conversation_id or conversations.create()
    thread_id = str(conv_id)

    past = memory.search_with_graph(user_message)
    system_prompt = _system_prompt(past)
    conversations.save_message(conv_id, "user", user_message)

    active_tools, _ = _router.select_tools(user_message, TOOLS)

    # Build user parts — text plus optional inline image (not persisted to DB)
    user_parts: list[dict] = [{"text": user_message}]
    if body.image_b64:
        user_parts.append({
            "inline_data": {
                "mime_type": body.image_mime or "image/jpeg",
                "data": body.image_b64,
            }
        })

    initial_state = {
        "history": [{"role": "user", "parts": user_parts}],
        "system_prompt": system_prompt,
        "active_tools": active_tools,
        "reply": "",
        "pending_write": [],
        "approved": None,
        "iteration": 0,
    }
    config = {"configurable": {"thread_id": thread_id}}

    event_q: _queue.Queue = _queue.Queue()

    def _stream_agent():
        try:
            graph = get_graph()
            prev_len = 0
            for update in graph.stream(initial_state, config, stream_mode="updates"):
                agent_up = update.get("agent", {})
                history = agent_up.get("history", [])
                new_entries = history[prev_len:]
                prev_len = len(history)
                for entry in new_entries:
                    if entry.get("role") == "model":
                        for part in entry.get("parts", []):
                            if "function_call" in part:
                                event_q.put(("progress", part["function_call"].get("name", "")))

            state = graph.get_state(config)
            interrupted = bool(state.next)
            reply = state.values.get("reply", "")
            if interrupted and not reply:
                pending = state.values.get("pending_write") or []
                descs = ", ".join(c["name"] for c in pending)
                reply = f"I'd like to perform: {descs}\nShall I go ahead? (yes / no)"
            event_q.put(("done", reply, interrupted))
        except Exception as exc:
            event_q.put(("done", f"[Error: {exc}]", False))

    threading.Thread(target=_stream_agent, daemon=True).start()

    async def _generate():
        yield f"data: {json.dumps({'meta': {'conversation_id': conv_id}})}\n\n"

        reply = ""
        interrupted = False

        while True:
            try:
                item = event_q.get_nowait()
            except _queue.Empty:
                await asyncio.sleep(0.05)
                continue

            if item[0] == "progress":
                label = item[1].replace("_", " ")
                yield f"data: {json.dumps({'progress': label})}\n\n"
            else:
                _, reply, interrupted = item
            chunk = await queue.get()
            if chunk is None:
                full_reply = "".join(collected)
                conversations.save_message(conv_id, "assistant", full_reply)
                memory.save_smart(user_message, full_reply)
                yield f"data: {json.dumps({'done': True})}\n\n"
                break

        # Stream reply word-by-word so the client renders it progressively.
        words = reply.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            await asyncio.sleep(0.02)

        conversations.save_message(conv_id, "assistant", reply)
        if not interrupted:
            memory.save(user_message, reply)
            _titler.title_async(conv_id)

        yield f"data: {json.dumps({'done': True, 'interrupted': interrupted})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Computer-use live stream — SSE progress feed + stop control
# ---------------------------------------------------------------------------

@app.get("/computer/stream")
async def computer_stream_sse(_token: str = Depends(_verify_token)):
    """
    Server-Sent Events stream of computer-use progress.

    Open this stream before (or immediately after) triggering a computer_use
    tool call.  The stream emits one JSON event per action taken by the agent
    and closes automatically when the run finishes or is stopped.

    Event types:
      data: {"event": "start",  "task": "...", "run_id": "..."}
      data: {"event": "action", "run_id": "...", "iteration": N,
                                "action": {...}, "screenshot_b64": "...",
                                "verify": "PROCEED|STUCK|ERROR|DONE|null"}
      data: {"event": "done",   "run_id": "...", "result": "...", "iterations": N}
      data: {"event": "stop"}

    Keep-alive comments (": keep-alive") are sent every 5 s between events.
    """
    async def _generate():
        for event in _cs.event_stream(timeout=5.0):
            if event is None:
                yield ": keep-alive\n\n"
            else:
                yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/computer/stop")
def computer_stop(_token: str = Depends(_verify_token)):
    """Request the currently running computer-use agent to stop after its next action."""
    stopped = _cs.request_stop()
    return {"message": "Stop signal sent." if stopped else "No active computer-use run."}


# ---------------------------------------------------------------------------
# Robot status — live state for the floor-plan UI panel
# ---------------------------------------------------------------------------

@app.get("/robot/status")
def robot_status_endpoint(_token: str = Depends(_verify_token)):
    """Return the current robot state (room, battery, last action).

    When ROS2_MCP_URL is unset this reads from the local simulator state file.
    """
    from src.connectors import ros2_sim as _sim
    return _sim.get_state()


# ---------------------------------------------------------------------------
# Morning briefing
# ---------------------------------------------------------------------------

@app.get("/briefing")
def briefing_now(_token: str = Depends(_verify_token)):
    """Generate and return a morning briefing immediately."""
    from src import briefing as _briefing
    return _briefing.generate()


@app.get("/briefing/schedule")
def briefing_schedule_get(_token: str = Depends(_verify_token)):
    """Return the current briefing schedule configuration."""
    from src import briefing as _briefing
    return _briefing.get_config()


@app.post("/briefing/schedule")
def briefing_schedule_set(body: dict, _token: str = Depends(_verify_token)):
    """
    Update the briefing schedule.

    Accepted fields: enabled (bool), hour (0-23), minute (0-59),
    timezone (IANA string, e.g. "America/New_York"), notify_via (str).
    """
    from src import briefing as _briefing, scheduler as _sched
    cfg = _briefing.set_config(body)
    # Re-register the scheduler job with the updated config
    try:
        _briefing.register_job(_sched._scheduler)
    except Exception as exc:
        log.warning("Could not re-register briefing job: %s", exc)
    return cfg


# ---------------------------------------------------------------------------
# Wearable always-on stream — WebSocket for glasses / watch / phone
# ---------------------------------------------------------------------------

@app.websocket("/wearable/ws")
async def wearable_websocket(ws: WebSocket, token: str = "") -> None:
    """
    Always-on wearable stream — implements the VisionClaw architecture.

    Connect from any wearable (Meta Ray-Ban glasses, phone, watch) and send
    a continuous stream of transcripts and optional camera frames.  A cheap
    intent classifier decides whether to invoke the full agent.

    Auth: pass ?token=<API_TOKEN> as a query parameter.

    Wire protocol (JSON over WebSocket):

      Client → Server
        {"type": "transcript", "text": "...", "device_id": "glasses-1"}
        {"type": "frame",      "image_b64": "...", "device_id": "...",
                               "image_mime": "image/jpeg"}
        {"type": "context",    "text": "...", "image_b64": "...",
                               "image_mime": "image/jpeg", "device_id": "..."}
        {"type": "ping"}

      Server → Client
        {"type": "ack",   "message": "...", "triggered": false}
        {"type": "reply", "text": "...",    "triggered": true}
        {"type": "pong"}
        {"type": "error", "message": "..."}

    The server replies with triggered=false to every non-intent transcript
    (so the client knows the message was received) and triggered=true only
    when the intent classifier fires and the agent has responded.
    """
    expected = os.environ.get("API_TOKEN", "")
    if not expected or token != expected:
        await ws.close(code=4001)
        return
    await handle_wearable_ws(ws, token)


# ---------------------------------------------------------------------------
# Slack two-way interface — Events API webhook + output helpers
# ---------------------------------------------------------------------------

def _slack_send(channel_id: str, text: str) -> None:
    """Post a message to a Slack channel via the Web API."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        return
    import requests as _req
    try:
        _req.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {token}"},
            json={"channel": channel_id, "text": text[:4000]},
            timeout=15,
        )
    except Exception:
        pass


def _handle_slack_message(channel_id: str, text: str) -> None:
    """Run the agent on a Slack message and post the reply — background thread."""
    past = memory.search_with_graph(text)
    try:
        reply = run(text, system=_system_prompt(past))
    except Exception as exc:
        reply = f"Sorry, something went wrong: {exc}"
    memory.save_smart(text, reply)
    _slack_send(channel_id, reply)


def _verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    """Verify Slack's HMAC-SHA256 request signature. Returns True if valid."""
    import hashlib
    import hmac
    import time as _time
    secret = os.environ.get("SLACK_SIGNING_SECRET", "")
    if not secret:
        return True  # Skip verification if secret not configured
    try:
        if abs(_time.time() - float(timestamp)) > 300:
            return False
        sig_base = f"v0:{timestamp}:{body.decode()}"
        expected = "v0=" + hmac.new(
            secret.encode(), sig_base.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)
    except Exception:
        return False


@app.post("/slack/events", include_in_schema=False)
async def slack_events(request: Request):
    """
    Receive events from Slack (Events API) and reply via the agent.

    Setup (one time):
      1. Add to .env:
           SLACK_BOT_TOKEN=xoxb-...
           SLACK_CHANNEL_ID=C...        (channel to post proactive messages to)
           SLACK_SIGNING_SECRET=...     (App → Basic Information → Signing Secret)
      2. In your Slack App → Event Subscriptions:
           - Enable Events
           - Request URL: https://<your-domain>/slack/events
           - Subscribe to bot events: message.im  (for DMs)
             and/or message.channels / message.groups (for channels)
      3. Invite the bot: /invite @your-bot  in the desired channel or DM it.
    """
    body_bytes = await request.body()

    # Verify Slack's request signature
    ts  = request.headers.get("X-Slack-Request-Timestamp", "")
    sig = request.headers.get("X-Slack-Signature", "")
    if not _verify_slack_signature(body_bytes, ts, sig):
        raise HTTPException(status_code=403, detail="Invalid Slack signature")

    body = json.loads(body_bytes)

    # URL verification challenge (one-time, during setup)
    if body.get("type") == "url_verification":
        return {"challenge": body["challenge"]}

    # Acknowledge immediately — Slack retries if it doesn't get 200 within 3 s.
    # Ignore retried deliveries so the agent doesn't run twice.
    if request.headers.get("X-Slack-Retry-Num"):
        return {"ok": True}

    event = body.get("event", {})

    # Only handle human text messages (not bot echoes or subtypes)
    if event.get("type") != "message" or event.get("bot_id") or event.get("subtype"):
        return {"ok": True}

    text = (event.get("text") or "").strip()
    channel_id = event.get("channel", "")
    if not text or not channel_id:
        return {"ok": True}

    threading.Thread(
        target=_handle_slack_message,
        args=(channel_id, text),
        daemon=True,
    ).start()
    return {"ok": True}


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
    past = memory.search_with_graph(text)
    try:
        reply = run(text, system=_system_prompt(past))
    except Exception as exc:
        reply = f"Sorry, something went wrong: {exc}"
    memory.save_smart(text, reply)
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
