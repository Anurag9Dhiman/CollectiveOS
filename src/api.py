"""
FastAPI layer — exposes the assistant over HTTP.

Endpoints
---------
  GET  /                         Web chat UI (static/index.html)
  POST /chat                     Send a message, get a full reply (JSON).
  POST /chat/stream              Send a message, stream reply tokens (SSE).
  GET  /history/{conversation_id} Return stored messages for a conversation.
  GET  /health                   Liveness check.

Auth
----
  Every /chat* and /history* request must include:
    Authorization: Bearer <API_TOKEN>
  where API_TOKEN is set in your .env file.

Run
---
  uvicorn src.api:app --reload --port 8000
  Then open http://localhost:8000 in your browser.
"""

import asyncio
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from src.assistant_starter import run, run_stream
from src import conversations, memory

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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


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
    system_prompt = "You are a helpful personal assistant."
    if past:
        system_prompt += "\n\nRelevant context from past conversations:\n" + past

    conversations.save_message(conv_id, "user", user_message)
    reply = run(user_message, system=system_prompt, history=history)
    conversations.save_message(conv_id, "assistant", reply)
    memory.save(user_message, reply)

    return ChatResponse(reply=reply, conversation_id=conv_id)


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
    system_prompt = "You are a helpful personal assistant."
    if past:
        system_prompt += "\n\nRelevant context from past conversations:\n" + past

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
