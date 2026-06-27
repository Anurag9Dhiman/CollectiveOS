"""
FastAPI layer — exposes the assistant over HTTP.

Endpoints
---------
  GET  /              Web chat UI (static/index.html)
  POST /chat          Send a message, get a full reply (JSON).
  POST /chat/stream   Send a message, stream reply tokens (SSE).
  GET  /health        Liveness check.

Auth
----
  Every /chat* request must include:
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

from src.assistant_starter import run, run_stream
from src import memory

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

class ChatResponse(BaseModel):
    reply: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, _token: str = Depends(_verify_token)):
    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message must not be empty.")

    past = memory.search(user_message)
    system_prompt = "You are a helpful personal assistant."
    if past:
        system_prompt += "\n\nRelevant context from past conversations:\n" + past

    reply = run(user_message, system=system_prompt)
    memory.save(user_message, reply)
    return ChatResponse(reply=reply)


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, _token: str = Depends(_verify_token)):
    """
    Stream reply tokens as Server-Sent Events.
    Each event: data: {"chunk": "..."}\n\n
    Final event: data: {"done": true}\n\n
    """
    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message must not be empty.")

    past = memory.search(user_message)
    system_prompt = "You are a helpful personal assistant."
    if past:
        system_prompt += "\n\nRelevant context from past conversations:\n" + past

    # run_stream is a sync generator; run it in a background thread and
    # forward chunks to an asyncio queue so this async endpoint can yield them.
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _producer():
        try:
            for chunk in run_stream(user_message, system=system_prompt):
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    threading.Thread(target=_producer, daemon=True).start()

    async def _event_generator():
        collected = []
        while True:
            chunk = await queue.get()
            if chunk is None:
                memory.save(user_message, "".join(collected))
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            collected.append(chunk)
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
