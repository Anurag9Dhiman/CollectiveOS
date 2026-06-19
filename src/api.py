"""
FastAPI layer — exposes the assistant over HTTP.

Endpoints
---------
  POST /chat       Send a message, get a reply.
  GET  /health     Liveness check.

Auth
----
  Every request must include:
    Authorization: Bearer <API_TOKEN>
  where API_TOKEN is set in your .env file.

Run
---
  uvicorn src.api:app --reload --port 8000

  Then call it:
    curl -X POST http://localhost:8000/chat \
         -H "Authorization: Bearer your-token" \
         -H "Content-Type: application/json" \
         -d '{"message": "what is on my calendar today?"}'
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from src.assistant_starter import run
from src import memory

app = FastAPI(title="Personal Assistant API", version="0.1.0")
bearer = HTTPBearer()

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    expected = os.environ.get("API_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=500, detail="API_TOKEN not configured in environment.")
    if credentials.credentials != expected:
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

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, _token: str = Depends(_verify_token)):
    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message must not be empty.")

    # Retrieve relevant past memory and build system prompt.
    past = memory.search(user_message)
    system_prompt = "You are a helpful personal assistant."
    if past:
        system_prompt += "\n\nRelevant context from past conversations:\n" + past

    # Run the agent loop.
    reply = run(user_message, system=system_prompt)

    # Persist this exchange.
    memory.save(user_message, reply)

    return ChatResponse(reply=reply)
