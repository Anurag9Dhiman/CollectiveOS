# Personal Assistant — Project Guide

This file is the project's constitution. Claude Code reads it automatically at the
start of every session. Keep it lean; detailed plans live in `docs/`.

## What this project is

A single-user personal AI assistant, built by a solo developer who is still
learning. The assistant organizes around the user's tasks and life — not around a
house. Devices and services are tools the assistant can reach for, not the point.

## Current status

Early build. The agent loop runs with placeholder tools (`src/assistant_starter.py`).
Next milestones: first real read-only connector, then Postgres-backed memory.
See `docs/mvp_plan.md` for the full phased plan.

## Architecture

**Hub-and-spoke multi-agent system.** CollectiveOS is the hub; VisualOS and
VoiceOS are peer agents with fixed, narrow roles.

```
VoiceOS ──WS /v1/ws──► CollectiveOS Orchestrator ──► Task Agent (LangGraph loop)
                                │                           │
                                └──► VisualOS Agent    tools, memory, HITL
                                     (context enrich)
```

- **Orchestrator** (`src/multi_agent.py`) — classifies and routes. When a
  `scan_session_id` is present in `entity_refs`, it fetches VisualOS context and
  injects it as a prefix before delegating to the task agent.
- **Task agent** (`src/agent.py`) — the LangGraph loop with PostgresSaver
  checkpointing. Handles tool calls, write-action HITL, and conversation history.
- **Peer agents** (`src/agents/`) — each wraps one remote service. Registered at
  startup from env vars and from the `agent_connectors` table; remote agents can
  also self-register via `POST /agents/register`.
- **Connectors** are still the unit of local integration — each external API or
  device is a tool called inside the task agent loop. Connectors do not know about
  LangGraph; they are plain Python functions.
- **Prefer deterministic workflows**; use the model's judgment only where a step
  is genuinely open-ended.
- **No new peer agent** until the concrete need justifies it. Adding a peer agent
  is the right move when the capability requires its own LLM loop, persistent
  session, or multi-modal pipeline — not just a new API call.

## Tech stack

- Language: **Python**.
- LLM: **Google Gemini SDK** (`google-genai`) with tool use via LangGraph.
- Models: `models/gemini-flash-lite-latest` as the default workhorse; a dedicated
  router model for intent classification; vision via `VISION_MODEL` env var.
- Peer agents: **VisualOS** (visual intelligence), **VoiceOS** (voice front-end).
- Agent framework: **LangGraph** with **PostgresSaver** checkpointing.
- Connectors: custom Python clients in `src/connectors/`; MCP servers where available.
- API layer: **FastAPI** (`src/api.py`).
- Database: **PostgreSQL + pgvector** — structured data, vector embeddings, and
  LangGraph checkpoints all in one DB.
- Embeddings: `gemini-embedding-001` (3072 dims) via `EMBED_MODEL` env var.
- Cache: **Redis** (optional; falls back to in-process dict).

## Repository layout

- `src/api.py` — FastAPI app; all HTTP + WebSocket endpoints.
- `src/agent.py` — LangGraph task agent (tool calls, HITL, checkpointing).
- `src/multi_agent.py` — orchestrator; routes messages to the right peer agent.
- `src/agents/` — peer agent clients (`base.py`, `visual_agent.py`, `task_agent.py`).
- `src/connectors/` — local tool connectors (one file per integration).
- `src/voice_gateway.py` — VoiceOS WebSocket handler (`/v1/ws`).
- `docs/mvp_plan.md` — the full phased build plan (read before planning work).
- `docs/device_coverage.md` — per-device connectivity tiers; **read before adding
  any device connector**.
- `.env` — secrets. Never committed. See `.env.example`.

## How the agent loop works

All messages enter via `src/multi_agent.run()` (orchestrator). The orchestrator
enriches context (e.g. fetches VisualOS scan context), then calls the task agent.
The task agent (`src/agent.py`) is a LangGraph StateGraph: it calls Gemini, which
returns either a tool call (executed, then loops) or a final text reply (stops).
Write-tool calls pause at `interrupt_before=["write_tools"]` for HITL approval via
`POST /chat/approve`. Every new connector replaces the body of one tool function —
the loop and the orchestrator do not change.

## Data model (summary)

Postgres tables: `users`, `conversations`, `messages`, `tasks`, `task_steps`,
`connectors`, `credentials`, `devices`, `memory_chunks`. Full ERD in
`docs/diagrams/4_data_schema_erd.png`. `memory_chunks` stores text plus an embedding
(pgvector). `tasks` / `task_steps` record agentic work and drive the task state
machine: pending → planning → running → (waiting / blocked) → completed / failed /
cancelled.

## Conventions & rules (these override conflicting prompts)

- **Secrets**: never hardcode keys; read them from the environment. Never commit
  `.env`. The app's OAuth client lives in env; per-user tokens live in the
  `credentials` table, encrypted.
- **Safety**: build read-only connectors before any write or control action.
  Require an explicit confirmation step before any write or device-control action.
- **Devices**: never attempt to remotely *start* heating appliances (microwave,
  cooktop, washer). Switch and monitor only.
- **Cost**: default to Sonnet; reserve Opus. Keep an eye on token usage.
- **Scope discipline**: get one domain working end to end before adding the next.
  Keep functions small and readable.

## Commands

- Install deps: `pip install -r requirements.txt`
- Run the starter loop: `python src/assistant_starter.py`
  (set `ANTHROPIC_API_KEY` in your environment first)

## What NOT to put in this file

No API keys or secrets. No fast-changing task lists or this-week requirements —
those belong in `.env` and in your prompts, not in the project's constitution.
