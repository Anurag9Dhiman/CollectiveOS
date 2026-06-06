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

## Architecture (do not drift from this without discussion)

- **Pattern: a single "augmented LLM" agent** — one model in a tool-use loop, with
  retrieval and memory. NOT a multi-agent system.
- **Prefer deterministic workflows**; use the model's judgment only where a step is
  genuinely open-ended.
- Add a cheap **routing** step (classify, then narrow the tool list) only once the
  tool count grows. No orchestrator / multi-agent until a concrete need proves it.
- **Connectors are the unit of integration**: each external service or device is a
  tool — via an MCP server where one exists, otherwise a custom Python client.

## Tech stack

- Language: **Python**.
- LLM: **Anthropic SDK** with tool use. No agent framework yet.
- Models: `claude-sonnet-4-6` as the default workhorse; `claude-haiku-4-5` for
  routing/classification; `claude-opus-4-8` only for genuinely hard reasoning.
- Connectors: **MCP servers** + custom API clients.
- API layer (later): **FastAPI**.
- Database: **PostgreSQL + pgvector** — one database for both structured data and
  vector search. SQLite is acceptable for the earliest local experiments.
- Embeddings: a **dedicated embeddings model** (Anthropic's models generate text,
  not embeddings) — a hosted embeddings API or a local open-source model.
- Cache / queue (later): Redis.

## Repository layout

- `src/` — application code. The agent loop lives here.
- `docs/mvp_plan.md` — the full phased build plan (read before planning work).
- `docs/device_coverage.md` — per-device connectivity tiers; **read before adding
  any device connector**.
- `docs/diagrams/` — architecture diagrams (images, for humans).
- `.env` — secrets. Never committed. See `.env.example`.

## How the agent loop works

See `src/assistant_starter.py`. The model returns either a tool call (we run it,
feed the result back, and loop) or a final answer (we stop). Every new connector
just replaces the *body* of a tool function — the loop itself never changes.

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
