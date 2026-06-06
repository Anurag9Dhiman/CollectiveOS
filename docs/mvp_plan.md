# MVP Build Plan

A phased plan for building the personal AI assistant from scratch. Complete one phase end-to-end before starting the next.

---

## Phase 0 — Skeleton (current)

**Goal:** prove the agent loop works with fake tools.

- [x] `src/assistant_starter.py` — agent loop with placeholder `get_current_time` and `set_light` tools
- [x] Project docs: `CLAUDE.md`, `docs/mvp_plan.md`, `docs/device_coverage.md`
- [x] Architecture diagrams in `docs/diagrams/`
- [x] `.env.example` and `requirements.txt`

Deliverable: `python src/assistant_starter.py` runs and responds to questions.

---

## Phase 1 — First real connector (read-only)

**Goal:** replace one fake tool with a real, read-only API call.

Candidates (pick one):
- Google Calendar — read upcoming events
- Weather API — current conditions for a location
- Local file/notes — read a markdown file

Steps:
1. Add the real API client to `requirements.txt`.
2. Replace the tool function body with the real call.
3. Store credentials in `.env`; read with `os.environ`.
4. Smoke-test end-to-end: ask the assistant a question that exercises the tool.

Deliverable: assistant answers a real question using live data.

---

## Phase 2 — Postgres-backed memory

**Goal:** the assistant remembers things across sessions.

Steps:
1. Stand up Postgres locally (Docker is fine) with pgvector extension.
2. Run the schema from `docs/diagrams/4_data_schema_erd.png`.
3. Add `save_memory` and `search_memory` tools to the agent loop.
4. On each conversation end, embed the exchange and store in `memory_chunks`.
5. On each new conversation, retrieve top-k relevant chunks and inject into the system prompt.

Deliverable: assistant recalls facts from a previous session.

---

## Phase 3 — Services phase

**Goal:** cover the most useful cloud services.

Connectors to build (in order):
1. Google Calendar — read + write events (confirm before write)
2. Gmail — read recent emails, draft replies (confirm before send)
3. Cloud storage (Google Drive or similar) — read files
4. Task / to-do list (Todoist or similar)

Each connector follows the same pattern: one Python file in `src/connectors/`, one or more tool definitions added to `TOOLS`, function registered in `TOOL_FUNCTIONS`.

---

## Phase 4 — Devices phase

**Goal:** control smart home devices.

Read `docs/device_coverage.md` before touching anything here.

Connectors to build (suggested order per device_coverage.md):
1. Home Assistant — covers most smart home devices via one integration
2. Router — network status and device presence
3. One vendor-cloud appliance (prove the pattern)
4. Bridged "dumb" devices via smart plugs

Safety rule: no remote-start of heating appliances (microwave, cooktop, washer).

---

## Phase 5 — FastAPI layer

**Goal:** expose the agent over HTTP so web/mobile clients can reach it.

Steps:
1. Add FastAPI + uvicorn to `requirements.txt`.
2. `POST /chat` endpoint — accepts a message, returns the agent's reply.
3. Auth: single-user token in `Authorization` header (from `.env`).
4. Conversation state: look up or create a `conversations` row per session.

---

## Phase 6 — Interface & voice

**Goal:** talk to the assistant from phone / browser.

Options:
- Simple web chat UI (plain HTML + JS, served by FastAPI)
- iOS Shortcuts / Android Tasker webhook → POST /chat
- Voice: whisper for STT, TTS API for speech output

---

## Routing (add when tool count > ~15)

Add a cheap classification step before the main agent call:
1. Send the user message + a tool-category list to `claude-haiku-4-5`.
2. Haiku returns a category (or "general").
3. Pass only the relevant tool subset to the main Sonnet call.

This keeps per-request token cost flat as connectors multiply.
