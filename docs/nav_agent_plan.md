# Navigation Agent — Implementation Plan

## The Idea

Instead of 85 individual API connectors (one per service), a single **Computer Navigation
Agent** navigates the macOS GUI the way a human assistant would — by looking at the screen
and controlling the cursor and keyboard. No API keys per app. No OAuth per service. Works with
anything that has a UI.

Three use-cases share the same core loop:

| Use-case | How it uses the nav agent |
|---|---|
| Autonomous computer agent | Default: sees screen, acts, loops until done |
| Wearable AI (Frame glasses) | Adds first-person frame from glasses as extra context |
| Robot learning | Adds robot camera frame; records demonstrations for imitation learning |

## What to Keep as Direct Connectors

Direct connectors stay **only** where the screen is not the right interface:

| Keep direct connector | Why |
|---|---|
| Home Assistant (smart home) | Sub-300ms voice latency; no screen needed |
| Apple Health streaming | Continuous sensor data; no UI |
| Push notifications | Event-driven; not task-driven |
| Car status | Realtime data endpoint; no screen |
| Wearable sensor stream | Raw biometric data; not screen-navigable |

Everything else — Gmail, Calendar, Notion, Slack, GitHub, Todoist, Finance, any browser app
— the navigation agent handles via the screen.

## Architecture After This Change

```
User (voice / text / Frame wearable / robot camera)
                    ↓
         CollectiveOS Orchestrator
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
  Computer Nav Agent      Sensor Connectors
  (all screen tasks)      (realtime only)
         │
  ┌──────┴─────────────────────────┐
  │  Perceive (hybrid)             │
  │    screencapture + AXUIElement │
  ├──────────────────────────────  │
  │  Plan (Gemini Flash)           │
  │    intent → sub-task steps     │
  ├──────────────────────────────  │
  │  Ground + Execute              │
  │    Gemini Flash (free tier)    │
  │    browser-use / pyautogui     │
  ├──────────────────────────────  │
  │  Verify + HITL gate            │
  │    before any write action     │
  └────────────────────────────────┘
         │
  ┌──────┴──────────────────────────────────┐
  │                                         │
  ▼                                         ▼
Wearable path                         Robot path
Frame streams first-person view       Robot camera sees screen
AI bridges real world → computer      Demonstrations recorded
                                      Imitation learning seeded
```

## Implementation Phases

### Phase 1 — Core Loop (this PR)
- `src/agents/nav_agent.py` — perceive-decide-act loop
  - Hybrid perceive: screencapture + AppleScript AXUIElement
  - **Browser path**: browser-use (MIT) + Gemini Flash free tier via langchain-google-genai
  - **Desktop path**: Gemini Flash vision loop (free tier, 1000 req/day) + pyautogui
  - Task router: browser-signal keywords → browser path; everything else → desktop vision path
  - HITL gate: pause before send / submit / delete / purchase
  - LangGraph tool wrapper: `navigate_computer(task, context)`
- Updated `.env.example` — removed paid Claude CU API references; GEMINI_API_KEY covers nav agent
- Added `browser-use` to `requirements.txt` (MIT, free); `langchain-google-genai` already present

### Phase 2 — Smarter Perceive
- Replace AppleScript AX with `pyobjc` AXUIElement (full element tree, not just frontmost)
- OpenCV histogram diff to verify each action actually changed the screen
- App-specific context hints injected into the system prompt (reduces iterations for
  Gmail, Calendar, Slack, etc.)
- Per-task iteration budget based on task complexity estimate

### Phase 3 — Routing Update
- Update `multi_agent.py` to route "screen tasks" to `NavAgent` as primary tool
- Task classifier: screen task vs. sensor task vs. voice-only
- Retire individual connectors as the nav agent proves reliable for each domain

### Phase 4 — Wearable Integration
- Frame SDK: receive JPEG frames from glasses over WiFi/BLE
- Pass `first_person_frame=` to `NavAgent.run()`
- New intent: user looks at something in the real world → AI acts on the computer
  (e.g., "book that restaurant" while glasses see the restaurant sign)
- Glasses can also receive text overlay responses (Frame's in-lens display)

### Phase 5 — Robot Learning
- Robot camera streams frames via ROS2 topic → convert to JPEG
- Pass `robot_camera_frame=` + `record=True` to `NavAgent.run()`
- Demonstrations saved to `data/demonstrations/` (JSON: before_screenshot + action)
- Imitation learning pipeline: demonstrations → policy training (follow-up phase)
- Robot watches the nav agent and learns which actions produce which outcomes

## Key Technical Decisions

| Decision | Choice | Why |
|---|---|---|
| Computer use backend | Gemini Flash (free tier) + browser-use (MIT) | Zero cost; 1000 req/day free; already in stack |
| Perceive mode | Hybrid: AX tree + screenshot | AT is 40-100× faster; vision fills gaps |
| Action execution | pyautogui | Cross-platform, simple, no dep |
| HITL trigger | Keyword match on action text | Low-latency, catches send/delete/submit |
| Robot learning format | JSON demonstrations | Simple, importable to any IL framework |
| Planning model | Gemini Flash (existing) | Cost-efficient; CU handles grounding |

## File Changes Summary

```
src/agents/nav_agent.py          ← NEW  (Phase 1, free stack)
src/multi_agent.py               ← UPDATE routing (Phase 3)
docs/nav_agent_plan.md           ← this file
requirements.txt                 ← added pyautogui, Pillow, browser-use
.env.example                     ← updated: removed paid CU API refs; GEMINI_API_KEY covers nav agent
data/demonstrations/             ← created at runtime (Phase 5)
```
