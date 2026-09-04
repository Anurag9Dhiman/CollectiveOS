"""CollectiveOS multi-agent orchestrator.

Routes incoming messages to the right peer agent based on registered
capabilities.  Handles context enrichment (pulling VisualOS scan context
when a scan_session_id is present) before delegating to the task agent.

Architecture
────────────
  VoiceOS  ──WS /v1/ws──►  voice_gateway  ──►  orchestrator.run()
  /chat    ──HTTP──────►  api.py          ──►  orchestrator.run()
                                                     │
                          ┌──────────────────────────┤
                          │                          │
                    VisualOSClient             TaskAgentClient
                 (context enrichment)       (LangGraph agent loop)

Capability routing
──────────────────
  screen-capture intent, no image_b64  →  capture screen → VisualOS /analyze
                                            (bypasses task agent; returns card directly)
  scan_session_id in entity_refs       →  fetch scan context from VisualOS,
                                            inject as prefix → task agent
  image_b64 present                    →  task agent (Gemini Vision + all tools)
  everything else                      →  task agent (router selects tools)

Adding a new peer agent
───────────────────────
  1. Implement AgentClient in src/agents/
  2. Call AgentRegistry.register("name", YourClient()) at startup
     — or POST /agents/register to self-register from the remote service.
"""

from __future__ import annotations

import logging
import os

from src.agents.base import AgentRegistry, AgentResult
from src.agents.task_agent import TaskAgentClient
from src.agents.visual_agent import VisualOSClient

logger = logging.getLogger("collectiveos.orchestrator")


# ---------------------------------------------------------------------------
# Registry bootstrap — called once from api.py lifespan
# ---------------------------------------------------------------------------

def setup() -> None:
    """Register built-in agents from environment config.

    Called at application startup.  Remote agents may also self-register
    later via POST /agents/register.
    """
    # Task agent is always available
    AgentRegistry.register("task", TaskAgentClient())

    # VisualOS — register when LENS_URL is configured
    lens_url = os.environ.get("LENS_URL", "").rstrip("/")
    if lens_url:
        api_key = os.environ.get("LENS_API_KEY", "")
        AgentRegistry.register("visual", VisualOSClient(lens_url, api_key))

    # Load any agents that were persisted in the DB from a previous session
    _load_from_db()


def _load_from_db() -> None:
    """Re-register any agent_connectors rows from the DB (survive restarts)."""
    try:
        from src.db import connect
        conn = connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, url, protocol, capabilities, api_key "
                "FROM agent_connectors WHERE enabled = TRUE"
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as exc:
        logger.debug("Could not load agent_connectors from DB: %s", exc)
        return

    for name, url, protocol, capabilities, api_key in rows:
        if AgentRegistry.get(name):
            continue   # already registered (from env)
        client = _build_client(name, url, protocol, capabilities or [], api_key or "")
        if client:
            AgentRegistry.register(name, client)


def _build_client(
    name: str, url: str, protocol: str, capabilities: list[str], api_key: str
) -> "AgentClient | None":
    from src.agents.base import AgentClient

    # Known named agents get their typed client
    if name == "visual" or "visual" in capabilities:
        return VisualOSClient(url, api_key)

    # Generic HTTP agent — a future base class; for now return None
    logger.warning("No client class for agent %r (protocol=%s) — skipped", name, protocol)
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Visual intent detection
# ---------------------------------------------------------------------------

# High-confidence phrases that mean "look at / capture the screen right now".
# Deliberately narrow — ambiguous queries fall through to the task agent which
# uses the capture_screen tool when it decides one is needed.
_VISUAL_CAPTURE_PHRASES: frozenset[str] = frozenset({
    "screenshot", "screen capture", "capture screen", "take a screenshot",
    "what's on my screen", "what is on my screen", "what do you see",
    "capture my screen", "look at my screen", "describe my screen",
    "what am i looking at", "what's on the screen", "what is on the screen",
    "scan my screen", "analyze my screen", "read my screen",
})


def _is_screen_capture_query(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _VISUAL_CAPTURE_PHRASES)


def run(
    text: str,
    entity_refs: dict | None = None,
    image_b64: str | None = None,
    image_mime: str = "image/jpeg",
    system_prompt: str = "",
    thread_id: str = "default",
) -> AgentResult:
    """Route one user message through the agent pipeline.

    Routing priority
    ────────────────
    1. Screen-capture intent (no image_b64) → capture screen → VisualOS /analyze
       Returns the card directly; no Gemini synthesis needed.
    2. scan_session_id in entity_refs → fetch VisualOS ScanContext → inject as
       prefix → task agent answers the follow-up with full tool access.
    3. Everything else → task agent (LangGraph loop, all tools available).

    image_b64 present → always goes to the task agent so Gemini Vision sees it
    in conversation context alongside all available tools.

    Returns an AgentResult whose .metadata carries:
      interrupted (bool)    — HITL approval needed (task agent only)
      destructive (bool)    — pending action is high-risk
      scan_session_id (str) — VisualOS session id when a new scan was created
    """
    entity_refs = entity_refs or {}

    # ── Route 1: Screen-capture intent ──────────────────────────────────────
    # Only trigger when no inline image is already provided (if the client
    # attached image_b64 they want Gemini Vision + full tool context).
    if not image_b64 and _is_screen_capture_query(text):
        visual_agent = AgentRegistry.find_by_capability("visual")
        if visual_agent is not None:
            result = _route_to_visualos(visual_agent, text)
            if result is not None:
                return result
        # VisualOS unavailable or not registered → fall through to task agent

    # ── Route 2: Follow-up on existing scan → enrich + task agent ───────────
    visual_prefix = _enrich_visual_context(text, entity_refs)
    enriched = (
        f"[Visual context]\n{visual_prefix}\n\n[User message]\n{text}"
        if visual_prefix else text
    )

    # ── Route 3: Task agent ──────────────────────────────────────────────────
    task_agent = AgentRegistry.find_by_capability("task")
    if task_agent is None:
        logger.error("No task agent registered — returning error reply")
        return AgentResult(
            text="Assistant is not available right now.",
            metadata={"interrupted": False, "destructive": False},
        )

    return task_agent.call(
        enriched,
        context=entity_refs,
        system_prompt=system_prompt,
        thread_id=thread_id,
        image_b64=image_b64,
        image_mime=image_mime,
    )


def _route_to_visualos(visual_agent: "VisualOSClient", question: str) -> "AgentResult | None":
    """Capture the screen and analyse it via VisualOS. Returns None on failure."""
    import os
    import platform
    import subprocess
    import tempfile

    if platform.system() != "Darwin":
        return None   # screencapture is macOS-only; fall through to task agent

    tmp = tempfile.mktemp(suffix=".png")
    try:
        r = subprocess.run(
            ["screencapture", "-x", tmp],
            capture_output=True, timeout=15,
        )
        if r.returncode != 0 or not os.path.exists(tmp):
            return None
        subprocess.run(["sips", "-Z", "1280", tmp], capture_output=True, timeout=10)
        with open(tmp, "rb") as fh:
            image_bytes = fh.read()
    except Exception as exc:
        logger.warning("Screen capture failed in orchestrator: %s", exc)
        return None
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    result = visual_agent.analyze_image(image_bytes, question)

    if result.text.startswith("[VisualOS unavailable"):
        return None   # let task agent fall back to its own Gemini Vision path

    # Surface the scan_session_id in metadata so the caller can thread it
    # back as entity_refs["scan_session_id"] on the next turn.
    if result.session_id:
        result.metadata["scan_session_id"] = result.session_id

    return result


def _enrich_visual_context(text: str, entity_refs: dict) -> str:
    """Fetch VisualOS scan context if a scan_session_id is present."""
    session_id = entity_refs.get("scan_session_id")
    if not session_id:
        return ""

    visual_agent = AgentRegistry.find_by_capability("visual")
    if visual_agent is None:
        return ""

    result = visual_agent.call(text, context=entity_refs)
    return result.text   # empty string if unavailable or session expired


# ---------------------------------------------------------------------------
# DB persistence helpers (called from api.py agent endpoints)
# ---------------------------------------------------------------------------

def persist_agent(
    name: str, url: str, protocol: str, capabilities: list[str],
    api_key: str = "", health_url: str = "",
) -> None:
    """Upsert an agent_connectors row in the DB."""
    try:
        from src.db import connect
        conn = connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_connectors
                        (name, url, protocol, capabilities, api_key, health_url, enabled)
                    VALUES (%s, %s, %s, %s, %s, %s, TRUE)
                    ON CONFLICT (name) DO UPDATE SET
                        url          = EXCLUDED.url,
                        protocol     = EXCLUDED.protocol,
                        capabilities = EXCLUDED.capabilities,
                        api_key      = EXCLUDED.api_key,
                        health_url   = EXCLUDED.health_url,
                        enabled      = TRUE,
                        last_seen_at = NOW()
                    """,
                    (name, url, protocol, capabilities, api_key, health_url),
                )
        conn.close()
    except Exception as exc:
        logger.warning("Could not persist agent %r to DB: %s", name, exc)


def remove_agent(name: str) -> None:
    """Mark an agent_connectors row as disabled in the DB."""
    try:
        from src.db import connect
        conn = connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE agent_connectors SET enabled = FALSE WHERE name = %s",
                    (name,),
                )
        conn.close()
    except Exception as exc:
        logger.warning("Could not disable agent %r in DB: %s", name, exc)
