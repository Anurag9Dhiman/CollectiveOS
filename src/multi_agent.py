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
  scan_session_id in entity_refs  →  fetch scan context from VisualOS,
                                     inject into task agent as prefix
  image_b64 present               →  skip router, pass to task agent directly
  everything else                 →  task agent (router selects tools)

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

def run(
    text: str,
    entity_refs: dict | None = None,
    image_b64: str | None = None,
    image_mime: str = "image/jpeg",
    system_prompt: str = "",
    thread_id: str = "default",
) -> AgentResult:
    """Route one user message through the agent pipeline.

    Steps:
      1. If entity_refs carries a scan_session_id, fetch VisualOS context.
      2. Inject any visual context as a prefix to the message.
      3. Delegate to the task agent (LangGraph loop).

    Returns an AgentResult whose .metadata carries:
      interrupted (bool)  — HITL approval needed
      destructive (bool)  — pending action is high-risk
    """
    entity_refs = entity_refs or {}

    # ── Step 1: Visual context enrichment ──────────────────────────────────
    visual_prefix = _enrich_visual_context(text, entity_refs)

    # ── Step 2: Build enriched message ─────────────────────────────────────
    enriched = (
        f"[Visual context]\n{visual_prefix}\n\n[User message]\n{text}"
        if visual_prefix else text
    )

    # ── Step 3: Task agent ─────────────────────────────────────────────────
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
