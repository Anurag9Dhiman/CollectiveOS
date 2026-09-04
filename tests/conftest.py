"""
conftest.py — session-level mocks that let the full test suite run without
a live database, Gemini API key, Google OAuth, or any other external service.

Strategy
--------
1. Heavy *third-party* libraries (google-genai, psycopg2, langgraph, …) are
   replaced in sys.modules with MagicMock() BEFORE any src.* module is
   imported.  Python's import system then hands the mock to any
   `import X` or `from X import Y` statement in the source files.

2. src.assistant_starter is stubbed as a bare module (it has heavy side
   effects at module level — importing all connectors, loading MCP servers —
   that we don't want in tests).

3. Per-test monkeypatch fixtures override the remaining src.* functions
   (memory, agent, scheduler, conversations) so every test gets a clean slate.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# 1. Mock heavy third-party libraries
# ---------------------------------------------------------------------------
_MOCK_LIBS = [
    # Google / Gemini
    "google", "google.genai", "google.genai.types",
    "google.auth", "google.auth.transport", "google.auth.transport.requests",
    "google.oauth2", "google.oauth2.credentials", "google.oauth2.service_account",
    "googleapiclient", "googleapiclient.discovery", "googleapiclient.errors",
    # Postgres / async DB
    "psycopg2", "psycopg2.extras", "psycopg2.pool",
    "psycopg", "psycopg.rows", "asyncpg",
    # LangGraph / LangChain
    "langchain_google_genai",
    "langgraph", "langgraph.graph", "langgraph.graph.message",
    "langgraph.types", "langgraph.pregel",
    "langgraph.checkpoint", "langgraph.checkpoint.postgres",
    "langgraph_checkpoint_postgres",
    "langchain_core", "langchain_core.messages",
    "langsmith", "langmem",
    # APScheduler
    "apscheduler",
    "apscheduler.schedulers", "apscheduler.schedulers.background",
    "apscheduler.triggers", "apscheduler.triggers.cron",
    # ML / audio
    "sentence_transformers", "pyautogui",
    "sounddevice", "pyaudio", "numpy",
    # Misc external
    "spotipy", "spotipy.oauth2",
    "tavily",
    "structlog",
    "mcp", "mcp.client", "mcp.client.stdio", "mcp.types",
    "redis", "redis.asyncio",
    "openai",
]

for _lib in _MOCK_LIBS:
    if _lib not in sys.modules:
        sys.modules[_lib] = MagicMock()

# Link google sub-package mocks so `from google import genai` and
# `patch("google.genai.Client", ...)` both target the same object.
# Without this, `sys.modules["google"].genai` and `sys.modules["google.genai"]`
# are two separate MagicMocks and patches on one don't affect the other.
sys.modules["google"].genai = sys.modules["google.genai"]
sys.modules["google.genai"].types = sys.modules["google.genai.types"]

# ---------------------------------------------------------------------------
# 2. Stub src.assistant_starter (heavy module-level side effects)
# ---------------------------------------------------------------------------
def _stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


_ast = _stub("src.assistant_starter")
_ast.run_stream = MagicMock()   # type: ignore[attr-defined]
_ast.TOOLS = []                 # type: ignore[attr-defined]
_ast.TOOL_FUNCTIONS = {}        # type: ignore[attr-defined]
_ast.MODEL = "test-model"       # type: ignore[attr-defined]
_ast.run = MagicMock(return_value="stub reply")  # type: ignore[attr-defined]

_mcpm = _stub("src.mcp_client")
_mcpm.load = lambda: None           # type: ignore[attr-defined]
_mcpm.tool_callables = lambda: {}   # type: ignore[attr-defined]
_mcpm.list_tools = lambda: []       # type: ignore[attr-defined]
_mcpm.list_servers = lambda: []     # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# 3. Environment defaults for tests
# ---------------------------------------------------------------------------
os.environ.setdefault("API_TOKEN", "test-token")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# 4. Fixtures
# ---------------------------------------------------------------------------
import pytest  # noqa: E402  (must be after sys.modules patching)


@pytest.fixture(autouse=True)
def _mock_memory(monkeypatch):
    import src.memory as mem
    monkeypatch.setattr(mem, "search",            lambda q, limit=5: "", raising=False)
    monkeypatch.setattr(mem, "search_with_graph", lambda q, limit=5: "", raising=False)
    monkeypatch.setattr(mem, "save_smart",        lambda u, a: None,    raising=False)
    monkeypatch.setattr(mem, "save_fact",         lambda f: None,       raising=False)
    monkeypatch.setattr(mem, "list_facts",        lambda: [],           raising=False)
    monkeypatch.setattr(mem, "get_all_facts_str", lambda: "",           raising=False)
    monkeypatch.setattr(mem, "delete_fact",       lambda f: None,       raising=False)


@pytest.fixture(autouse=True)
def _mock_agent(monkeypatch):
    import src.agent as agent
    monkeypatch.setattr(
        agent, "run",
        lambda msg, system_prompt=None, thread_id=None, **kw: ("Test reply", False, False),
        raising=False,
    )
    monkeypatch.setattr(
        agent, "approve",
        lambda thread_id, approved: "Approved." if approved else "Cancelled.",
        raising=False,
    )


@pytest.fixture(autouse=True)
def _mock_scheduler(monkeypatch):
    import src.scheduler as sched
    monkeypatch.setattr(sched, "load_all",       lambda: None,   raising=False)
    monkeypatch.setattr(sched, "start",          lambda: None,   raising=False)
    monkeypatch.setattr(sched, "shutdown",       lambda: None,   raising=False)
    monkeypatch.setattr(sched, "reload_routine", lambda rid: None, raising=False)
    monkeypatch.setattr(sched, "remove_job",     lambda rid: None, raising=False)


@pytest.fixture(autouse=True)
def _mock_conversations(monkeypatch):
    import src.conversations as conv
    monkeypatch.setattr(conv, "create",             lambda user_id=1: 1,   raising=False)
    monkeypatch.setattr(conv, "save_message",       lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(conv, "list_conversations", lambda *a, **kw: [],   raising=False)
    monkeypatch.setattr(conv, "get_messages",       lambda cid: [],        raising=False)
    monkeypatch.setattr(conv, "search_messages",    lambda q, limit=20: [], raising=False)
    monkeypatch.setattr(conv, "load_history",       lambda cid: [],        raising=False)
    if hasattr(conv, "set_title"):
        monkeypatch.setattr(conv, "set_title", lambda *a: None, raising=False)


@pytest.fixture(autouse=True)
def _mock_orchestrator(monkeypatch):
    import src.multi_agent as ma
    from src.agents.base import AgentResult
    monkeypatch.setattr(ma, "setup", lambda: None, raising=False)
    monkeypatch.setattr(
        ma, "run",
        lambda text, entity_refs=None, image_b64=None, image_mime="image/jpeg",
               system_prompt="", thread_id="default":
            AgentResult(text="Test reply", metadata={"interrupted": False, "destructive": False}),
        raising=False,
    )


@pytest.fixture(autouse=True)
def _mock_permissions(monkeypatch):
    import src.permissions as perm
    monkeypatch.setattr(perm, "is_enabled", lambda name: True, raising=False)
    monkeypatch.setattr(perm, "all_statuses", lambda: {}, raising=False)


@pytest.fixture()
def client():
    """Return a FastAPI TestClient with all heavy deps mocked."""
    from fastapi.testclient import TestClient
    from src.api import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture()
def auth(client):
    """Pre-authenticated TestClient session (injects Bearer token)."""
    client.headers.update({"Authorization": "Bearer test-token"})
    return client
