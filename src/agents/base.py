"""Base classes for CollectiveOS agent clients.

An AgentClient wraps one peer agent service (VisualOS, a future CalendarOS, etc.)
and presents a uniform call() interface to the orchestrator.

AgentRegistry is a lightweight in-memory registry. On startup the orchestrator
populates it from environment vars and from the agent_connectors table.  Remote
agents can also self-register by calling POST /agents/register.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("collectiveos.agents")


@dataclass
class AgentResult:
    """Unified result returned by every AgentClient.call()."""
    text: str
    session_id: str | None = None          # e.g. VisualOS scan session id
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentClient(ABC):
    """Abstract base for all peer-agent clients."""

    @property
    @abstractmethod
    def capabilities(self) -> list[str]:
        """List of capability tags this agent covers, e.g. ["visual", "screen"]."""

    @abstractmethod
    def call(self, text: str, context: dict | None = None, **kwargs) -> AgentResult:
        """Invoke the agent with a user message and optional context dict."""

    def is_healthy(self) -> bool:
        """Override to add a liveness check (used by the registry health sweep)."""
        return True


class AgentRegistry:
    """In-memory registry of named AgentClient instances.

    Thread-safe for reads; registrations happen only at startup or via the
    /agents/register endpoint (serialised by FastAPI's event loop).
    """

    _agents: dict[str, AgentClient] = {}

    @classmethod
    def register(cls, name: str, client: AgentClient) -> None:
        cls._agents[name] = client
        logger.info("Agent registered: %s  capabilities=%s", name, client.capabilities)

    @classmethod
    def unregister(cls, name: str) -> bool:
        if name in cls._agents:
            del cls._agents[name]
            logger.info("Agent unregistered: %s", name)
            return True
        return False

    @classmethod
    def get(cls, name: str) -> AgentClient | None:
        return cls._agents.get(name)

    @classmethod
    def find_by_capability(cls, cap: str) -> AgentClient | None:
        for agent in cls._agents.values():
            if cap in agent.capabilities:
                return agent
        return None

    @classmethod
    def list_all(cls) -> list[dict]:
        return [
            {"name": name, "capabilities": client.capabilities, "healthy": client.is_healthy()}
            for name, client in cls._agents.items()
        ]
