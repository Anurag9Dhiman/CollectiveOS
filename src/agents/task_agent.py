"""Task agent client — wraps the existing LangGraph agent loop.

This is the default agent for general task execution: tool calls, memory,
write-action HITL, etc. Every message that isn't routed to a specialist
agent ends up here.
"""

from __future__ import annotations

from .base import AgentClient, AgentResult


class TaskAgentClient(AgentClient):
    """Thin wrapper around src.agent.run() / src.agent.approve()."""

    @property
    def capabilities(self) -> list[str]:
        return ["task", "general", "memory", "tools", "hitl"]

    def call(
        self,
        text: str,
        context: dict | None = None,
        *,
        system_prompt: str = "",
        thread_id: str = "default",
        image_b64: str | None = None,
        image_mime: str = "image/jpeg",
        **_,
    ) -> AgentResult:
        from src.agent import run as agent_run

        reply, interrupted, destructive = agent_run(
            text,
            system_prompt=system_prompt,
            thread_id=thread_id,
            image_b64=image_b64,
            image_mime=image_mime,
        )
        return AgentResult(
            text=reply,
            metadata={"interrupted": interrupted, "destructive": destructive},
        )
