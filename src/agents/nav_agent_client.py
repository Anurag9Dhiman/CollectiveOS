"""NavAgentClient — wraps NavAgent for the AgentRegistry.

The sole execution engine for all task requests: every message that isn't a
screen-capture query routes here and is carried out entirely through GUI
interaction via UI-TARS (or browser-use for web tasks), with zero connectors.
"""

from __future__ import annotations

from .base import AgentClient, AgentResult


class NavAgentClient(AgentClient):
    """Registry adapter for NavAgent.  Exposes acall(); call() is a thin sync stub."""

    def __init__(self) -> None:
        from src.agents.nav_agent import NavAgent
        self._agent = NavAgent()

    @property
    def capabilities(self) -> list[str]:
        return ["task", "nav", "desktop", "browser", "general", "hitl"]

    def call(self, text: str, context: dict | None = None, **kwargs) -> AgentResult:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "NavAgentClient.call() must not be called from an async context — "
                "use acall() instead."
            )
        return asyncio.run(self.acall(text, context, **kwargs))

    async def acall(
        self,
        text: str,
        context: dict | None = None,
        *,
        hitl_callback=None,
        **_,
    ) -> AgentResult:
        from src.agents.nav_agent import NavResult

        ctx_str = ""
        if context:
            if isinstance(context, str):
                ctx_str = context
            elif isinstance(context, dict):
                # Flatten visual prefix injected by the orchestrator
                ctx_str = context.get("visual_prefix", "") or ""

        result: NavResult = await self._agent.run(
            text,
            context=ctx_str,
            hitl_callback=hitl_callback,
        )

        interrupted = result.status == "hitl_paused"
        return AgentResult(
            text=result.result,
            metadata={
                "interrupted": interrupted,
                "destructive": interrupted,
                "nav_status": result.status,
                "nav_steps": len(result.steps),
                "verified": result.verified,
                "pending_action": result.pending_action,
            },
        )

    def is_healthy(self) -> bool:
        return True
