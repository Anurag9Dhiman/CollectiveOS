"""MCP client manager for CollectiveOS.

Reads mcp_servers.json (or the path in MCP_SERVERS_CONFIG), connects to each
configured MCP server at startup, discovers their tools, and makes them callable
via call_tool().

Config file format (JSON):
{
  "servers": [
    {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Documents"],
      "env": {}
    },
    {
      "name": "github",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"}
    }
  ]
}

Environment variable values are expanded with os.path.expandvars.
Servers that fail to connect are skipped with a warning.

Tool registration in assistant_starter.py:
  - Each MCP tool is registered as  mcp_{server}_{tool}  in TOOL_FUNCTIONS / TOOLS.
  - Write-capable MCP tools must be added to WRITE_TOOLS in agent.py manually
    (the MCP spec has no read/write flag, so classify conservatively on first use).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from contextlib import AsyncExitStack
from typing import Any

logger = logging.getLogger("collectiveos.mcp")

# ---------------------------------------------------------------------------
# Background event loop — MCP sessions are async; we run them here forever
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
_loop_thread = threading.Thread(target=_loop.run_forever, daemon=True, name="mcp-loop")
_loop_thread.start()


def _run(coro) -> Any:
    """Submit a coroutine to the MCP event loop and block until done."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=30)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_sessions: dict[str, Any] = {}          # server_name → ClientSession
_tool_index: dict[str, dict] = {}       # "mcp_{server}_{tool}" → tool info
_exit_stack: AsyncExitStack | None = None
_loaded = False


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _config_path() -> str | None:
    explicit = os.environ.get("MCP_SERVERS_CONFIG", "")
    if explicit:
        return explicit
    candidate = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mcp_servers.json",
    )
    return candidate if os.path.exists(candidate) else None


async def _connect_all(cfg: dict) -> None:
    global _exit_stack
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        logger.warning("mcp package not installed — MCP servers unavailable. pip install mcp")
        return

    _exit_stack = AsyncExitStack()
    await _exit_stack.__aenter__()

    for server_cfg in cfg.get("servers", []):
        name = server_cfg.get("name", "unknown")
        try:
            raw_env = server_cfg.get("env") or {}
            expanded_env = {k: os.path.expandvars(v) for k, v in raw_env.items()}
            merged_env = {**os.environ, **expanded_env}

            params = StdioServerParameters(
                command=server_cfg["command"],
                args=server_cfg.get("args", []),
                env=merged_env,
            )
            read, write = await _exit_stack.enter_async_context(stdio_client(params))
            session = await _exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            _sessions[name] = session

            # Discover tools
            result = await session.list_tools()
            for tool in result.tools:
                key = _tool_key(name, tool.name)
                _tool_index[key] = {
                    "server": name,
                    "mcp_name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema or {"type": "object", "properties": {}},
                }

            logger.info("MCP server '%s' connected — %d tools", name, len(result.tools))

        except Exception as exc:
            logger.warning("MCP server '%s' failed to connect: %s", name, exc)


def load() -> None:
    """Connect to all configured MCP servers. Idempotent; safe to call at import time."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    path = _config_path()
    if path is None:
        logger.debug("No mcp_servers.json found — MCP disabled")
        return

    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except Exception as exc:
        logger.warning("Could not read MCP config %s: %s", path, exc)
        return

    _run(_connect_all(cfg))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_key(server: str, mcp_tool_name: str) -> str:
    """Stable CollectiveOS tool name for an MCP tool: mcp_{server}_{tool}."""
    safe = mcp_tool_name.replace("-", "_").replace(" ", "_").replace("/", "_").lower()
    return f"mcp_{server}_{safe}"


def list_servers() -> list[dict]:
    """Return connected server names and their tool counts."""
    counts: dict[str, int] = {}
    for info in _tool_index.values():
        counts[info["server"]] = counts.get(info["server"], 0) + 1
    return [{"server": s, "tools": n} for s, n in counts.items()]


def list_tools() -> list[dict]:
    """Return all discovered MCP tools in CollectiveOS TOOLS-list format."""
    result = []
    for key, info in _tool_index.items():
        result.append({
            "name": key,
            "description": f"[{info['server']}] {info['description']}",
            "input_schema": info["input_schema"],
        })
    return result


def tool_callables() -> dict[str, Any]:
    """Return a dict of {tool_key: callable} for all discovered MCP tools."""
    out = {}
    for key, info in _tool_index.items():
        server = info["server"]
        mcp_name = info["mcp_name"]
        out[key] = _make_callable(server, mcp_name)
    return out


def _make_callable(server: str, mcp_name: str):
    def _call(**kwargs) -> str:
        return call_tool(server, mcp_name, kwargs)
    _call.__name__ = f"mcp_{server}_{mcp_name}"
    return _call


def call_tool(server: str, mcp_tool_name: str, args: dict) -> str:
    """Call an MCP tool synchronously. Returns the result as a string."""
    session = _sessions.get(server)
    if session is None:
        return f"[ERROR: MCP server '{server}' is not connected]"
    return _run(_async_call(session, mcp_tool_name, args))


async def _async_call(session, tool_name: str, args: dict) -> str:
    result = await session.call_tool(tool_name, arguments=args)
    parts = []
    for content in result.content:
        if hasattr(content, "text"):
            parts.append(content.text)
        else:
            parts.append(str(content))
    return "\n".join(parts) if parts else "(no output)"
