"""
ROS2 MCP connector.

When ROS2_MCP_URL is set, calls are forwarded to a running MCP server
(e.g. github.com/tiny-robot/ros2-mcp-server).  When the env var is absent
the connector delegates to ros2_sim.py — a room-graph simulator that lets
the full agent → robot pipeline run without physical hardware.

Safety:
  robot_move and robot_navigate are DESTRUCTIVE (physical motion — HITL gate).
  robot_cancel is also DESTRUCTIVE (emergency stop).
  Never start heating appliances remotely.
"""
from __future__ import annotations

import json
import os

import requests

_BASE = os.environ.get("ROS2_MCP_URL", "").rstrip("/")
_TIMEOUT = 10


def _call(tool: str, args: dict) -> dict:
    resp = requests.post(
        f"{_BASE}/tools/{tool}",
        json={"args": args},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _sim():
    from src.connectors import ros2_sim
    return ros2_sim


# ---------------------------------------------------------------------------
# Existing tools — now route to simulator instead of returning inline stubs
# ---------------------------------------------------------------------------

def robot_status() -> str:
    """Return the current status and position of the connected robot."""
    if not _BASE:
        return _sim().sim_status()
    try:
        return json.dumps(_call("robot_status", {}), indent=2)
    except Exception as exc:
        return f"[ERROR: {exc}]"


def robot_move(
    direction: str,
    distance_m: float = 1.0,
    speed_ms: float = 0.3,
) -> str:
    """
    Command the robot to move in a direction.

    direction: forward | backward | left | right | stop
    distance_m: distance in metres (ignored for stop)
    speed_ms: speed in m/s (capped server-side at 0.5 m/s)

    Note: use robot_navigate to move between named rooms.
    """
    if direction not in ("forward", "backward", "left", "right", "stop"):
        return f"[ERROR: invalid direction '{direction}'. Use: forward, backward, left, right, stop]"
    if not _BASE:
        if direction == "stop":
            return _sim().sim_cancel()
        return _sim().sim_move(direction, distance_m)
    try:
        return json.dumps(_call("robot_move", {
            "direction": direction,
            "distance_m": distance_m,
            "speed_ms": min(speed_ms, 0.5),
        }), indent=2)
    except Exception as exc:
        return f"[ERROR: {exc}]"


def robot_cancel() -> str:
    """Stop all robot motion immediately (emergency stop)."""
    if not _BASE:
        return _sim().sim_cancel()
    try:
        return json.dumps(_call("robot_cancel", {}), indent=2)
    except Exception as exc:
        return f"[ERROR: {exc}]"


# ---------------------------------------------------------------------------
# New tools
# ---------------------------------------------------------------------------

def robot_navigate(destination: str) -> str:
    """
    Navigate the robot to a named room or location.

    Plans the shortest path between rooms and executes it.
    destination: bedroom | hallway | living_room | office | kitchen

    Requires HITL approval before execution (physical motion).
    """
    if not _BASE:
        return _sim().sim_navigate(destination)
    try:
        return json.dumps(_call("robot_navigate", {"destination": destination}), indent=2)
    except Exception as exc:
        return f"[ERROR: {exc}]"


def robot_describe_scene() -> str:
    """
    Describe what the robot can currently perceive in its environment.

    Returns a visual scan of the current room including visible objects,
    exits, and notable features.  Read-only — does not move the robot.
    """
    if not _BASE:
        return _sim().sim_describe_scene()
    try:
        return json.dumps(_call("robot_describe_scene", {}), indent=2)
    except Exception as exc:
        return f"[ERROR: {exc}]"
