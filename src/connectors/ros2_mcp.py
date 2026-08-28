"""
ROS2 MCP connector — stub that speaks to a ROS2 MCP server.

When ROS2_MCP_URL is set, calls are forwarded to a running MCP server
(e.g. github.com/tiny-robot/ros2-mcp-server). When the env var is absent
the connector returns descriptive stub data so the rest of the system
can be tested without a physical robot.

Safety: robot_move is in WRITE_TOOLS (requires HITL before any motion).
        Never attempt to start heating appliances or unsafe actuators.
"""
from __future__ import annotations

import json
import os

import requests

_BASE = os.environ.get("ROS2_MCP_URL", "").rstrip("/")
_TIMEOUT = 10


def _call(tool: str, args: dict) -> dict:
    """POST to the ROS2 MCP server; returns the result dict."""
    if not _BASE:
        raise RuntimeError("ROS2_MCP_URL is not set — stub mode active")
    resp = requests.post(
        f"{_BASE}/tools/{tool}",
        json={"args": args},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def robot_status() -> str:
    """Return the current status and position of the connected robot."""
    if not _BASE:
        return (
            "[Stub] Robot status: IDLE\n"
            "Position: x=0.0, y=0.0, heading=0°\n"
            "Battery: 87%\n"
            "Connected: False (ROS2_MCP_URL not configured)"
        )
    try:
        data = _call("robot_status", {})
        return json.dumps(data, indent=2)
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
    distance_m: how far to travel in metres (ignored for stop)
    speed_ms: speed in m/s (capped server-side at 0.5 m/s)
    """
    if direction not in ("forward", "backward", "left", "right", "stop"):
        return f"[ERROR: invalid direction '{direction}'. Use: forward, backward, left, right, stop]"

    if not _BASE:
        return (
            f"[Stub] Move command accepted: {direction} "
            f"{distance_m}m @ {speed_ms}m/s\n"
            "(ROS2_MCP_URL not configured — no actual movement)"
        )
    try:
        data = _call("robot_move", {
            "direction": direction,
            "distance_m": distance_m,
            "speed_ms": min(speed_ms, 0.5),
        })
        return json.dumps(data, indent=2)
    except Exception as exc:
        return f"[ERROR: {exc}]"


def robot_cancel() -> str:
    """Stop all robot motion immediately."""
    if not _BASE:
        return "[Stub] Emergency stop sent (ROS2_MCP_URL not configured)"
    try:
        data = _call("robot_cancel", {})
        return json.dumps(data, indent=2)
    except Exception as exc:
        return f"[ERROR: {exc}]"
