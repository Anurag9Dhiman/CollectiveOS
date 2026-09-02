"""
Simulated robot environment — drop-in backend when ROS2_MCP_URL is absent.

Models a 5-room home as a graph. The robot has a current room, battery level,
and idle status. Navigation uses BFS to find the shortest path between rooms.
State persists to .robot_sim_state.json so position survives across restarts.

Room graph:
    bedroom ─── hallway ─── living_room
                  │
               office
                  │
               kitchen

This module is internal — external code always goes through ros2_mcp.py.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque

_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".robot_sim_state.json",
)

_GRAPH: dict[str, list[str]] = {
    "hallway":     ["bedroom", "living_room", "office"],
    "bedroom":     ["hallway"],
    "living_room": ["hallway"],
    "office":      ["hallway", "kitchen"],
    "kitchen":     ["office"],
}

ROOMS: list[str] = list(_GRAPH.keys())

_SCENES: dict[str, str] = {
    "hallway": (
        "The hallway connects all rooms. Front door with a deadbolt, coat rack "
        "with a few jackets, shoe rack near the entrance, umbrella stand. "
        "Exits: bedroom (north), living room (west), office (east)."
    ),
    "bedroom": (
        "Quiet bedroom. Double bed against the north wall, wardrobe with sliding "
        "doors, bedside table with a reading lamp and water glass, desk with a "
        "phone charger and laptop. Blackout curtains drawn. "
        "Exit: hallway (south)."
    ),
    "living_room": (
        "Open living room. Grey L-shaped sofa facing a 55-inch wall-mounted TV, "
        "glass coffee table, tall bookshelf with books and a speaker, "
        "fiddle-leaf fig plant in the south-west corner. Good natural light. "
        "Exit: hallway (east)."
    ),
    "office": (
        "Home office. Sit-stand desk with a desktop computer and dual monitors, "
        "laser printer, whiteboard with handwritten notes, filing cabinet, "
        "ergonomic chair. "
        "Exits: hallway (west), kitchen (south)."
    ),
    "kitchen": (
        "Modern kitchen. Stainless-steel refrigerator, stove with induction top, "
        "microwave mounted above counter, dishwasher under the counter, "
        "coffee maker and electric kettle on the worktop. "
        "Exit: office (north)."
    ),
}


def _default_state() -> dict:
    return {
        "room": "hallway",
        "battery": 92,
        "status": "idle",
        "last_action": None,
        "last_action_ts": None,
    }


def _load() -> dict:
    try:
        with open(_STATE_FILE) as f:
            data = json.load(f)
        if data.get("room") not in _GRAPH:
            return _default_state()
        return data
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return _default_state()


def _save(state: dict) -> None:
    try:
        with open(_STATE_FILE, "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def get_state() -> dict:
    """Return the current robot state as a dict (used by the API status endpoint)."""
    return _load()


def sim_status() -> str:
    s = _load()
    lines = [
        f"Status: {s['status'].upper()}",
        f"Location: {s['room'].replace('_', ' ').title()}",
        f"Battery: {s['battery']}%",
        "Simulator: active (ROS2_MCP_URL not configured)",
    ]
    if s.get("last_action"):
        ts = f" at {s['last_action_ts']}" if s.get("last_action_ts") else ""
        lines.append(f"Last action: {s['last_action']}{ts}")
    return "\n".join(lines)


def _bfs(start: str, goal: str) -> list[str] | None:
    if start == goal:
        return [start]
    queue: deque[list[str]] = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        for nb in _GRAPH.get(path[-1], []):
            if nb == goal:
                return path + [nb]
            if nb not in visited:
                visited.add(nb)
                queue.append(path + [nb])
    return None


def sim_navigate(destination: str) -> str:
    dest = destination.lower().replace(" ", "_").replace("-", "_")
    if dest not in _GRAPH:
        known = ", ".join(r.replace("_", " ") for r in ROOMS)
        return f"[ERROR: unknown room '{destination}'. Known rooms: {known}]"

    state = _load()
    current = state["room"]

    if current == dest:
        return f"Already in {dest.replace('_', ' ').title()} — no movement needed."

    path = _bfs(current, dest)
    if path is None:
        return f"[ERROR: no path found from {current} to {dest}]"

    hops = len(path) - 1
    state["battery"] = max(5, state["battery"] - hops * 2)
    state["room"] = dest
    state["status"] = "idle"
    route = " → ".join(r.replace("_", " ").title() for r in path)
    state["last_action"] = f"Navigated: {route}"
    state["last_action_ts"] = time.strftime("%H:%M:%S")
    _save(state)

    return (
        f"Navigation complete.\n"
        f"Route: {route}\n"
        f"Current location: {dest.replace('_', ' ').title()}\n"
        f"Battery: {state['battery']}%"
    )


def sim_describe_scene() -> str:
    state = _load()
    room = state["room"]
    scene = _SCENES.get(room, "No scene data for this location.")
    return (
        f"Current location: {room.replace('_', ' ').title()}\n"
        f"Visual scan:\n{scene}\n"
        f"Battery: {state['battery']}%"
    )


def sim_move(direction: str, distance_m: float = 1.0) -> str:
    state = _load()
    drain = max(1, int(distance_m))
    state["battery"] = max(5, state["battery"] - drain)
    state["last_action"] = f"Moved {direction} {distance_m}m (within {state['room'].replace('_', ' ').title()})"
    state["last_action_ts"] = time.strftime("%H:%M:%S")
    _save(state)
    return (
        f"Moved {direction} {distance_m}m.\n"
        f"Still in: {state['room'].replace('_', ' ').title()}\n"
        f"Battery: {state['battery']}%\n"
        f"Tip: use robot_navigate to move between rooms."
    )


def sim_cancel() -> str:
    state = _load()
    state["status"] = "idle"
    state["last_action"] = "Emergency stop"
    state["last_action_ts"] = time.strftime("%H:%M:%S")
    _save(state)
    return f"Emergency stop. Robot is IDLE in {state['room'].replace('_', ' ').title()}."
