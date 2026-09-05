"""Tool risk-tier registry — single source of truth for all CollectiveOS tools.

Three tiers:
  READ        — read-only; never requires confirmation.
  WRITE       — mutates local state or queues an action for review.
                Requires user confirmation via the HITL interrupt gate.
  DESTRUCTIVE — controls a physical device or takes an action that cannot
                easily be undone.
                Requires confirmation AND signals the client to show a stronger warning.

Any tool name not listed here defaults to WRITE (conservative fallback).

Note: external-service API connectors (Gmail, Calendar, Todoist, Slack, etc.)
have been removed. All screen-navigable tasks go through navigate_computer.
"""

from __future__ import annotations

READ        = "read"
WRITE       = "write"
DESTRUCTIVE = "destructive"

# ---------------------------------------------------------------------------
# Canonical tier assignments
# ---------------------------------------------------------------------------

TOOL_TIERS: dict[str, str] = {
    # ── Memory ──────────────────────────────────────────────────────────────
    "memory_remember":           WRITE,
    "memory_list":               READ,
    "memory_forget":             WRITE,
    "memory_graph_query":        READ,

    # ── Observability ────────────────────────────────────────────────────────
    "usage_summary":             READ,
    "mcp_list_servers":          READ,

    # ── Home Assistant ───────────────────────────────────────────────────────
    "get_devices":               READ,
    "get_device_state":          READ,
    "control_device":            DESTRUCTIVE,
    "set_light":                 WRITE,

    # ── Spotify ──────────────────────────────────────────────────────────────
    "spotify_now_playing":       READ,
    "spotify_get_devices":       READ,
    "spotify_control":           WRITE,
    "spotify_set_volume":        WRITE,
    "spotify_search_play":       WRITE,

    # ── Mac system ───────────────────────────────────────────────────────────
    "get_system_info":           READ,
    "get_wifi_info":             READ,
    "show_notification":         WRITE,
    "open_application":          WRITE,
    "set_system_volume":         WRITE,

    # ── Local filesystem ─────────────────────────────────────────────────────
    "list_directory":            READ,
    "read_local_file":           READ,
    "write_local_file":          WRITE,

    # ── Health ────────────────────────────────────────────────────────────────
    "health_get_sleep":          READ,
    "health_get_activity":       READ,
    "health_get_readiness":      READ,

    # ── Car connector ─────────────────────────────────────────────────────────
    "car_get_status":            READ,
    "car_lock":                  DESTRUCTIVE,
    "car_climate":               DESTRUCTIVE,

    # ── Smart appliances ──────────────────────────────────────────────────────
    "appliances_list":           READ,
    "appliances_get_status":     READ,
    "appliances_control":        DESTRUCTIVE,

    # ── iOS push notifications ────────────────────────────────────────────────
    "push_notification":         WRITE,

    # ── Output bus ───────────────────────────────────────────────────────────
    "notify_user":               WRITE,

    # ── Wearable devices ─────────────────────────────────────────────────────
    "wearable_get_events":       READ,
    "wearable_list_devices":     READ,

    # ── Robot (ROS2 MCP) ──────────────────────────────────────────────────────
    "robot_status":              READ,
    "robot_move":                DESTRUCTIVE,
    "robot_cancel":              DESTRUCTIVE,
    "robot_navigate":            DESTRUCTIVE,
    "robot_describe_scene":      READ,

    # ── Orchestrator ─────────────────────────────────────────────────────────
    "task_plan":                 WRITE,
    "task_status":               READ,
    "task_list":                 READ,
    "task_cancel":               WRITE,

    # ── Computer Navigation Agent ─────────────────────────────────────────────
    # Marked WRITE (not DESTRUCTIVE) because the nav agent has its own internal
    # HITL gate for irreversible sub-actions (send, delete, purchase …).
    "navigate_computer":         WRITE,
}


# ---------------------------------------------------------------------------
# Derived sets — used by agent.py and api.py
# ---------------------------------------------------------------------------

WRITE_TOOLS: frozenset[str] = frozenset(
    name for name, tier in TOOL_TIERS.items() if tier in (WRITE, DESTRUCTIVE)
)

DESTRUCTIVE_TOOLS: frozenset[str] = frozenset(
    name for name, tier in TOOL_TIERS.items() if tier == DESTRUCTIVE
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tier_of(name: str) -> str:
    """Return the risk tier for a tool name. Unknown tools default to WRITE."""
    return TOOL_TIERS.get(name, WRITE)


def is_write(name: str) -> bool:
    """True if the tool requires HITL confirmation (WRITE or DESTRUCTIVE)."""
    return tier_of(name) in (WRITE, DESTRUCTIVE)


def is_destructive(name: str) -> bool:
    """True if the tool controls physical hardware or is otherwise irreversible."""
    return tier_of(name) == DESTRUCTIVE
