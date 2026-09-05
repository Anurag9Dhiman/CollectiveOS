"""Tool risk-tier registry — single source of truth for all CollectiveOS tools.

Three tiers:
  READ        — read-only; never requires confirmation.
  WRITE       — mutates local state (calendar, tasks, notes, memory, files).
                Requires user confirmation via the HITL interrupt gate.
  DESTRUCTIVE — sends a message to an external party, controls a physical device,
                or takes an action that cannot easily be undone.
                Requires confirmation AND signals the client to show a stronger warning.

Any tool name not listed here defaults to WRITE (conservative fallback).

Import pattern — prefer the helpers over the raw dicts:
    from src.tool_registry import tier_of, WRITE_TOOLS, DESTRUCTIVE_TOOLS
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

    # ── External AI models ───────────────────────────────────────────────────
    "ai_ask":                    READ,
    "ai_compare":                READ,

    # ── Google Calendar ──────────────────────────────────────────────────────
    "get_calendar_events":       READ,
    "create_event":              WRITE,

    # ── Gmail ────────────────────────────────────────────────────────────────
    "get_recent_emails":         READ,
    "search_emails":             READ,
    "create_draft":              WRITE,
    "send_email":                DESTRUCTIVE,   # sends to external party

    # ── Google Drive ─────────────────────────────────────────────────────────
    "list_drive_files":          READ,
    "read_drive_file":           READ,

    # ── Todoist ──────────────────────────────────────────────────────────────
    "get_tasks":                 READ,
    "get_projects":              READ,
    "add_task":                  WRITE,
    "complete_task":             WRITE,
    "update_task":               WRITE,

    # ── Home Assistant ───────────────────────────────────────────────────────
    "get_devices":               READ,
    "get_device_state":          READ,
    "control_device":            DESTRUCTIVE,   # physical device control
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

    # ── Web search ───────────────────────────────────────────────────────────
    "web_search":                READ,

    # ── iMessage ─────────────────────────────────────────────────────────────
    "imessage_get_messages":     READ,
    "imessage_send":             DESTRUCTIVE,   # sends to external party

    # ── Screen capture ───────────────────────────────────────────────────────
    "capture_screen":            READ,

    # ── Local filesystem ─────────────────────────────────────────────────────
    "list_directory":            READ,
    "read_local_file":           READ,
    "write_local_file":          WRITE,

    # ── Browser ──────────────────────────────────────────────────────────────
    "browser_get_active_tab":    READ,
    "browser_list_tabs":         READ,
    "browser_open_url":          WRITE,

    # ── Apple native (Contacts / Reminders / Notes / Clipboard) ─────────────
    "contacts_search":           READ,
    "reminders_list":            READ,
    "reminders_add":             WRITE,
    "reminders_complete":        WRITE,
    "notes_list":                READ,
    "notes_read":                READ,
    "notes_create":              WRITE,
    "notes_append":              WRITE,
    "clipboard_read":            READ,
    "clipboard_write":           WRITE,

    # ── Telegram ─────────────────────────────────────────────────────────────
    "telegram_get_messages":     READ,
    "telegram_send":             DESTRUCTIVE,   # sends to external party

    # ── Notion ───────────────────────────────────────────────────────────────
    "notion_search":             READ,
    "notion_read_page":          READ,
    "notion_create_page":        WRITE,
    "notion_append_to_page":     WRITE,

    # ── GitHub ───────────────────────────────────────────────────────────────
    "github_list_repos":         READ,
    "github_list_prs":           READ,
    "github_list_issues":        READ,
    "github_get_ci_status":      READ,
    "github_create_issue":       WRITE,

    # ── Slack ─────────────────────────────────────────────────────────────────
    "slack_list_channels":       READ,
    "slack_read_messages":       READ,
    "slack_send_message":        DESTRUCTIVE,   # sends to external party / channel

    # ── Health ────────────────────────────────────────────────────────────────
    "health_get_sleep":          READ,
    "health_get_activity":       READ,
    "health_get_readiness":      READ,

    # ── Finance ───────────────────────────────────────────────────────────────
    "finance_get_accounts":      READ,
    "finance_get_transactions":  READ,
    "finance_get_spending_summary": READ,

    # ── Car connector ─────────────────────────────────────────────────────────
    "car_get_status":            READ,
    "car_lock":                  DESTRUCTIVE,   # physical vehicle action
    "car_climate":               DESTRUCTIVE,   # physical vehicle action

    # ── Smart appliances ──────────────────────────────────────────────────────
    "appliances_list":           READ,
    "appliances_get_status":     READ,
    "appliances_control":        DESTRUCTIVE,   # physical device control

    # ── VisualOS connector ────────────────────────────────────────────────────
    "lens_analyze":              READ,

    # ── iOS push notifications ────────────────────────────────────────────────
    "push_notification":         WRITE,

    # ── Wearable devices ─────────────────────────────────────────────────────
    "wearable_get_events":       READ,
    "wearable_list_devices":     READ,

    # ── Robot (ROS2 MCP) ──────────────────────────────────────────────────────
    "robot_status":              READ,
    "robot_move":                DESTRUCTIVE,   # physical motion
    "robot_cancel":              DESTRUCTIVE,   # emergency stop (also needs gate)
    "robot_navigate":            DESTRUCTIVE,   # path-planned motion between rooms
    "robot_describe_scene":      READ,

    # ── Computer Navigation Agent ─────────────────────────────────────────────
    # Marked WRITE (not DESTRUCTIVE) because the nav agent has its own internal
    # HITL gate for irreversible sub-actions (send, delete, purchase …).
    # The LangGraph gate fires once before the agent starts; the internal gate
    # fires again before each irreversible step inside the agent loop.
    "navigate_computer":         WRITE,
    "computer_use":              WRITE,         # legacy connector — prefer navigate_computer
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
    """True if the tool sends to an external party or controls physical hardware."""
    return tier_of(name) == DESTRUCTIVE
