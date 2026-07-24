"""
Permission store — per-connector enable/disable, backed by Postgres.

The table is created on first import, so no separate migration step is needed.

Layers
------
1. Connector-level toggle  (this module) — hard block if connector is disabled.
2. Write-tool confirmation  (system prompt) — model must ask before any write.
3. UI visibility            (index.html)   — write-tool calls shown in amber.
"""

from src.db import connect

# ---------------------------------------------------------------------------
# Connector → tools mapping
# ---------------------------------------------------------------------------

CONNECTOR_TOOLS: dict[str, list[str]] = {
    "google_calendar": ["get_calendar_events", "create_event"],
    "gmail":           ["get_recent_emails", "search_emails", "create_draft", "send_email"],
    "google_drive":    ["list_drive_files", "read_drive_file"],
    "todoist":         ["get_tasks", "get_projects", "add_task", "complete_task", "update_task"],
    "home_assistant":  ["get_devices", "get_device_state", "control_device", "set_light"],
    "spotify":         ["spotify_now_playing", "spotify_get_devices", "spotify_control",
                        "spotify_set_volume", "spotify_search_play"],
    "mac_system":      ["get_system_info", "get_wifi_info", "show_notification",
                        "open_application", "set_system_volume"],
    "web_search":      ["web_search"],
    "imessage":        ["imessage_get_messages", "imessage_send"],
    "screen_capture":  ["capture_screen"],
}

# Reverse index: tool name → connector
TOOL_CONNECTOR: dict[str, str] = {
    tool: conn
    for conn, tools in CONNECTOR_TOOLS.items()
    for tool in tools
}

# Tools that modify state or perform real-world actions — require user confirmation
WRITE_TOOLS: frozenset[str] = frozenset({
    "create_event",
    "create_draft", "send_email",
    "add_task", "complete_task", "update_task",
    "control_device", "set_light",
    "spotify_control", "spotify_set_volume", "spotify_search_play",
    "show_notification", "open_application", "set_system_volume",
    "imessage_send",
})

CONNECTOR_LABELS: dict[str, str] = {
    "google_calendar": "Google Calendar",
    "gmail":           "Gmail",
    "google_drive":    "Google Drive",
    "todoist":         "Todoist",
    "home_assistant":  "Home Assistant",
    "spotify":         "Spotify",
    "mac_system":      "Mac System",
    "web_search":      "Web Search",
    "imessage":        "iMessage",
    "screen_capture":  "Screen Capture",
}

# ---------------------------------------------------------------------------
# Table bootstrap (runs once on import)
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS connector_permissions (
    connector  TEXT PRIMARY KEY,
    enabled    BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_SEED = """
INSERT INTO connector_permissions (connector, enabled)
VALUES (%(connector)s, TRUE)
ON CONFLICT (connector) DO NOTHING;
"""


def _bootstrap() -> None:
    try:
        conn = connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE)
                for name in CONNECTOR_TOOLS:
                    cur.execute(_SEED, {"connector": name})
        conn.close()
    except Exception:
        pass  # Degrade gracefully — DB may not be available in every environment


_bootstrap()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_enabled(connector: str) -> bool:
    """Return True if the connector is allowed to run tools."""
    try:
        conn = connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT enabled FROM connector_permissions WHERE connector = %s",
                (connector,),
            )
            row = cur.fetchone()
        conn.close()
        return bool(row[0]) if row else True
    except Exception:
        return True  # Fail open — don't block tools if DB is unreachable


def check_tool(tool_name: str) -> tuple[bool, str]:
    """
    Return (allowed, error_message).
    allowed=True means the tool may run; error_message is empty.
    allowed=False means the connector is disabled; error_message explains why.
    """
    connector = TOOL_CONNECTOR.get(tool_name)
    if not connector:
        return True, ""

    if not is_enabled(connector):
        label = CONNECTOR_LABELS.get(connector, connector)
        return False, (
            f"Access to {label} is currently disabled. "
            f"Enable it in Settings (⚙) to use this feature."
        )
    return True, ""


def set_permission(connector: str, enabled: bool) -> None:
    """Enable or disable a connector. Raises ValueError for unknown connectors."""
    if connector not in CONNECTOR_TOOLS:
        raise ValueError(f"Unknown connector: {connector!r}")

    conn = connect()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO connector_permissions (connector, enabled, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (connector) DO UPDATE
                    SET enabled    = EXCLUDED.enabled,
                        updated_at = EXCLUDED.updated_at
                """,
                (connector, enabled),
            )
    conn.close()


def list_all() -> list[dict]:
    """
    Return all connectors with their current permission state.
    Each entry: {connector, label, enabled, read_tools, write_tools}.
    """
    try:
        conn = connect()
        with conn.cursor() as cur:
            cur.execute("SELECT connector, enabled FROM connector_permissions")
            rows = {r[0]: r[1] for r in cur.fetchall()}
        conn.close()
    except Exception:
        rows = {}

    result = []
    for connector, tools in CONNECTOR_TOOLS.items():
        result.append({
            "connector":   connector,
            "label":       CONNECTOR_LABELS.get(connector, connector),
            "enabled":     rows.get(connector, True),
            "read_tools":  [t for t in tools if t not in WRITE_TOOLS],
            "write_tools": [t for t in tools if t in WRITE_TOOLS],
        })
    return result
