"""
Tool router — a cheap Haiku call classifies the user's intent,
then narrows the tool list before the main Sonnet call.

Adding a new connector: add an entry to _CATEGORY_TOOLS with the
new category name and its tool names. The rest is automatic.
"""

import json
from anthropic import Anthropic

ROUTER_MODEL = "claude-haiku-4-5-20251001"

# Map intent category → tool names. Extend as connectors grow.
_CATEGORY_TOOLS: dict[str, list[str]] = {
    "calendar": ["get_calendar_events", "create_event"],
    "email":    ["get_recent_emails", "search_emails", "create_draft", "send_email"],
    "drive":    ["list_drive_files", "read_drive_file"],
    "tasks":    ["get_tasks", "get_projects", "add_task", "complete_task", "update_task"],
    "home":     ["get_devices", "get_device_state", "control_device", "set_light"],
    "music":    ["spotify_now_playing", "spotify_get_devices", "spotify_control", "spotify_set_volume", "spotify_search_play"],
    "system":   ["get_system_info", "get_wifi_info", "show_notification", "open_application", "set_system_volume"],
    "search":   ["web_search"],
    "messages": ["imessage_get_messages", "imessage_send"],
}

_CATEGORIES_STR = ", ".join(_CATEGORY_TOOLS)

_SYSTEM = (
    "You are a routing classifier for a personal AI assistant. "
    "Given a user message, return a JSON array of the relevant categories from this list: "
    f"{_CATEGORIES_STR}. "
    "Return ONLY a valid JSON array — no explanation, no markdown. "
    "Multiple categories are allowed. "
    "If no category fits, return []."
)

_EXAMPLES = (
    "Examples:\n"
    "  what's on my calendar this week → [\"calendar\"]\n"
    "  any emails from Alice? → [\"email\"]\n"
    "  show my tasks and upcoming events → [\"tasks\", \"calendar\"]\n"
    "  turn off the living room lights → [\"home\"]\n"
    "  find the budget doc in Drive → [\"drive\"]\n"
    "  what's the weather today → [\"search\"]\n"
    "  latest news about AI → [\"search\"]\n"
    "  what is the current price of Bitcoin → [\"search\"]\n"
    "  how do I fix a Python import error → [\"search\"]\n"
    "  what movies are playing this weekend → [\"search\"]\n"
    "  schedule a meeting and search for the venue address → [\"calendar\", \"search\"]\n"
    "  did Alice text me? → [\"messages\"]\n"
    "  show my recent texts → [\"messages\"]\n"
    "  send a message to mom → [\"messages\"]\n"
    "  text John that I'm running late → [\"messages\"]\n"
)

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def select_tools(user_message: str, all_tools: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Classify *user_message* with Haiku and return (filtered_tools, categories).

    Falls back to (all_tools, []) if classification fails or returns no match,
    so the main Sonnet call always has at least one tool available.
    """
    try:
        resp = _get_client().messages.create(
            model=ROUTER_MODEL,
            max_tokens=64,
            system=_SYSTEM + "\n\n" + _EXAMPLES,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = resp.content[0].text.strip()
        categories: list[str] = json.loads(raw)
        # Ignore any category names not in our map
        categories = [c for c in categories if c in _CATEGORY_TOOLS]
    except Exception:
        return all_tools, []

    if not categories:
        return all_tools, []

    allowed: set[str] = set()
    for cat in categories:
        allowed.update(_CATEGORY_TOOLS[cat])

    filtered = [t for t in all_tools if t["name"] in allowed]
    return (filtered if filtered else all_tools), categories
