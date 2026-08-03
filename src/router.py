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
    "system":   ["get_system_info", "get_wifi_info", "show_notification", "open_application", "set_system_volume", "capture_screen"],
    "search":   ["web_search"],
    "messages": ["imessage_get_messages", "imessage_send"],
    "files":    ["list_directory", "read_local_file", "write_local_file"],
    "browser":   ["browser_get_active_tab", "browser_list_tabs", "browser_open_url"],
    "contacts":  ["contacts_search"],
    "reminders": ["reminders_list", "reminders_add", "reminders_complete"],
    "notes":     ["notes_list", "notes_read", "notes_create", "notes_append"],
    "clipboard": ["clipboard_read", "clipboard_write"],
    "notion":    ["notion_search", "notion_read_page", "notion_create_page", "notion_append_to_page"],
    "github":    ["github_list_repos", "github_list_prs", "github_list_issues", "github_get_ci_status", "github_create_issue"],
    "slack":     ["slack_list_channels", "slack_read_messages", "slack_send_message"],
    "health":    ["health_get_sleep", "health_get_activity", "health_get_readiness"],
    "finance":   ["finance_get_accounts", "finance_get_transactions", "finance_get_spending_summary"],
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
    "  what's on my screen right now → [\"system\"]\n"
    "  what does this error say → [\"system\"]\n"
    "  can you see what app I have open → [\"system\"]\n"
    "  what's in my Downloads folder → [\"files\"]\n"
    "  read the file ~/Documents/notes.txt → [\"files\"]\n"
    "  save this to a file on my Desktop → [\"files\"]\n"
    "  list my project files → [\"files\"]\n"
    "  what page am I on → [\"browser\"]\n"
    "  what tabs do I have open → [\"browser\"]\n"
    "  open this URL in my browser → [\"browser\"]\n"
    "  show me the GitHub page for this repo → [\"browser\"]\n"
    "  what's John's phone number → [\"contacts\"]\n"
    "  find Alice's email → [\"contacts\"]\n"
    "  what are my reminders → [\"reminders\"]\n"
    "  remind me to call the dentist tomorrow → [\"reminders\"]\n"
    "  what notes do I have → [\"notes\"]\n"
    "  read my shopping list note → [\"notes\"]\n"
    "  add milk to my shopping list note → [\"notes\"]\n"
    "  what's in my clipboard → [\"clipboard\"]\n"
    "  copy this to clipboard → [\"clipboard\"]\n"
    "  search my Notion for meeting notes → [\"notion\"]\n"
    "  what does my project plan page say → [\"notion\"]\n"
    "  create a new Notion page for my ideas → [\"notion\"]\n"
    "  add this to my daily notes in Notion → [\"notion\"]\n"
    "  show my GitHub repos → [\"github\"]\n"
    "  any open PRs on CollectiveOS → [\"github\"]\n"
    "  what issues are open in my repo → [\"github\"]\n"
    "  did CI pass on main → [\"github\"]\n"
    "  create a GitHub issue for this bug → [\"github\"]\n"
    "  what channels do I have in Slack → [\"slack\"]\n"
    "  show recent messages in #general → [\"slack\"]\n"
    "  what did the team say in Slack today → [\"slack\"]\n"
    "  send a Slack message to the engineering channel → [\"slack\"]\n"
    "  DM John on Slack → [\"slack\"]\n"
    "  how did I sleep last night → [\"health\"]\n"
    "  what's my HRV this week → [\"health\"]\n"
    "  show my step count for the last 7 days → [\"health\"]\n"
    "  what's my readiness score today → [\"health\"]\n"
    "  how is my recovery looking → [\"health\"]\n"
    "  what's my bank balance → [\"finance\"]\n"
    "  show my recent transactions → [\"finance\"]\n"
    "  how much did I spend this month → [\"finance\"]\n"
    "  what did I spend on food last week → [\"finance\"]\n"
    "  break down my spending by category → [\"finance\"]\n"
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
