"""
Tool router — a cheap Flash-Lite call classifies the user's intent,
then narrows the tool list before the main Gemini call.

Adding a new connector: add an entry to _CATEGORY_TOOLS with the
new category name and its tool names. The rest is automatic.
"""

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

ROUTER_MODEL = os.environ.get("GEMINI_ROUTER_MODEL", "gemini-2.0-flash-lite")

# Map intent category → tool names. Extend as connectors grow.
_CATEGORY_TOOLS: dict[str, list[str]] = {
    "memory":   ["memory_remember", "memory_list", "memory_forget", "memory_graph_query", "usage_summary"],
    "calendar": ["get_calendar_events", "create_event"],
    "email":    ["get_recent_emails", "search_emails", "create_draft", "send_email"],
    "drive":    ["list_drive_files", "read_drive_file"],
    "tasks":    ["get_tasks", "get_projects", "add_task", "complete_task", "update_task"],
    "home":     ["get_devices", "get_device_state", "control_device", "set_light"],
    "music":    ["spotify_now_playing", "spotify_get_devices", "spotify_control", "spotify_set_volume", "spotify_search_play"],
    "system":   ["get_system_info", "get_wifi_info", "show_notification", "open_application", "set_system_volume", "capture_screen", "computer_use"],
    "search":   ["web_search"],
    "messages": ["imessage_get_messages", "imessage_send"],
    "files":    ["list_directory", "read_local_file", "write_local_file"],
    "browser":   ["browser_get_active_tab", "browser_list_tabs", "browser_open_url"],
    "contacts":  ["contacts_search"],
    "reminders": ["reminders_list", "reminders_add", "reminders_complete"],
    "notes":     ["notes_list", "notes_read", "notes_create", "notes_append"],
    "clipboard": ["clipboard_read", "clipboard_write"],
    "telegram":  ["telegram_get_messages", "telegram_send"],
    "notion":    ["notion_search", "notion_read_page", "notion_create_page", "notion_append_to_page"],
    "github":    ["github_list_repos", "github_list_prs", "github_list_issues", "github_get_ci_status", "github_create_issue"],
    "slack":     ["slack_list_channels", "slack_read_messages", "slack_send_message"],
    "health":    ["health_get_sleep", "health_get_activity", "health_get_readiness"],
    "finance":   ["finance_get_accounts", "finance_get_transactions", "finance_get_spending_summary"],
    "car":       ["car_get_status", "car_lock", "car_climate"],
    "appliances":["appliances_list", "appliances_get_status", "appliances_control"],
    "ai":        ["ai_ask", "ai_compare"],
    "orchestrator": ["task_plan", "task_status", "task_list", "task_cancel"],
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
    "  any new Telegram messages → [\"telegram\"]\n"
    "  send a Telegram to John → [\"telegram\"]\n"
    "  message my Telegram bot → [\"telegram\"]\n"
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
    "  how much have I spent on the API today → [\"memory\"]\n"
    "  what's my API cost this week → [\"memory\"]\n"
    "  which tools are slowest → [\"memory\"]\n"
    "  show my usage summary → [\"memory\"]\n"
    "  remember that I prefer dark roast coffee → [\"memory\"]\n"
    "  remember my gym schedule is Monday Wednesday Friday → [\"memory\"]\n"
    "  what do you remember about me → [\"memory\"]\n"
    "  forget that I said I was vegetarian → [\"memory\"]\n"
    "  what facts have you saved → [\"memory\"]\n"
    "  what do you know about Alice → [\"memory\"]\n"
    "  how is Alice related to the project → [\"memory\"]\n"
    "  show me the knowledge graph for CollectiveOS → [\"memory\"]\n"
    "  what connections do you see between Bob and the gym → [\"memory\"]\n"
    "  what's my bank balance → [\"finance\"]\n"
    "  show my recent transactions → [\"finance\"]\n"
    "  how much did I spend on food this month → [\"finance\"]\n"
    "  what's my account balance → [\"finance\"]\n"
    "  is my car locked → [\"car\"]\n"
    "  lock the car → [\"car\"]\n"
    "  turn on the car climate → [\"car\"]\n"
    "  what's the charge level of my car → [\"car\"]\n"
    "  list my smart appliances → [\"appliances\"]\n"
    "  turn off the washing machine → [\"appliances\"]\n"
    "  what's the status of my dryer → [\"appliances\"]\n"
    "  ask ChatGPT to explain quantum computing → [\"ai\"]\n"
    "  what does Grok think about climate change → [\"ai\"]\n"
    "  compare what GPT and Gemini say about Python → [\"ai\"]\n"
    "  ask all AIs about the meaning of life → [\"ai\"]\n"
    "  get Gemini to write a haiku → [\"ai\"]\n"
    "  check my Telegram messages → [\"telegram\"]\n"
    "  send a Telegram message to John → [\"telegram\"]\n"
    "  run a multi-step task for me → [\"orchestrator\"]\n"
    "  research X and save it to Notion → [\"orchestrator\", \"search\", \"notion\"]\n"
    "  check my tasks and create follow-ups → [\"orchestrator\", \"tasks\"]\n"
    "  what tasks have you run recently → [\"orchestrator\"]\n"
    "  cancel that task → [\"orchestrator\"]\n"
    "  what's the status of task 5 → [\"orchestrator\"]\n"
    "  click the submit button for me → [\"system\"]\n"
    "  automate filling out this form → [\"system\"]\n"
    "  control my desktop to open that file → [\"system\"]\n"
    "  use the computer to complete this task → [\"system\"]\n"
)

_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=ROUTER_MODEL,
            google_api_key=os.environ["GEMINI_API_KEY"],
            temperature=0,
        )
    return _llm


def select_tools(user_message: str, all_tools: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Classify *user_message* with a cheap Flash-Lite call and return (filtered_tools, categories).

    Falls back to (all_tools, []) if classification fails or returns no match,
    so the main model call always has at least one tool available.
    """
    try:
        response = _get_llm().invoke([
            SystemMessage(content=_SYSTEM + "\n\n" + _EXAMPLES),
            HumanMessage(content=user_message),
        ])
        try:
            from src import observability as _obs
            usage = response.usage_metadata or {}
            _obs.log_api_call(
                ROUTER_MODEL,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                source="router",
            )
        except Exception:
            pass
        raw = (response.content or "").strip()
        # Strip markdown fences if the model added them
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        categories: list[str] = json.loads(raw)
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
