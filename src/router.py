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

# Map intent category → tool names.
# External service connectors (Gmail, Calendar, Slack, etc.) are removed;
# those tasks now go through navigate_computer.
_CATEGORY_TOOLS: dict[str, list[str]] = {
    "memory":      ["memory_remember", "memory_list", "memory_forget", "memory_graph_query", "usage_summary"],
    "home":        ["get_devices", "get_device_state", "control_device", "set_light"],
    "music":       ["spotify_now_playing", "spotify_get_devices", "spotify_control", "spotify_set_volume", "spotify_search_play"],
    "system":      ["get_system_info", "get_wifi_info", "show_notification", "notify_user", "open_application", "set_system_volume"],
    "files":       ["list_directory", "read_local_file", "write_local_file"],
    "health":      ["health_get_sleep", "health_get_activity", "health_get_readiness"],
    "car":         ["car_get_status", "car_lock", "car_climate"],
    "appliances":  ["appliances_list", "appliances_get_status", "appliances_control"],
    "orchestrator":["task_plan", "task_status", "task_list", "task_cancel"],
    "wearable":    ["wearable_get_events", "wearable_list_devices"],
    "robot":       ["robot_status", "robot_move", "robot_cancel"],
    "computer":    ["navigate_computer"],
}

_CATEGORIES_STR = ", ".join(_CATEGORY_TOOLS)

_SYSTEM = (
    "You are a routing classifier for a personal AI assistant. "
    "Given a user message, return a JSON array of the relevant categories from this list: "
    f"{_CATEGORIES_STR}. "
    "Return ONLY a valid JSON array — no explanation, no markdown. "
    "Multiple categories are allowed. "
    "For any task involving an app, website, email client, calendar app, messaging app, "
    "or any screen interaction, return [\"computer\"]. "
    "If no category fits, return []."
)

_EXAMPLES = (
    "Examples:\n"
    "  turn off the living room lights → [\"home\"]\n"
    "  dim the bedroom lights to 40% → [\"home\"]\n"
    "  what devices are online → [\"home\"]\n"
    "  play some jazz on Spotify → [\"music\"]\n"
    "  pause the music → [\"music\"]\n"
    "  set volume to 60 → [\"music\"]\n"
    "  what song is playing → [\"music\"]\n"
    "  what's the system info → [\"system\"]\n"
    "  what wifi am I on → [\"system\"]\n"
    "  send me a notification → [\"system\"]\n"
    "  open Finder → [\"system\"]\n"
    "  set volume to 50 → [\"system\"]\n"
    "  what's in my Downloads folder → [\"files\"]\n"
    "  read the file ~/Documents/notes.txt → [\"files\"]\n"
    "  save this to a file on my Desktop → [\"files\"]\n"
    "  how did I sleep last night → [\"health\"]\n"
    "  what's my HRV this week → [\"health\"]\n"
    "  show my step count → [\"health\"]\n"
    "  what's my readiness score → [\"health\"]\n"
    "  is my car locked → [\"car\"]\n"
    "  lock the car → [\"car\"]\n"
    "  turn on the car climate → [\"car\"]\n"
    "  list my smart appliances → [\"appliances\"]\n"
    "  turn off the washing machine → [\"appliances\"]\n"
    "  remember that I prefer dark roast coffee → [\"memory\"]\n"
    "  what do you remember about me → [\"memory\"]\n"
    "  forget that I said I was vegetarian → [\"memory\"]\n"
    "  how much have I spent on the API today → [\"memory\"]\n"
    "  what connections exist between Alice and the gym → [\"memory\"]\n"
    "  run a multi-step task for me → [\"orchestrator\"]\n"
    "  cancel that task → [\"orchestrator\"]\n"
    "  what tasks have you run recently → [\"orchestrator\"]\n"
    "  what's the status of task 5 → [\"orchestrator\"]\n"
    "  what did my wearable detect → [\"wearable\"]\n"
    "  show me my Garmin events → [\"wearable\"]\n"
    "  what is the robot doing → [\"robot\"]\n"
    "  move the robot forward 2 metres → [\"robot\"]\n"
    "  stop the robot → [\"robot\"]\n"
    "  click the submit button for me → [\"computer\"]\n"
    "  open Gmail and send an email to John → [\"computer\"]\n"
    "  check my calendar for today → [\"computer\"]\n"
    "  send a Slack message to the engineering channel → [\"computer\"]\n"
    "  go to github.com and check my latest PR → [\"computer\"]\n"
    "  open Notion and update my weekly plan page → [\"computer\"]\n"
    "  search the web for the best Python libraries → [\"computer\"]\n"
    "  what's the weather today → [\"computer\"]\n"
    "  open Terminal and run git status → [\"computer\"]\n"
    "  book that restaurant on OpenTable → [\"computer\"]\n"
    "  navigate to Slack and post a message in #general → [\"computer\"]\n"
    "  open Finder and move the file to Downloads → [\"computer\"]\n"
    "  did Alice text me → [\"computer\"]\n"
    "  reply to that email from Bob → [\"computer\"]\n"
    "  show my tasks in Todoist → [\"computer\"]\n"
    "  find the budget doc in Drive → [\"computer\"]\n"
    "  what issues are open in my GitHub repo → [\"computer\"]\n"
    "  any new Telegram messages → [\"computer\"]\n"
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
