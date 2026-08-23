"""
assistant_starter.py
---------------------
Personal-assistant agent loop powered by Google Gemini with tool use.

Pattern:
    retrieve relevant memory
    -> build system prompt with context
    -> user asks -> model calls tools -> your code runs them -> loop
    -> model gives final answer
    -> save exchange to memory

SETUP (do this once):
    1. pip install -r requirements.txt
    2. export GEMINI_API_KEY="AIza..."
    3. python src/assistant_starter.py
"""

import datetime
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


from collectiveos.connectors.google_calendar import get_calendar_events, create_event
from collectiveos.connectors.gmail import get_recent_emails, search_emails, create_draft, send_email
from collectiveos.connectors.google_drive import list_files, read_file
from collectiveos.connectors.todoist import get_tasks, get_projects, add_task, complete_task, update_task
from collectiveos.connectors.home_assistant import get_devices, get_device_state, control_device
from collectiveos.connectors import spotify as _spotify
from collectiveos.connectors import mac_system as _mac
from collectiveos.connectors.web_search import search as web_search
from collectiveos.connectors import imessage as _imessage
from collectiveos.connectors.screen_capture import capture_screen
from collectiveos.connectors import local_files as _fs
from collectiveos.connectors import browser as _browser
from collectiveos.connectors import apple_native as _apple
from collectiveos.connectors import telegram_bot as _telegram
from collectiveos.connectors import notion as _notion
from collectiveos.connectors import github as _github
from collectiveos.connectors import slack_bot as _slack
from collectiveos.connectors import health as _health
from collectiveos.connectors import finance as _finance
from collectiveos.connectors import car as _car
from collectiveos.connectors import appliances as _appliances
from collectiveos.connectors import ai_models as _ai
from collectiveos import memory, graph_memory, router, permissions, observability as _obs

MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-flash-latest")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def memory_remember(fact: str) -> str:
    """Save an explicit fact to long-term memory."""
    memory.save_fact(fact)
    return f"Got it — I've saved: {fact}"


def memory_list() -> str:
    """Return all explicitly saved facts."""
    facts = memory.list_facts()
    if not facts:
        return "No facts saved yet."
    return "\n".join(f"[{f['date']}] {f['content']}" for f in facts)


def memory_forget(fact: str) -> str:
    """Remove the saved fact most closely matching the given description."""
    deleted = memory.delete_fact(fact)
    if not deleted:
        return "No matching fact found."
    return f"Forgotten: {deleted}"


def memory_graph_query(entity: str) -> str:
    """Look up an entity in the knowledge graph and return its relationships."""
    return graph_memory.query_graph(entity)


def usage_summary(days: int = 1) -> str:
    """Return API cost and tool latency report for the last N days."""
    return _obs.usage_summary(days)


def set_light(room: str, state: str) -> str:
    """Placeholder — replace body with a real Home Assistant call later."""
    return f"OK, the {room} light is now {state} (pretend action)."


TOOL_FUNCTIONS = {
    "memory_remember":    memory_remember,
    "memory_list":        memory_list,
    "memory_forget":      memory_forget,
    "memory_graph_query": memory_graph_query,
    "usage_summary":      usage_summary,
    "ai_ask":             _ai.ai_ask,
    "ai_compare":         _ai.ai_compare,
    "get_calendar_events": get_calendar_events,
    "create_event":        create_event,
    "get_recent_emails":   get_recent_emails,
    "search_emails":       search_emails,
    "create_draft":        create_draft,
    "send_email":          send_email,
    "list_drive_files":    list_files,
    "read_drive_file":     read_file,
    "get_tasks":           get_tasks,
    "get_projects":        get_projects,
    "add_task":            add_task,
    "complete_task":       complete_task,
    "update_task":         update_task,
    "get_devices":         get_devices,
    "get_device_state":    get_device_state,
    "control_device":      control_device,
    "set_light":           set_light,
    "spotify_now_playing":  _spotify.get_now_playing,
    "spotify_get_devices":  _spotify.get_devices,
    "spotify_control":      _spotify.control_playback,
    "spotify_set_volume":   _spotify.set_volume,
    "spotify_search_play":  _spotify.search_and_play,
    "get_system_info":      _mac.get_system_info,
    "get_wifi_info":        _mac.get_wifi_info,
    "show_notification":    _mac.show_notification,
    "open_application":     _mac.open_application,
    "set_system_volume":    _mac.set_system_volume,
    "web_search":           web_search,
    "imessage_get_messages": _imessage.get_messages,
    "imessage_send":         _imessage.send_message,
    "capture_screen":        capture_screen,
    "list_directory":        _fs.list_directory,
    "read_local_file":       _fs.read_local_file,
    "write_local_file":      _fs.write_local_file,
    "browser_get_active_tab": _browser.get_active_tab,
    "browser_list_tabs":      _browser.list_tabs,
    "browser_open_url":       _browser.open_url,
    "contacts_search":        _apple.contacts_search,
    "reminders_list":         _apple.reminders_list,
    "reminders_add":          _apple.reminders_add,
    "reminders_complete":     _apple.reminders_complete,
    "notes_list":             _apple.notes_list,
    "notes_read":             _apple.notes_read,
    "notes_create":           _apple.notes_create,
    "notes_append":           _apple.notes_append,
    "clipboard_read":         _apple.clipboard_read,
    "clipboard_write":        _apple.clipboard_write,
    "telegram_get_messages":  _telegram.get_messages,
    "telegram_send":          _telegram.send_message,
    "notion_search":          _notion.notion_search,
    "notion_read_page":       _notion.notion_read_page,
    "notion_create_page":     _notion.notion_create_page,
    "notion_append_to_page":  _notion.notion_append_to_page,
    "github_list_repos":      _github.github_list_repos,
    "github_list_prs":        _github.github_list_prs,
    "github_list_issues":     _github.github_list_issues,
    "github_get_ci_status":   _github.github_get_ci_status,
    "github_create_issue":    _github.github_create_issue,
    "slack_list_channels":    _slack.slack_list_channels,
    "slack_read_messages":    _slack.slack_read_messages,
    "slack_send_message":     _slack.slack_send_message,
    "health_get_sleep":       _health.health_get_sleep,
    "health_get_activity":    _health.health_get_activity,
    "health_get_readiness":   _health.health_get_readiness,
    "finance_get_accounts":          _finance.finance_get_accounts,
    "finance_get_transactions":      _finance.finance_get_transactions,
    "finance_get_spending_summary":  _finance.finance_get_spending_summary,
    "car_get_status":   _car.car_get_status,
    "car_lock":         _car.car_lock,
    "car_climate":      _car.car_climate,
    "appliances_list":       _appliances.appliances_list,
    "appliances_get_status": _appliances.appliances_get_status,
    "appliances_control":    _appliances.appliances_control,
    "telegram_get_messages": _telegram.get_messages,
    "telegram_send":         _telegram.send_message,
}

TOOLS = [
    # -----------------------------------------------------------------------
    # Long-term memory
    # -----------------------------------------------------------------------
    {
        "name": "memory_remember",
        "description": (
            "Save a fact, preference, or personal detail to long-term memory so it "
            "is always available in future conversations. "
            "Use when the user says 'remember that…', 'keep in mind that…', "
            "'my X is Y', or gives you information they expect you to retain. "
            "Always confirm what you saved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "A concise statement of the fact to remember, e.g. "
                                   "'User prefers dark roast coffee' or "
                                   "'User's partner is named Alex'.",
                },
            },
            "required": ["fact"],
        },
    },
    {
        "name": "memory_list",
        "description": "List all facts that have been explicitly saved to long-term memory.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "memory_forget",
        "description": (
            "Remove a previously saved fact from long-term memory. "
            "Use when the user says 'forget that…' or 'that's no longer true'. "
            "Finds the closest matching saved fact and deletes it. Always confirm what was removed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "Description of the fact to forget — the closest match will be deleted.",
                },
            },
            "required": ["fact"],
        },
    },
    {
        "name": "memory_graph_query",
        "description": (
            "Query the knowledge graph for a person, project, place, or concept the assistant "
            "has encountered in past conversations. Returns the entity's type, all known "
            "relationships to other entities, and the conversations where it appeared. "
            "Use when the user asks about connections between people or topics, "
            "e.g. 'what do you know about Alice', 'how is Bob related to the project', "
            "'show me everything about CollectiveOS'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Name of the entity to look up, e.g. 'Alice', 'CollectiveOS', 'gym'.",
                },
            },
            "required": ["entity"],
        },
    },
    {
        "name": "usage_summary",
        "description": (
            "Return a summary of API token usage and estimated cost for the last N days, "
            "plus latency and success rate for the most-called tools. "
            "Use when the user asks 'how much have I spent today?', 'what's my API cost this week?', "
            "'which tools are slowest?', or any question about assistant usage or cost."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back. Default is 1 (today).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "ai_ask",
        "description": (
            "Send a prompt to a specific external AI model and return its response. "
            "Use when the user wants to ask ChatGPT, Grok, Gemini, or Claude a specific question, "
            "e.g. 'ask ChatGPT to explain quantum computing', 'what does Grok think about X', "
            "'get Gemini to write a poem'. model must be one of: chatgpt, claude, grok, gemini."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "AI model to query. One of: chatgpt, claude, grok, gemini.",
                },
                "prompt": {
                    "type": "string",
                    "description": "The question or instruction to send to the model.",
                },
                "system": {
                    "type": "string",
                    "description": "Optional system prompt / persona to set for the model.",
                },
            },
            "required": ["model", "prompt"],
        },
    },
    {
        "name": "ai_compare",
        "description": (
            "Send the same prompt to all configured AI models (ChatGPT, Claude, Grok, Gemini) "
            "simultaneously and return a side-by-side comparison of their responses. "
            "Use when the user says 'compare AI opinions on X', 'what do different AIs think about Y', "
            "'ask all models about Z'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The question or task to send to all models.",
                },
                "system": {
                    "type": "string",
                    "description": "Optional system prompt to apply to all models.",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "get_calendar_events",
        "description": "Get upcoming events from the user's primary Google Calendar.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days_ahead": {
                    "type": "integer",
                    "description": "How many days ahead to look. Defaults to 7.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "create_event",
        "description": (
            "Create a new event on the user's primary Google Calendar. "
            "Always confirm the details (title, date, time, duration) with the user before calling. "
            "Use ISO 8601 for datetimes, e.g. '2024-07-10T14:00:00'. "
            "Ask the user for their timezone if not already known; default to UTC."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Event title / summary.",
                },
                "start_datetime": {
                    "type": "string",
                    "description": "Start time in ISO 8601 format, e.g. '2024-07-10T14:00:00'.",
                },
                "end_datetime": {
                    "type": "string",
                    "description": "End time in ISO 8601 format, e.g. '2024-07-10T15:00:00'.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional event description / notes.",
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. 'America/New_York'. Defaults to UTC.",
                },
            },
            "required": ["title", "start_datetime", "end_datetime"],
        },
    },
    {
        "name": "get_recent_emails",
        "description": "Fetch the most recent emails from the user's Gmail inbox.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Number of emails to return. Defaults to 10.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "search_emails",
        "description": (
            "Search the user's Gmail using a query string. "
            "Supports Gmail search syntax, e.g. 'from:alice', 'subject:invoice', "
            "'is:unread', 'after:2024/01/01'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Gmail search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return. Defaults to 5.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_draft",
        "description": (
            "Create a Gmail draft. Does NOT send — the user reviews and sends it from Gmail. "
            "Always confirm recipient, subject, and body with the user before calling. "
            "Prefer this over send_email unless the user explicitly says 'send now'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address.",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject. Leave blank when reply_to_id is set to auto-fill 'Re: ...'.",
                },
                "body": {
                    "type": "string",
                    "description": "Plain-text email body.",
                },
                "reply_to_id": {
                    "type": "string",
                    "description": "Optional message id to reply to (from get_recent_emails). Keeps the email in the same thread.",
                },
            },
            "required": ["to", "body"],
        },
    },
    {
        "name": "send_email",
        "description": (
            "Send an email immediately via Gmail. "
            "Only call this after the user has explicitly confirmed the full content — "
            "recipient, subject, and body. For safety, use create_draft first so the user "
            "can review before sending."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address.",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject.",
                },
                "body": {
                    "type": "string",
                    "description": "Plain-text email body.",
                },
                "reply_to_id": {
                    "type": "string",
                    "description": "Optional message id to reply to (keeps thread).",
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "list_drive_files",
        "description": (
            "List files in the user's Google Drive. "
            "Optionally filter with a Drive query, e.g. \"name contains 'budget'\" "
            "or \"mimeType='application/vnd.google-apps.document'\"."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Max number of files to return. Defaults to 20.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional Drive query string to filter results.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_drive_file",
        "description": (
            "Read the text content of a Google Drive file by its ID. "
            "Works for Google Docs, Sheets, Slides, and plain text files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The Drive file ID (from list_drive_files).",
                }
            },
            "required": ["file_id"],
        },
    },
    {
        "name": "get_tasks",
        "description": (
            "List active Todoist tasks. Optionally filter by project name or "
            "a Todoist filter expression (e.g. 'today', 'overdue', 'p1', '7 days')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Filter to tasks in this project (partial match).",
                },
                "filter_str": {
                    "type": "string",
                    "description": "Todoist filter expression, e.g. 'today' or 'overdue'.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_projects",
        "description": "List all Todoist projects.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "add_task",
        "description": (
            "Add a new task to Todoist. Always confirm with the user before calling this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Task title.",
                },
                "due_string": {
                    "type": "string",
                    "description": "Natural-language due date, e.g. 'tomorrow', 'next Monday'.",
                },
                "project_name": {
                    "type": "string",
                    "description": "Destination project name. Uses Inbox if omitted.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "complete_task",
        "description": (
            "Mark a Todoist task as completed. "
            "Always confirm with the user before calling. "
            "Use get_tasks first if you don't have the task id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task id shown in brackets by get_tasks.",
                },
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "update_task",
        "description": (
            "Update an existing Todoist task — rename it or change its due date. "
            "Confirm the change with the user before calling. "
            "Use get_tasks first if you don't have the task id. "
            "Pass due_string='none' to remove the due date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task id from get_tasks.",
                },
                "content": {
                    "type": "string",
                    "description": "New task title. Omit to keep current.",
                },
                "due_string": {
                    "type": "string",
                    "description": "New due date in natural language, e.g. 'Friday'. Pass 'none' to clear.",
                },
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "get_devices",
        "description": (
            "List Home Assistant entities and their current states. "
            "Optionally filter by domain: 'light', 'switch', 'sensor', "
            "'binary_sensor', 'climate', 'media_player', etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Entity domain to filter by. Leave blank for all.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_device_state",
        "description": "Get the full state and attributes of a single Home Assistant entity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "Entity ID, e.g. 'light.living_room'.",
                }
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "control_device",
        "description": (
            "Turn a Home Assistant entity on or off. "
            "Always confirm with the user before calling. "
            "Heating appliances (microwave, cooktop, washer, oven) may only be turned OFF."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "Entity ID, e.g. 'switch.fan' or 'light.kitchen'.",
                },
                "action": {
                    "type": "string",
                    "enum": ["turn_on", "turn_off"],
                    "description": "Action to perform.",
                },
            },
            "required": ["entity_id", "action"],
        },
    },
    {
        "name": "set_light",
        "description": "Turn a light on or off in a specific room.",
        "input_schema": {
            "type": "object",
            "properties": {
                "room":  {"type": "string", "description": "e.g. 'kitchen'."},
                "state": {
                    "type": "string",
                    "enum": ["on", "off"],
                    "description": "Whether to turn the light on or off.",
                },
            },
            "required": ["room", "state"],
        },
    },
    {
        "name": "spotify_now_playing",
        "description": "Get the currently playing track on Spotify, including artist, album, position, and active device.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "spotify_get_devices",
        "description": "List all active Spotify Connect devices (phone, laptop, speaker, car, etc.) with their ids and volume.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "spotify_control",
        "description": "Control Spotify playback — play, pause, skip to next track, or go to previous track.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["play", "pause", "next", "previous"],
                    "description": "Playback action to perform.",
                },
                "device_id": {
                    "type": "string",
                    "description": "Optional Spotify device id from spotify_get_devices. Defaults to active device.",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "spotify_set_volume",
        "description": "Set the Spotify playback volume on the active (or specified) device.",
        "input_schema": {
            "type": "object",
            "properties": {
                "volume_percent": {
                    "type": "integer",
                    "description": "Volume level 0–100.",
                },
                "device_id": {
                    "type": "string",
                    "description": "Optional Spotify device id. Defaults to active device.",
                },
            },
            "required": ["volume_percent"],
        },
    },
    {
        "name": "spotify_search_play",
        "description": (
            "Search Spotify for a track, artist, album, or playlist and immediately play the top result. "
            "Examples: 'Bohemian Rhapsody', 'The Beatles', 'Chill focus playlist'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term — song name, artist, album, or playlist description.",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["track", "artist", "album", "playlist"],
                    "description": "What to search for. Defaults to track.",
                },
                "device_id": {
                    "type": "string",
                    "description": "Optional Spotify device id. Defaults to active device.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_system_info",
        "description": (
            "Get a snapshot of this Mac's current status: battery level and "
            "charging state, disk usage, free memory, CPU model, macOS version, and uptime."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_wifi_info",
        "description": "Get the current Wi-Fi network name (SSID) and local IP address.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "show_notification",
        "description": (
            "Show a macOS notification banner on this Mac. "
            "Useful for reminders, alerts, or confirming a completed action."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Notification title.",
                },
                "body": {
                    "type": "string",
                    "description": "Notification message text.",
                },
            },
            "required": ["title", "body"],
        },
    },
    {
        "name": "open_application",
        "description": (
            "Open a macOS application by name, e.g. 'Safari', 'Spotify', 'VS Code', 'Calendar'. "
            "Always confirm with the user before opening apps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Application name as it appears in /Applications.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "set_system_volume",
        "description": (
            "Set the macOS system audio output volume (0–100). "
            "This controls the Mac's speaker/headphone volume, "
            "independent of Spotify's own volume control."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "integer",
                    "description": "Volume level 0 (silent) to 100 (maximum).",
                },
            },
            "required": ["level"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for current, real-time information — news, weather, prices, "
            "sports scores, documentation, local businesses, travel info, or anything "
            "that may have changed since the model's training cutoff. "
            "Use this whenever the user asks about something recent, live, or factual "
            "that you cannot answer from memory alone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. Be specific for better results, e.g. "
                                   "'weather Toronto today' not just 'weather'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of sources to return (1–10). Defaults to 5.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "imessage_get_messages",
        "description": (
            "Read recent iMessages from this Mac's Messages history. "
            "Optionally filter to a single conversation by phone number or Apple ID. "
            "Requires Full Disk Access granted to the terminal or server process."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "contact": {
                    "type": "string",
                    "description": "Phone number (+1…) or Apple ID email to filter to "
                                   "one thread. Leave blank for recent messages across all chats.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of messages to return. Defaults to 10.",
                },
                "days": {
                    "type": "integer",
                    "description": "How many days back to look. Defaults to 3.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "imessage_send",
        "description": (
            "Send an iMessage to a phone number (+1…) or Apple ID email. "
            "Always confirm the recipient and the full message text with the user "
            "before calling. Messages.app will prompt for Automation permission on first use."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient phone number (e.g. '+14155552671') or Apple ID email.",
                },
                "message": {
                    "type": "string",
                    "description": "Message text to send.",
                },
            },
            "required": ["to", "message"],
        },
    },
    {
        "name": "capture_screen",
        "description": (
            "Take a screenshot of the current Mac screen and analyse it using "
            "Gemini's vision. Use this when the user asks about something on their "
            "screen — an error message, the current app state, a piece of UI, code "
            "they are looking at, or any visible text. "
            "The screenshot is sent to Google's API for analysis and then deleted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Specific question to answer about the screen, e.g. "
                        "'What does this error say?' or 'Which tab is active?'. "
                        "Leave blank for a general description of what's visible."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_directory",
        "description": (
            "List the contents of a directory on this Mac. "
            "Accepts ~ (home), relative paths, or absolute paths inside the home directory. "
            "Shows file names, sizes, and last-modified dates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list. Defaults to ~ (home). "
                                   "Examples: '~', '~/Downloads', '~/Documents/CollectiveOS'.",
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files and dotfiles. Defaults to false.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_local_file",
        "description": (
            "Read the text content of a file on this Mac. "
            "Accepts ~ and relative paths (resolved from home). "
            "Refuses binary files. Caps at 50 KB to stay within token limits."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file. Examples: '~/Downloads/notes.txt', "
                                   "'~/Documents/CollectiveOS/README.md'.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_local_file",
        "description": (
            "Write (create or overwrite) a file on this Mac with text content. "
            "Creates any missing parent directories automatically. "
            "Always confirm the file path and full content with the user before calling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to write. Must be inside the home directory. "
                                   "Example: '~/Documents/notes.txt'.",
                },
                "content": {
                    "type": "string",
                    "description": "Full text content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "browser_get_active_tab",
        "description": (
            "Get the title and URL of the currently active tab in Safari or Chrome. "
            "Use this when the user asks what page they are on, what site is open, "
            "or wants context about their current browsing."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "browser_list_tabs",
        "description": (
            "List all open tabs across all windows in Safari and Google Chrome. "
            "Returns titles and URLs grouped by browser and window."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "browser_open_url",
        "description": (
            "Open a URL in the default browser. "
            "Only http:// and https:// URLs are allowed. "
            "Always confirm the URL with the user before opening."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to open, e.g. 'https://example.com'.",
                },
            },
            "required": ["url"],
        },
    },
    # -----------------------------------------------------------------------
    # Apple-native: Contacts
    # -----------------------------------------------------------------------
    {
        "name": "contacts_search",
        "description": (
            "Search Apple Contacts by name and return phone numbers and email addresses. "
            "macOS will prompt for Contacts access on first use."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Full or partial name to search for, e.g. 'Alice' or 'Smith'.",
                },
            },
            "required": ["name"],
        },
    },
    # -----------------------------------------------------------------------
    # Apple-native: Reminders
    # -----------------------------------------------------------------------
    {
        "name": "reminders_list",
        "description": "List incomplete reminders from the Mac Reminders app. "
                       "Optionally filter by list name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "list_name": {
                    "type": "string",
                    "description": "Reminders list name to filter to, e.g. 'Groceries'. "
                                   "Leave blank to show all lists.",
                },
                "due_today": {
                    "type": "boolean",
                    "description": "If true, show only reminders due today or overdue.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "reminders_add",
        "description": (
            "Add a new reminder to the Mac Reminders app. "
            "Always confirm the title and due date with the user before calling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Reminder text.",
                },
                "due_date": {
                    "type": "string",
                    "description": "Optional due date in YYYY-MM-DD format, e.g. '2026-07-28'.",
                },
                "list_name": {
                    "type": "string",
                    "description": "Reminders list to add to. Defaults to the default list.",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "reminders_complete",
        "description": (
            "Mark a reminder as completed. "
            "Always confirm with the user before calling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title of the reminder to complete (partial match).",
                },
                "list_name": {
                    "type": "string",
                    "description": "Optional list name to narrow the search.",
                },
            },
            "required": ["title"],
        },
    },
    # -----------------------------------------------------------------------
    # Apple-native: Notes
    # -----------------------------------------------------------------------
    {
        "name": "notes_list",
        "description": "List note titles in the Mac Notes app. "
                       "Optionally filter by folder name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": "Notes folder name to filter to. Leave blank for all notes.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "notes_read",
        "description": "Read the full text content of a note by title (partial match).",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Full or partial note title.",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "notes_create",
        "description": (
            "Create a new note in the Mac Notes app. "
            "Always confirm the title and content with the user before calling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Note title."},
                "body":  {"type": "string", "description": "Note body text."},
                "folder": {
                    "type": "string",
                    "description": "Notes folder to create in (optional).",
                },
            },
            "required": ["title", "body"],
        },
    },
    {
        "name": "notes_append",
        "description": (
            "Append text to an existing note matched by title. "
            "Always confirm which note and what text with the user before calling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title of the note to append to (partial match).",
                },
                "text": {
                    "type": "string",
                    "description": "Text to add at the end of the note.",
                },
            },
            "required": ["title", "text"],
        },
    },
    # -----------------------------------------------------------------------
    # Apple-native: Clipboard
    # -----------------------------------------------------------------------
    {
        "name": "clipboard_read",
        "description": "Read the current text contents of the Mac clipboard. "
                       "Useful when the user says 'I just copied something' or "
                       "'use what's in my clipboard'.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "clipboard_write",
        "description": (
            "Copy text to the Mac clipboard, replacing its current contents. "
            "Always confirm the text with the user before calling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to copy to the clipboard.",
                },
            },
            "required": ["text"],
        },
    },
    # -----------------------------------------------------------------------
    # Telegram
    # -----------------------------------------------------------------------
    {
        "name": "telegram_get_messages",
        "description": (
            "Read recent messages sent to the Telegram bot. "
            "Returns sender name, chat ID, and message text. "
            "Use to proactively check for incoming Telegram messages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max messages to return. Defaults to 10."},
            },
            "required": [],
        },
    },
    {
        "name": "telegram_send",
        "description": (
            "Send a Telegram message to a specific chat ID or to the configured default chat. "
            "Requires explicit user confirmation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message text (Markdown supported)."},
                "chat_id": {"type": "string", "description": "Target chat ID (optional — uses TELEGRAM_CHAT_ID env var if omitted)."},
            },
            "required": ["message"],
        },
    },
    # -----------------------------------------------------------------------
    # Finance (read-only)
    # -----------------------------------------------------------------------
    {
        "name": "finance_get_accounts",
        "description": (
            "List connected bank and credit card accounts with real-time balances "
            "(current and available). Requires Plaid setup."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "finance_get_transactions",
        "description": (
            "Fetch recent transactions from all connected accounts, sorted newest first. "
            "Shows date, amount, merchant name, and spending category. "
            "Optionally filter to a single account by its Plaid account ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "How many days of history to fetch. Defaults to 30.",
                },
                "account_id": {
                    "type": "string",
                    "description": "Optional Plaid account ID to filter to one account.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "finance_get_spending_summary",
        "description": (
            "Summarize spending by top-level Plaid category (Food and Drink, "
            "Travel, Shopping, etc.) over the last N days. "
            "Shows totals and percentage of overall spend per category."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "How many days to summarize. Defaults to 30.",
                },
            },
            "required": [],
        },
    },
    # -----------------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------------
    {
        "name": "health_get_sleep",
        "description": (
            "Get sleep data for the last N days — total duration, deep sleep, REM, "
            "HRV, resting heart rate, and efficiency score. "
            "Uses Oura Ring if OURA_TOKEN is set, otherwise the Apple Health cache."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "How many days of history to return. Defaults to 7.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "health_get_activity",
        "description": (
            "Get activity data for the last N days — steps, active calories, "
            "total calories, and activity score. "
            "Uses Oura Ring if OURA_TOKEN is set, otherwise the Apple Health cache."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "How many days of history to return. Defaults to 7.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "health_get_readiness",
        "description": (
            "Get readiness score and HRV balance for the last N days — overall readiness, "
            "HRV balance, resting heart rate score, sleep balance, and activity balance. "
            "Uses Oura Ring if OURA_TOKEN is set, otherwise the Apple Health cache."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "How many days of history to return. Defaults to 7.",
                },
            },
            "required": [],
        },
    },
    # -----------------------------------------------------------------------
    # Slack
    # -----------------------------------------------------------------------
    {
        "name": "slack_list_channels",
        "description": (
            "List Slack channels the bot has access to, with member count and topic. "
            "Use this to discover channel names and IDs before reading messages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max channels to return. Defaults to 30.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "slack_read_messages",
        "description": (
            "Read recent messages from a Slack channel. "
            "Pass a channel name (e.g. 'general') or channel ID (e.g. 'C12345'). "
            "Messages are shown oldest-first with timestamps and sender names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel name (with or without #) or Slack channel ID.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of recent messages to fetch. Defaults to 20.",
                },
            },
            "required": ["channel"],
        },
    },
    {
        "name": "slack_send_message",
        "description": (
            "Send a message to a Slack channel or DM. "
            "Pass a channel name, channel ID, or a Slack user ID (U...) to send a direct message. "
            "Always confirm the channel and full message text with the user before calling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel name, channel ID (C...), or user ID (U...) for a DM.",
                },
                "text": {
                    "type": "string",
                    "description": "Message text to send. Supports Slack mrkdwn formatting.",
                },
            },
            "required": ["channel", "text"],
        },
    },
    # -----------------------------------------------------------------------
    # GitHub
    # -----------------------------------------------------------------------
    {
        "name": "github_list_repos",
        "description": (
            "List the authenticated GitHub user's repositories, most recently updated first. "
            "Includes repo name, language, visibility, and description."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max repos to return. Defaults to 20.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "github_list_prs",
        "description": "List pull requests for a GitHub repository (owner/repo format).",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Repository in owner/repo format, e.g. 'Anurag9Dhiman/CollectiveOS'.",
                },
                "state": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Filter by PR state. Defaults to 'open'.",
                },
            },
            "required": ["repo"],
        },
    },
    {
        "name": "github_list_issues",
        "description": "List issues for a GitHub repository. Excludes pull requests.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Repository in owner/repo format.",
                },
                "state": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Filter by issue state. Defaults to 'open'.",
                },
                "labels": {
                    "type": "string",
                    "description": "Comma-separated label names to filter by, e.g. 'bug,help wanted'.",
                },
            },
            "required": ["repo"],
        },
    },
    {
        "name": "github_get_ci_status",
        "description": (
            "Get CI check-run results for a branch, tag, or commit SHA in a repository. "
            "Shows pass/fail status for each check (Actions workflows, status checks, etc.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Repository in owner/repo format.",
                },
                "ref": {
                    "type": "string",
                    "description": "Branch name (e.g. 'main'), tag, or full commit SHA.",
                },
            },
            "required": ["repo", "ref"],
        },
    },
    {
        "name": "github_create_issue",
        "description": (
            "Create a new GitHub issue in a repository. "
            "Always confirm the repo, title, body, and labels with the user before calling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Repository in owner/repo format.",
                },
                "title": {
                    "type": "string",
                    "description": "Issue title.",
                },
                "body": {
                    "type": "string",
                    "description": "Issue body / description in Markdown. Optional.",
                },
                "labels": {
                    "type": "string",
                    "description": "Comma-separated label names to apply, e.g. 'bug,enhancement'. Optional.",
                },
            },
            "required": ["repo", "title"],
        },
    },
    # -----------------------------------------------------------------------
    # Notion
    # -----------------------------------------------------------------------
    {
        "name": "notion_search",
        "description": (
            "Search Notion for pages and databases by keyword. "
            "Returns titles, IDs, and URLs for matching results. "
            "Use this to find a page ID before reading or appending to it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term, e.g. 'daily notes', 'project plan', 'reading list'.",
                },
                "filter_type": {
                    "type": "string",
                    "enum": ["page", "database", "all"],
                    "description": "Limit results to 'page', 'database', or 'all' for both. Defaults to 'all'.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "notion_read_page",
        "description": "Read the full text content of a Notion page by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "page_id": {
                    "type": "string",
                    "description": "Notion page ID (from notion_search). Dashes are optional.",
                },
            },
            "required": ["page_id"],
        },
    },
    {
        "name": "notion_create_page",
        "description": (
            "Create a new Notion page under a parent page or database. "
            "Always confirm the title, parent, and content with the user before calling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "parent_id": {
                    "type": "string",
                    "description": "ID of the parent page or database (from notion_search).",
                },
                "title": {
                    "type": "string",
                    "description": "Title of the new page.",
                },
                "content": {
                    "type": "string",
                    "description": "Optional initial text content. Paragraphs separated by blank lines.",
                },
            },
            "required": ["parent_id", "title"],
        },
    },
    {
        "name": "notion_append_to_page",
        "description": (
            "Append text to the end of an existing Notion page. "
            "Always confirm which page and what content with the user before calling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "page_id": {
                    "type": "string",
                    "description": "Notion page ID to append to (from notion_search).",
                },
                "content": {
                    "type": "string",
                    "description": "Text to append. Paragraphs separated by blank lines.",
                },
            },
            "required": ["page_id", "content"],
        },
    },
    # -----------------------------------------------------------------------
    # Car
    # -----------------------------------------------------------------------
    {
        "name": "car_get_status",
        "description": "Get the current status of the user's car — charge level, lock state, climate, location.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "car_lock",
        "description": (
            "Lock or unlock the car. Requires explicit user confirmation before calling. "
            "action must be 'lock' or 'unlock'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "'lock' or 'unlock'."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "car_climate",
        "description": (
            "Turn the car climate on or off, optionally setting a target temperature. "
            "Requires explicit user confirmation. action must be 'on' or 'off'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "'on' or 'off'."},
                "temp_f": {"type": "number",  "description": "Target temperature in Fahrenheit (optional)."},
            },
            "required": ["action"],
        },
    },
    # -----------------------------------------------------------------------
    # Smart Appliances
    # -----------------------------------------------------------------------
    {
        "name": "appliances_list",
        "description": "List all smart appliances / devices connected via SmartThings or LG ThinQ.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "appliances_get_status",
        "description": "Get the current state of a specific smart appliance by device ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "string", "description": "SmartThings device ID."},
            },
            "required": ["device_id"],
        },
    },
    {
        "name": "appliances_control",
        "description": (
            "Send a command to a smart appliance. Requires explicit user confirmation. "
            "Heating appliances (microwave, cooktop, washer) can only be turned off — "
            "never started remotely. command is typically 'on', 'off', or a JSON dict."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "string", "description": "SmartThings device ID."},
                "command":   {"type": "string", "description": "'on', 'off', or a capability command."},
            },
            "required": ["device_id", "command"],
        },
    },
]

# Filter tool list to only those available on this device/platform
try:
    from collectiveos.platform_profile import enabled_tool_names
    _enabled = enabled_tool_names()
    TOOLS = [t for t in TOOLS if t["name"] in _enabled]
except Exception:
    pass  # if profile detection fails, keep all tools

# ---------------------------------------------------------------------------
# Gemini tool schema converters
# ---------------------------------------------------------------------------

def _json_schema_to_gemini(schema: dict) -> types.Schema:
    """Recursively convert a JSON Schema dict to a Gemini types.Schema."""
    type_map = {
        "string":  types.Type.STRING,
        "number":  types.Type.NUMBER,
        "integer": types.Type.INTEGER,
        "boolean": types.Type.BOOLEAN,
        "array":   types.Type.ARRAY,
        "object":  types.Type.OBJECT,
    }
    raw_type = schema.get("type", "string")
    t = type_map.get(raw_type.lower() if isinstance(raw_type, str) else "string",
                     types.Type.STRING)

    kwargs: dict = {"type": t, "description": schema.get("description", "")}

    if t == types.Type.OBJECT:
        props = {
            k: _json_schema_to_gemini(v)
            for k, v in schema.get("properties", {}).items()
        }
        if props:
            kwargs["properties"] = props
        if schema.get("required"):
            kwargs["required"] = list(schema["required"])

    elif t == types.Type.ARRAY and "items" in schema:
        kwargs["items"] = _json_schema_to_gemini(schema["items"])

    if "enum" in schema:
        kwargs["enum"] = [str(e) for e in schema["enum"] if e != ""]

    return types.Schema(**kwargs)


def _to_gemini_tools(anthropic_tools: list[dict]) -> list:
    """Convert the Anthropic tool-spec list to a Gemini Tool list."""
    if not anthropic_tools:
        return []
    declarations = []
    for tool in anthropic_tools:
        schema = tool.get("input_schema", {"type": "object", "properties": {}})
        declarations.append(types.FunctionDeclaration(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters=_json_schema_to_gemini(schema),
        ))
    return [types.Tool(function_declarations=declarations)]


def _convert_history(anthropic_history: list[dict]) -> list[dict]:
    """Convert Anthropic-format message history to Gemini format."""
    gemini = []
    for msg in anthropic_history:
        role = "model" if msg["role"] == "assistant" else "user"
        content = msg["content"]
        if isinstance(content, str):
            gemini.append({"role": role, "parts": [{"text": content}]})
        elif isinstance(content, list):
            texts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if texts:
                gemini.append({"role": role, "parts": [{"text": " ".join(texts)}]})
    return gemini


# ---------------------------------------------------------------------------
# Agent loop — batch (CLI) and streaming (API) variants
# ---------------------------------------------------------------------------

MAX_LOOP = 10  # hard cap on tool-use iterations per user message

_TRANSIENT_MARKERS = ("Timeout", "Connection", "Network", "Reset", "BrokenPipe")


def _exec_tool_fn(name: str, args: dict) -> str:
    """
    Execute a single tool by name+args and return its string output.
    Permission checks, retries on transient errors, and observability logging
    are all handled here. Errors are returned as [ERROR: ...] strings so the
    model can decide what to do rather than crashing the loop.
    """
    allowed, err = permissions.check_tool(name)
    if not allowed:
        return err

    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"[ERROR: unknown tool '{name}']"

    t0 = time.monotonic()
    for attempt in range(2):
        try:
            result = str(func(**args))
            _obs.log_tool_call(name, int((time.monotonic() - t0) * 1000), success=True)
            return result
        except Exception as exc:
            exc_type = type(exc).__name__
            if attempt == 0 and any(m in exc_type for m in _TRANSIENT_MARKERS):
                time.sleep(1)
                continue
            _obs.log_tool_call(name, int((time.monotonic() - t0) * 1000),
                               success=False, error_type=exc_type)
            return f"[ERROR: {exc_type} — {exc}]"

    return "[ERROR: unexpected retry exhaustion]"


def _exec_one_gemini(fc) -> types.Part:
    """Execute a Gemini FunctionCall and wrap the result in a FunctionResponse Part."""
    result = _exec_tool_fn(fc.name, dict(fc.args))
    return types.Part(
        function_response=types.FunctionResponse(
            name=fc.name,
            response={"result": result},
        )
    )


def _cancelled_part(fc) -> types.Part:
    return types.Part(
        function_response=types.FunctionResponse(
            name=fc.name,
            response={"result": "[Action cancelled by user.]"},
        )
    )


def _run_tools_parallel_gemini(fn_calls: list) -> list:
    """Execute all Gemini FunctionCall objects concurrently (up to 8 threads)."""
    if not fn_calls:
        return []
    with ThreadPoolExecutor(max_workers=min(len(fn_calls), 8)) as pool:
        parts = list(pool.map(_exec_one_gemini, fn_calls))
    return parts


def _run_tools_with_confirm(fn_calls: list, confirm_fn) -> list:
    """Execute tool calls, gating every write tool through confirm_fn.

    confirm_fn(tool_name, args) must be synchronous and return True (proceed)
    or False (cancel). Write tools are executed sequentially so confirmation
    requests don't overlap; read tools are batched in parallel between them.
    """
    parts = []
    read_batch = []

    def flush_reads():
        if read_batch:
            parts.extend(_run_tools_parallel_gemini(read_batch))
            read_batch.clear()

    for fc in fn_calls:
        if fc.name in permissions.WRITE_TOOLS:
            flush_reads()
            print(f"  [tool: {fc.name} — awaiting confirmation]")
            if confirm_fn(fc.name, dict(fc.args)):
                print(f"  [tool: {fc.name} — approved, executing]")
                parts.append(_exec_one_gemini(fc))
            else:
                print(f"  [tool: {fc.name} — cancelled by user]")
                parts.append(_cancelled_part(fc))
        else:
            print(f"  [tool: {fc.name}({dict(fc.args)})]")
            read_batch.append(fc)

    flush_reads()
    return parts


def _extract_fn_calls(response) -> list:
    """Extract all FunctionCall objects from a Gemini response."""
    calls = []
    try:
        for part in response.candidates[0].content.parts:
            if part.function_call and part.function_call.name:
                calls.append(part.function_call)
    except (IndexError, AttributeError):
        pass
    return calls


def run(user_message: str, system: str = "", history: list = [],
        confirm_fn=None) -> str:
    """Run one user message through the Gemini tool-use loop; return the full reply.

    history:    prior turns in Anthropic format — converted automatically.
    confirm_fn: optional sync callable(tool_name, args) -> bool. When set,
                every write tool call is gated through it before execution.
                Use this in voice sessions so the user must approve each action.
                Read tools always run immediately.
    Independent tool calls within a single turn execute in parallel threads.
    Capped at MAX_LOOP iterations; if hit, returns whatever text is available.
    """
    active_tools, categories = router.select_tools(user_message, TOOLS)
    if categories:
        print(f"  [router: {', '.join(categories)}  →  {len(active_tools)}/{len(TOOLS)} tools]")

    gemini_tools = _to_gemini_tools(active_tools)
    chat = _get_client().chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            tools=gemini_tools or None,
            system_instruction=system or None,
        ),
        history=_convert_history(history),
    )

    # First message is the user string; subsequent messages are FunctionResponse parts
    message = user_message

    for _iter in range(MAX_LOOP):
        response = chat.send_message(message)

        try:
            if response.usage_metadata:
                _obs.log_api_call(
                    MODEL,
                    response.usage_metadata.prompt_token_count,
                    response.usage_metadata.candidates_token_count,
                )
        except Exception:
            pass

        fn_calls = _extract_fn_calls(response)
        if not fn_calls:
            return response.text or ""

        if confirm_fn is not None:
            message = _run_tools_with_confirm(fn_calls, confirm_fn)
        else:
            for fc in fn_calls:
                print(f"  [tool: {fc.name}({dict(fc.args)})]")
            message = _run_tools_parallel_gemini(fn_calls)

    return response.text or "[Reached the iteration limit. Please try a more focused request.]"


def run_stream(user_message: str, system: str = "", history: list = []):
    """
    Generator — yields text tokens as they arrive, then executes any tool calls
    and continues. Status markers are yielded before tool execution so the UI
    stays responsive. Capped at MAX_LOOP iterations.

    history: prior turns in Anthropic format — converted automatically.
    """
    active_tools, categories = router.select_tools(user_message, TOOLS)
    if categories:
        yield f"_[routing: {', '.join(categories)}]_\n\n"

    gemini_tools = _to_gemini_tools(active_tools)
    chat = _get_client().chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            tools=gemini_tools or None,
            system_instruction=system or None,
        ),
        history=_convert_history(history),
    )

    message = user_message

    for _iter in range(MAX_LOOP):
        fn_calls: list = []
        last_chunk = None

        for chunk in chat.send_message_stream(message):
            if chunk.text:
                yield chunk.text
            last_chunk = chunk

        # Function calls come in the last chunk's candidates
        fn_calls = _extract_fn_calls(last_chunk) if last_chunk else []

        try:
            if last_chunk and last_chunk.usage_metadata:
                _obs.log_api_call(
                    MODEL,
                    last_chunk.usage_metadata.prompt_token_count,
                    last_chunk.usage_metadata.candidates_token_count,
                )
        except Exception:
            pass

        if not fn_calls:
            break

        for fc in fn_calls:
            marker = "ACTION" if fc.name in permissions.WRITE_TOOLS else "reading"
            yield f"\n\n_[{marker}: {fc.name}...]_\n\n"

        message = _run_tools_parallel_gemini(fn_calls)
    else:
        yield "\n\n_[iteration limit reached — summarising...]_\n\n"
        try:
            for chunk in chat.send_message_stream(message):
                if chunk.text:
                    yield chunk.text
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Personal assistant ready. Type 'quit' to exit.\n")
    print("Try: 'what's on my calendar?', 'show recent emails', 'list my Drive files'\n")

    cli_history: list[dict] = []

    while True:
        user_input = input("you> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break

        # 1. Build system prompt with current date + semantic memory.
        past = memory.search(user_input)
        now = datetime.datetime.now(datetime.timezone.utc)
        date_str = now.strftime("%A, %B %-d, %Y, %H:%M UTC")
        system_prompt = (
            f"You are a helpful personal assistant.\n"
            f"Today is {date_str}."
        )
        if past:
            system_prompt += "\n\nRelevant context from past conversations:\n" + past

        # 2. Run the agent with in-session history so it remembers prior turns.
        reply = run(user_input, system=system_prompt, history=cli_history)
        print(f"assistant> {reply}\n")

        # 3. Extend in-session history (keep last 20 messages = 10 turns).
        cli_history.append({"role": "user", "content": user_input})
        cli_history.append({"role": "assistant", "content": reply})
        cli_history = cli_history[-20:]

        # 4. Save this exchange so future sessions can recall it.
        memory.save(user_input, reply)
