"""
assistant_starter.py
---------------------
Personal-assistant agent loop with:
  - Google Calendar  (read upcoming events)
  - Gmail            (read inbox, search emails)
  - Google Drive     (list and read files)
  - Todoist          (list tasks, projects; add tasks)
  - Home Assistant   (read device states, control lights/switches)
  - Postgres memory  (semantic search via pgvector)

Pattern:
    retrieve relevant memory
    -> build system prompt with context
    -> user asks -> model calls tools -> your code runs them -> loop
    -> model gives final answer
    -> save exchange to memory

SETUP (do this once):
    1. pip install -r requirements.txt
    2. Follow Google Cloud setup in src/connectors/google_auth.py docstring,
       save credentials.json (Desktop app type) in the repo root.
    3. export ANTHROPIC_API_KEY="sk-ant-..."
    4. python src/assistant_starter.py
       (first run opens a browser for Google OAuth — grants Calendar + Gmail)
"""

import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic

from src.connectors.google_calendar import get_calendar_events, create_event
from src.connectors.gmail import get_recent_emails, search_emails, create_draft, send_email
from src.connectors.google_drive import list_files, read_file
from src.connectors.todoist import get_tasks, get_projects, add_task, complete_task, update_task
from src.connectors.home_assistant import get_devices, get_device_state, control_device
from src.connectors import spotify as _spotify
from src.connectors import mac_system as _mac
from src.connectors.web_search import search as web_search
from src import memory, router, permissions

client = Anthropic()
MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def set_light(room: str, state: str) -> str:
    """Placeholder — replace body with a real Home Assistant call later."""
    return f"OK, the {room} light is now {state} (pretend action)."


TOOL_FUNCTIONS = {
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
}

TOOLS = [
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
]

# ---------------------------------------------------------------------------
# Agent loop — batch (CLI) and streaming (API) variants
# ---------------------------------------------------------------------------

def run(user_message: str, system: str = "", history: list = []) -> str:
    """Run one user message through the tool-use loop and return the full reply.

    history: prior [{"role": ..., "content": ...}] turns to prepend, oldest first.
    """
    messages = history + [{"role": "user", "content": user_message}]

    active_tools, categories = router.select_tools(user_message, TOOLS)
    if categories:
        print(f"  [router: {', '.join(categories)}  →  {len(active_tools)}/{len(TOOLS)} tools]")

    kwargs = dict(model=MODEL, max_tokens=1024, tools=active_tools, messages=messages)
    if system:
        kwargs["system"] = system

    while True:
        response = client.messages.create(**kwargs)

        if response.stop_reason != "tool_use":
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                allowed, err = permissions.check_tool(block.name)
                if not allowed:
                    output = err
                else:
                    func   = TOOL_FUNCTIONS[block.name]
                    output = func(**block.input)
                print(f"  [tool: {block.name}({block.input})]")
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     output,
                })

        messages.append({"role": "user", "content": tool_results})
        kwargs["messages"] = messages


def run_stream(user_message: str, system: str = "", history: list = []):
    """
    Generator variant — yields text tokens as they arrive from the API.
    Tool calls are executed synchronously between stream calls; a short
    status line is yielded while tools run so the UI stays responsive.

    history: prior [{"role": ..., "content": ...}] turns to prepend, oldest first.
    """
    messages = history + [{"role": "user", "content": user_message}]

    active_tools, categories = router.select_tools(user_message, TOOLS)
    if categories:
        yield f"_[routing: {', '.join(categories)}]_\n\n"

    kwargs = dict(model=MODEL, max_tokens=1024, tools=active_tools, messages=messages)
    if system:
        kwargs["system"] = system

    while True:
        with client.messages.stream(**kwargs) as stream:
            for token in stream.text_stream:
                yield token
            final = stream.get_final_message()

        if final.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": final.content})

        tool_results = []
        for block in final.content:
            if block.type == "tool_use":
                is_write = block.name in permissions.WRITE_TOOLS
                marker = "ACTION" if is_write else "reading"
                yield f"\n\n_[{marker}: {block.name}…]_\n\n"
                allowed, err = permissions.check_tool(block.name)
                if not allowed:
                    output = err
                else:
                    func   = TOOL_FUNCTIONS[block.name]
                    output = func(**block.input)
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     output,
                })

        messages.append({"role": "user", "content": tool_results})
        kwargs["messages"] = messages

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
