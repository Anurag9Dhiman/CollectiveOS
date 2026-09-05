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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


from src.connectors.home_assistant import get_devices, get_device_state, control_device
from src.connectors import spotify as _spotify
from src.connectors import mac_system as _mac
from src.connectors import local_files as _fs
from src.connectors import health as _health
from src.connectors import car as _car
from src.connectors import appliances as _appliances
from src.connectors.ios_push import push_notification as _push_notification
from src.agents.nav_agent import (
    navigate_computer as _nav_async,
    get_and_clear_first_person_frame,
)


def _navigate_computer_sync(task: str, context: str = "") -> str:
    """Sync shim — runs the async NavAgent in a fresh event loop (ThreadPoolExecutor-safe).

    Picks up any wearable frame stored by set_first_person_frame() before this
    agent turn (e.g. from a Frame glasses WebSocket session) and passes it as
    first_person_frame so the nav agent can see what the user physically sees.
    """
    import asyncio
    frame = get_and_clear_first_person_frame()  # consume once; None for non-wearable tasks
    return asyncio.run(_nav_async(task, context, _first_person_frame=frame))
from src import memory, graph_memory, router, permissions, observability as _obs
from src import output_bus as _output_bus
from src import orchestrator as _orchestrator
from src.connectors import wearable as _wearable
from src.connectors import ros2_mcp as _ros2
from src import mcp_client as _mcp

_mcp.load()  # connect to any configured MCP servers at import time

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

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


def notify_user(message: str, channel: str = "notification") -> str:
    """Deliver a proactive message to the user via the specified channel."""
    if channel not in _output_bus.VALID_CHANNELS:
        return f"[ERROR: unknown channel '{channel}'. Use: notification, telegram, push, both]"
    _output_bus.deliver(title="Assistant", body=message, channel=channel)
    return f"Delivered via {channel}."
# ---------------------------------------------------------------------------
# Orchestrator wrappers
# ---------------------------------------------------------------------------

def task_plan(description: str) -> str:
    """Plan and execute a multi-step agentic task using the orchestrator."""
    return _orchestrator.plan_and_run(description)


def task_status(task_id: int) -> str:
    """Return the current status of a planned task and its steps."""
    task = _orchestrator.get_task(task_id)
    if not task:
        return f"Task #{task_id} not found."
    lines = [f"Task #{task['id']} [{task['status']}]: {task['description']}"]
    for s in task.get("steps", []):
        mark = {"completed": "✓", "failed": "✗", "running": "⟳", "pending": "·"}.get(s["status"], "?")
        out = f" → {s['output'][:100]}" if s.get("output") else ""
        lines.append(f"  {mark} {s['tool']}{out}")
    return "\n".join(lines)


def task_list() -> str:
    """List the 10 most recent tasks and their statuses."""
    tasks = _orchestrator.list_tasks(limit=10)
    if not tasks:
        return "No tasks found."
    lines = [f"#{t['id']} [{t['status']}] {t['description'][:80]}" for t in tasks]
    return "\n".join(lines)


def task_cancel(task_id: int) -> str:
    """Cancel a pending or running task."""
    return _orchestrator.cancel_task(task_id)


def mcp_list_servers() -> str:
    """List connected MCP servers and the tools they expose."""
    servers = _mcp.list_servers()
    if not servers:
        return "No MCP servers connected. Add servers to mcp_servers.json to enable them."
    lines = [f"Connected MCP servers ({len(servers)}):"]
    for s in servers:
        lines.append(f"  • {s['server']} — {s['tools']} tool(s)")
    return "\n".join(lines)


TOOL_FUNCTIONS = {
    "memory_remember":    memory_remember,
    "memory_list":        memory_list,
    "memory_forget":      memory_forget,
    "memory_graph_query": memory_graph_query,
    "usage_summary":      usage_summary,
    "mcp_list_servers":   mcp_list_servers,
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
    "list_directory":        _fs.list_directory,
    "read_local_file":       _fs.read_local_file,
    "write_local_file":      _fs.write_local_file,
    "health_get_sleep":       _health.health_get_sleep,
    "health_get_activity":    _health.health_get_activity,
    "health_get_readiness":   _health.health_get_readiness,
    "car_get_status":   _car.car_get_status,
    "car_lock":         _car.car_lock,
    "car_climate":      _car.car_climate,
    "appliances_list":       _appliances.appliances_list,
    "appliances_get_status": _appliances.appliances_get_status,
    "appliances_control":    _appliances.appliances_control,
    "push_notification":     _push_notification,
    "notify_user":           notify_user,
    "task_plan":             task_plan,
    "task_status":           task_status,
    "task_list":             task_list,
    "task_cancel":           task_cancel,
    "navigate_computer":     _navigate_computer_sync,
    "wearable_get_events":   _wearable.wearable_get_events,
    "wearable_list_devices": _wearable.wearable_list_devices,
    "robot_status":          _ros2.robot_status,
    "robot_move":            _ros2.robot_move,
    "robot_cancel":          _ros2.robot_cancel,
    "robot_navigate":        _ros2.robot_navigate,
    "robot_describe_scene":  _ros2.robot_describe_scene,
    # MCP server tools are merged in below
}
TOOL_FUNCTIONS.update(_mcp.tool_callables())

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
    # -----------------------------------------------------------------------
    # iOS push notifications (APNs)
    # -----------------------------------------------------------------------
    {
        "name": "push_notification",
        "description": (
            "Send a push notification to the user's iPhone via Apple Push Notification service (APNs). "
            "Use when the user says 'remind me on my phone', 'send me a notification', or 'ping me when…', "
            "or when a proactive routine needs to alert the user on their device. "
            "Requires APNS_KEY_ID, APNS_TEAM_ID, APNS_AUTH_KEY_PATH, APNS_BUNDLE_ID, "
            "and APNS_DEVICE_TOKEN to be configured in the environment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short notification title shown in bold at the top of the alert.",
                },
                "body": {
                    "type": "string",
                    "description": "Notification body text — the main message.",
                },
                "badge": {
                    "type": "integer",
                    "description": "Optional badge count to display on the app icon. Omit to leave the badge unchanged.",
                },
                "sound": {
                    "type": "string",
                    "description": "Alert sound to play — 'default' or a custom sound filename bundled in the app. Default: 'default'.",
                },
            },
            "required": ["title", "body"],
        },
    },
    # -----------------------------------------------------------------------
    # Output bus
    # -----------------------------------------------------------------------
    {
        "name": "notify_user",
        "description": (
            "Proactively deliver a message to the user via a chosen channel. "
            "Use when the user says 'let me know on Telegram', 'send me a notification', "
            "'ping me on my phone', or when you need to alert the user after completing "
            "background work. "
            "Channels: notification (Mac banner), telegram (Telegram message), "
            "push (iOS APNs), both (Mac + Telegram)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message text to deliver to the user.",
                },
                "channel": {
                    "type": "string",
                    "description": "Delivery channel: notification | telegram | push | both. Default: notification.",
                    "enum": ["notification", "telegram", "push", "both"],
                },
            },
            "required": ["message"],
        },
    },
    # -----------------------------------------------------------------------
    # Task orchestrator
    # -----------------------------------------------------------------------
    {
        "name": "task_plan",
        "description": (
            "Plan and execute a complex multi-step task using the task orchestrator. "
            "Use when the user asks for something that requires several sequential tool calls — "
            "for example 'research X, summarise it, and save to Notion', or "
            "'check my emails and add follow-up tasks to Todoist'. "
            "The orchestrator creates a plan (up to 10 steps), runs each step, "
            "and returns a summary of what was done. "
            "Results are stored in the tasks database so you can check them later with task_status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Full description of the task to accomplish, including any specific constraints or output requirements.",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "task_status",
        "description": (
            "Check the status of a previously created task, including each step's result. "
            "Use after task_plan to confirm completion, or when the user asks 'how did that task go'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The task ID returned by task_plan or task_list.",
                },
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "task_list",
        "description": "List the 10 most recent tasks and their statuses (completed, failed, running, cancelled).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "task_cancel",
        "description": "Cancel a task that is still pending, planning, or running.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The task ID to cancel.",
                },
            },
            "required": ["task_id"],
        },
    },
    # -----------------------------------------------------------------------
    # Computer use (desktop agent)
    # -----------------------------------------------------------------------
    {
        "name": "navigate_computer",
        "description": (
            "Navigate the macOS GUI to complete any task — no separate API key needed. "
            "Automatically picks the best path:\n"
            "  • Web / browser tasks (Gmail, GitHub, Notion, Slack, Google Calendar, "
            "    any website): uses Playwright browser automation + Gemini Vision — "
            "    follows links, fills forms, clicks buttons.\n"
            "  • Native desktop app tasks (Finder, Terminal, Xcode, Mail, any native app): "
            "    uses Gemini Vision loop + pyautogui — sees the screen, decides the next "
            "    action, executes, repeats.\n\n"
            "Use this for ANY task that involves interacting with an app on screen: "
            "read and send emails, manage GitHub PRs, edit Notion pages, browse websites, "
            "run Terminal commands, or use any macOS app. "
            "Built-in safety gate: pauses and asks the user before irreversible sub-actions "
            "(send, delete, purchase, submit, book …). "
            "PREFERRED over computer_use for all new tasks. "
            "IMPORTANT: confirm with the user before calling this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Precise description of what to do. Name the app or website, "
                        "the target element, and the desired outcome. "
                        "Example: 'Open Gmail, find the latest email from Alice, "
                        "reply with: On my way!'"
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Optional user context to help the agent make decisions, "
                        "e.g. account email, username, or relevant preferences."
                    ),
                },
            },
            "required": ["task"],
        },
    },
    # -----------------------------------------------------------------------
    # MCP server management
    # -----------------------------------------------------------------------
    {
        "name": "mcp_list_servers",
        "description": (
            "List all connected MCP (Model Context Protocol) servers and the number of tools "
            "each one exposes. Use when the user asks what MCP servers are running, or to "
            "diagnose why an MCP-backed tool isn't available."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # -----------------------------------------------------------------------
    # Wearable devices
    # -----------------------------------------------------------------------
    {
        "name": "wearable_get_events",
        "description": (
            "Retrieve recent events from wearable devices (Garmin, Frame glasses, "
            "Apple Watch via Shortcuts, or any custom device that POSTs to /wearable/ingest). "
            "Use when the user asks what their wearable detected, or to check gestures, "
            "sensor readings, or button presses from a device."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of events to return (default 20, max 100).",
                },
                "device_id": {
                    "type": "string",
                    "description": "Filter to a specific device ID (e.g. 'garmin-forerunner-265').",
                },
                "event_type": {
                    "type": "string",
                    "description": "Filter by event type: gesture, sensor, location, button, voice, heartrate.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "wearable_list_devices",
        "description": "List all wearable devices that have sent data, with their last-seen time and event count.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # -----------------------------------------------------------------------
    # Robot (ROS2 MCP)
    # -----------------------------------------------------------------------
    {
        "name": "robot_status",
        "description": (
            "Get the current status, position, and battery of the connected robot. "
            "Works with any ROS2-based robot via the ROS2_MCP_URL endpoint. "
            "Returns stub data when ROS2_MCP_URL is not configured."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "robot_move",
        "description": (
            "Command the robot to move in a direction. "
            "Requires HITL approval before execution. "
            "direction: forward | backward | left | right | stop"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "description": "Movement direction: forward, backward, left, right, or stop.",
                    "enum": ["forward", "backward", "left", "right", "stop"],
                },
                "distance_m": {
                    "type": "number",
                    "description": "Distance to travel in metres (default 1.0, ignored for stop).",
                },
                "speed_ms": {
                    "type": "number",
                    "description": "Speed in metres per second (default 0.3, capped at 0.5).",
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "robot_cancel",
        "description": "Send an emergency stop to the robot — halts all motion immediately.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "robot_navigate",
        "description": (
            "Navigate the robot to a named room or location by planning the shortest "
            "path through the home and executing it. "
            "destination must be one of: bedroom, hallway, living_room, office, kitchen. "
            "Requires HITL approval before execution — physical motion."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "Target room name. One of: bedroom, hallway, living_room, office, kitchen.",
                    "enum": ["bedroom", "hallway", "living_room", "office", "kitchen"],
                },
            },
            "required": ["destination"],
        },
    },
    {
        "name": "robot_describe_scene",
        "description": (
            "Ask the robot to describe its current environment — visible objects, "
            "exits, and notable features in the room it is currently in. "
            "Read-only: does not move the robot. "
            "Use before navigating or when the user asks 'what can you see?'."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

# Merge MCP-discovered tool schemas (populated once mcp.load() ran above)
TOOLS.extend(_mcp.list_tools())

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

# Read-only tools whose results can be cached in Redis.
# TTL in seconds — chosen conservatively to balance freshness vs. API load.
CACHE_TTL: dict[str, int] = {
    "health_get_sleep":              300,
    "health_get_activity":           300,
    "health_get_readiness":          300,
    "car_get_status":                 30,
    "get_devices":                    60,
    "get_device_state":               15,
    "appliances_list":                60,
    "appliances_get_status":          30,
}


def _exec_tool_fn(name: str, args: dict) -> str:
    """
    Execute a single tool by name+args and return its string output.
    Permission checks, retries on transient errors, and observability logging
    are all handled here. Errors are returned as [ERROR: ...] strings so the
    model can decide what to do rather than crashing the loop.

    Read-only tools listed in CACHE_TTL are served from Redis (or the
    in-process fallback) when a matching result exists; the live API is
    called on a cache miss and the result is stored for the configured TTL.
    """
    allowed, err = permissions.check_tool(name)
    if not allowed:
        return err

    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"[ERROR: unknown tool '{name}']"

    # Cache read-only tool results to reduce external API calls
    ttl = CACHE_TTL.get(name, 0)
    if ttl > 0:
        import json as _json
        from src import redis_client as _cache
        cache_key = f"tool:{name}:{_json.dumps(args, sort_keys=True)}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return str(cached)

    t0 = time.monotonic()
    for attempt in range(2):
        try:
            result = str(func(**args))
            _obs.log_tool_call(name, int((time.monotonic() - t0) * 1000), success=True)
            if ttl > 0:
                _cache.set(cache_key, result, ttl=ttl)
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


def _run_tools_parallel_gemini(fn_calls: list) -> list:
    """Execute all Gemini FunctionCall objects concurrently (up to 8 threads)."""
    if not fn_calls:
        return []
    with ThreadPoolExecutor(max_workers=min(len(fn_calls), 8)) as pool:
        parts = list(pool.map(_exec_one_gemini, fn_calls))
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


def run(user_message: str, system: str = "", history: list = []) -> str:
    """Run one user message through the Gemini tool-use loop; return the full reply.

    history: prior turns in Anthropic format — converted automatically.
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
