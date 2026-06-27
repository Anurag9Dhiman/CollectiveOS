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

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic

from src.connectors.google_calendar import get_calendar_events
from src.connectors.gmail import get_recent_emails, search_emails
from src.connectors.google_drive import list_files, read_file
from src.connectors.todoist import get_tasks, get_projects, add_task
from src.connectors.home_assistant import get_devices, get_device_state, control_device
from src import memory

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
    "get_recent_emails":   get_recent_emails,
    "search_emails":       search_emails,
    "list_drive_files":    list_files,
    "read_drive_file":     read_file,
    "get_tasks":           get_tasks,
    "get_projects":        get_projects,
    "add_task":            add_task,
    "get_devices":         get_devices,
    "get_device_state":    get_device_state,
    "control_device":      control_device,
    "set_light":           set_light,
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
]

# ---------------------------------------------------------------------------
# Agent loop — batch (CLI) and streaming (API) variants
# ---------------------------------------------------------------------------

def run(user_message: str, system: str = "") -> str:
    """Run one user message through the tool-use loop and return the full reply."""
    messages = [{"role": "user", "content": user_message}]

    kwargs = dict(model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages)
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


def run_stream(user_message: str, system: str = ""):
    """
    Generator variant — yields text tokens as they arrive from the API.
    Tool calls are executed synchronously between stream calls; a short
    status line is yielded while tools run so the UI stays responsive.
    """
    messages = [{"role": "user", "content": user_message}]

    kwargs = dict(model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages)
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
                yield f"\n\n_[using {block.name}…]_\n\n"
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

    while True:
        user_input = input("you> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break

        # 1. Retrieve relevant past memory to give the model context.
        past = memory.search(user_input)
        system_prompt = "You are a helpful personal assistant."
        if past:
            system_prompt += (
                "\n\nRelevant context from past conversations:\n" + past
            )

        # 2. Run the agent.
        reply = run(user_input, system=system_prompt)
        print(f"assistant> {reply}\n")

        # 3. Save this exchange so future turns can recall it.
        memory.save(user_input, reply)
