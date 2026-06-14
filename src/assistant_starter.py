"""
assistant_starter.py
---------------------
Personal-assistant agent loop with:
  - Google Calendar (read upcoming events)
  - Gmail          (read inbox, search emails)
  - SQLite memory  (remembers past conversations)

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
# Agent loop
# ---------------------------------------------------------------------------

def run(user_message: str, system: str = "") -> str:
    """Run one user message through the tool-use loop and return the reply."""
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

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Personal assistant ready. Type 'quit' to exit.\n")
    print("Try: 'what's on my calendar this week?' or 'show my recent emails'\n")

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
