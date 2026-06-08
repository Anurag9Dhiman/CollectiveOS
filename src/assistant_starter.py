"""
assistant_starter.py
---------------------
A personal-assistant agent loop with a real Google Calendar connector.

Pattern:
    you ask -> model decides to use a tool -> your code runs the tool
    -> the result goes back to the model -> model replies in plain words

SETUP (do this once):
    1. pip install -r requirements.txt
    2. Follow the Google Cloud setup in src/connectors/google_calendar.py,
       then save credentials.json in the repo root.
    3. export ANTHROPIC_API_KEY="sk-ant-..."
    4. python src/assistant_starter.py
       (first run opens a browser for Google OAuth consent)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from src.connectors.google_calendar import get_calendar_events

# The SDK automatically reads the ANTHROPIC_API_KEY environment variable.
client = Anthropic()

# Sonnet is a good, affordable default for this kind of work.
MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# 1. THE TOOLS
#
# Real connector: get_calendar_events (calls Google Calendar API).
# Placeholder: set_light (replace body with Home Assistant later).
# ---------------------------------------------------------------------------

def set_light(room: str, state: str) -> str:
    """Placeholder — replace body with a real Home Assistant call later."""
    return f"OK, the {room} light is now {state} (pretend action)."


# A simple registry so the loop can find a function by its name.
TOOL_FUNCTIONS = {
    "get_calendar_events": get_calendar_events,
    "set_light": set_light,
}


# ---------------------------------------------------------------------------
# 2. THE TOOL DESCRIPTIONS
#
# This is how you DESCRIBE the tools to the model so it knows what exists
# and what arguments each one takes. The model never runs your code; it just
# tells you "I'd like to call set_light with these arguments," and your loop
# does the actual calling.
# ---------------------------------------------------------------------------

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
        "name": "set_light",
        "description": "Turn a light on or off in a specific room.",
        "input_schema": {
            "type": "object",
            "properties": {
                "room": {"type": "string", "description": "e.g. 'kitchen'."},
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
# 3. THE AGENT LOOP  <-- this is the part worth understanding deeply
# ---------------------------------------------------------------------------

def run(user_message: str) -> str:
    # The running transcript of the conversation.
    messages = [{"role": "user", "content": user_message}]

    # Loop because the model may want several tool calls before it's done.
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # If the model did NOT ask for a tool, it's giving a final answer.
        if response.stop_reason != "tool_use":
            # Pull out the plain-text parts of the reply.
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        # Otherwise: record what the model said (its tool request)...
        messages.append({"role": "assistant", "content": response.content})

        # ...run each requested tool and collect the results.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                func = TOOL_FUNCTIONS[block.name]      # find the function
                output = func(**block.input)           # run it with model's args
                print(f"  [ran {block.name}({block.input}) -> {output}]")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )

        # Send the results back so the model can continue. Then loop.
        messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# 4. TRY IT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Personal-assistant starter. Type 'quit' to exit.\n")
    print("Try: 'what's on my calendar this week?' or 'turn off the kitchen light'\n")
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        reply = run(user_input)
        print(f"assistant> {reply}\n")
