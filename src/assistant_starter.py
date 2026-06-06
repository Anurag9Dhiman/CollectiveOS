"""
assistant_starter.py
---------------------
A minimal personal-assistant agent loop.

The whole point of this file is to teach you ONE pattern:

    you ask -> model decides to use a tool -> your code runs the tool
    -> the result goes back to the model -> model replies in plain words

That loop is the same for EVERY connector you'll ever add. Right now the
"tools" are fake (they just return made-up data). Later you replace the
*insides* of these functions with real API calls (calendar, lights, car...),
and the loop below does not change at all.

SETUP (do this once):
    1. Install Python 3.9+        ->  https://www.python.org/downloads/
    2. pip install anthropic
    3. Get an API key            ->  https://console.anthropic.com/
    4. Set it in your terminal:
           macOS/Linux:  export ANTHROPIC_API_KEY="sk-ant-..."
           Windows:      setx ANTHROPIC_API_KEY "sk-ant-..."
    5. Run it:  python assistant_starter.py
"""

from anthropic import Anthropic

# The SDK automatically reads the ANTHROPIC_API_KEY environment variable.
client = Anthropic()

# Sonnet is a good, affordable default for this kind of work.
MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# 1. THE TOOLS (fake for now)
#
# Each tool is just a normal Python function. These two pretend to do
# something. When you're ready, the body of each function is where a REAL
# API call goes -- the rest of this file stays exactly the same.
# ---------------------------------------------------------------------------

def get_current_time(timezone: str = "local") -> str:
    """Pretend to look up the time. Replace with a real clock/API later."""
    return f"It is 3:42 PM in the {timezone} timezone (pretend value)."


def set_light(room: str, state: str) -> str:
    """Pretend to switch a light. Later this calls Home Assistant, etc."""
    return f"OK, the {room} light is now {state} (pretend action)."


# A simple registry so the loop can find a function by its name.
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
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
        "name": "get_current_time",
        "description": "Get the current time in a given timezone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name, e.g. 'local' or 'UTC'.",
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
    print("Try: 'what time is it?'  or  'turn off the kitchen light'\n")
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        reply = run(user_input)
        print(f"assistant> {reply}\n")
