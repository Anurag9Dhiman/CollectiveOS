"""
Computer Agent connector — Gemini vision-based desktop control.

Exposes one tool: computer_use(task) — an agentic loop that takes screenshots,
sends them to Gemini Vision, and receives structured actions (click, type,
scroll, key) to execute via pyautogui, repeating until the task is done.

Uses the existing GEMINI_API_KEY — no additional API keys required.

Safety: registered as a WRITE-tier tool so every invocation goes through the
HITL interrupt gate before executing. Requires explicit user approval.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
from typing import Any

_MODEL = os.environ.get("GEMINI_COMPUTER_MODEL", "gemini-3.6-flash")
_MAX_ITER = 20
_ACTION_PAUSE = 0.5   # seconds to wait after each action for the UI to settle

_SYSTEM = """You are a computer-use agent controlling a macOS desktop.
You will receive a screenshot and a task. Respond ONLY with a JSON object
describing the single next action to take. Do not explain. Do not add text
outside the JSON.

Action schema (choose ONE):
  {"action": "left_click",  "coordinate": [x, y]}
  {"action": "right_click", "coordinate": [x, y]}
  {"action": "double_click","coordinate": [x, y]}
  {"action": "mouse_move",  "coordinate": [x, y]}
  {"action": "type",        "text": "string to type"}
  {"action": "key",         "text": "key name, e.g. Return, Tab, Escape, cmd+c"}
  {"action": "scroll",      "coordinate": [x, y], "direction": "up"|"down", "amount": 3}
  {"action": "screenshot"}
  {"action": "done",        "result": "brief summary of what was accomplished"}

Rules:
- Coordinates are pixels from the top-left of the screen.
- Use "done" only when the task is fully complete or impossible.
- When unsure what is on screen, use "screenshot" to get a fresh view.
- Prefer clicking visible UI elements over keyboard shortcuts.
- On macOS, use cmd (not ctrl) for copy/paste/etc.
"""


def computer_use(task: str) -> str:
    """Execute a desktop task using Gemini Vision and pyautogui.

    Takes screenshots, sends them to Gemini, receives structured actions
    (click, type, scroll, key), executes them, and repeats until done.
    No additional API key required — uses existing GEMINI_API_KEY.

    Always confirm with the user before invoking — WRITE-tier action.
    """
    try:
        from google import genai
        from google.genai import types as _gtypes
    except ImportError:
        return "[ERROR: google-genai not installed]"

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "[ERROR: GEMINI_API_KEY not set in environment]"

    client = genai.Client(api_key=api_key)
    screen_w, screen_h = _screen_size()

    from src import observability as _obs

    # Seed: take first screenshot and start the loop
    screenshot_data = _take_screenshot_b64()
    history: list[dict[str, Any]] = []

    for iteration in range(_MAX_ITER):
        # Build the prompt for this iteration
        if iteration == 0:
            user_text = (
                f"Screen resolution: {screen_w}x{screen_h}.\n"
                f"Task: {task}\n\n"
                "Look at the screenshot and return the next action as JSON."
            )
        else:
            user_text = "Action executed. Here is the updated screenshot. Return the next action as JSON."

        parts: list[Any] = [_gtypes.Part(text=user_text)]
        if screenshot_data:
            parts.append(_gtypes.Part(
                inline_data=_gtypes.Blob(
                    mime_type="image/png",
                    data=base64.b64decode(screenshot_data),
                )
            ))

        history.append({"role": "user", "parts": parts})

        contents = [_gtypes.Content(role=m["role"], parts=m["parts"]) for m in history]
        response = client.models.generate_content(
            model=_MODEL,
            contents=contents,
            config=_gtypes.GenerateContentConfig(
                system_instruction=_SYSTEM,
                temperature=0.1,
            ),
        )

        try:
            meta = response.usage_metadata
            _obs.log_api_call(
                _MODEL,
                meta.prompt_token_count or 0,
                meta.candidates_token_count or 0,
                source="computer_use",
            )
        except Exception:
            pass

        raw = (response.text or "").strip()

        # Append model response to history
        history.append({"role": "model", "parts": [_gtypes.Part(text=raw)]})

        # Parse the action JSON
        action = _parse_action(raw)
        if action is None:
            return f"[computer_use: could not parse action from model response: {raw[:200]}]"

        if action["action"] == "done":
            return action.get("result", "Task completed.")

        screenshot_data = _execute_action(action)

    return f"[computer_use: stopped after {_MAX_ITER} iterations without completing the task]"


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def _parse_action(text: str) -> dict[str, Any] | None:
    """Extract a JSON action dict from the model response."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Find the first {...} block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------

def _execute_action(action: dict[str, Any]) -> str | None:
    """Execute one action; return new screenshot b64 or None on failure."""
    action_type = action.get("action", "")

    try:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05

        if action_type == "screenshot":
            pass

        elif action_type == "mouse_move":
            x, y = action["coordinate"]
            pyautogui.moveTo(x, y, duration=0.15)

        elif action_type == "left_click":
            x, y = action["coordinate"]
            pyautogui.click(x, y)

        elif action_type == "right_click":
            x, y = action["coordinate"]
            pyautogui.rightClick(x, y)

        elif action_type == "double_click":
            x, y = action["coordinate"]
            pyautogui.doubleClick(x, y)

        elif action_type == "type":
            pyautogui.write(action.get("text", ""), interval=0.03)

        elif action_type == "key":
            _press_key(action.get("text", ""))

        elif action_type == "scroll":
            x, y = action["coordinate"]
            direction = action.get("direction", "down")
            amount = int(action.get("amount", 3))
            pyautogui.moveTo(x, y, duration=0.1)
            pyautogui.scroll(-amount if direction == "down" else amount)

        time.sleep(_ACTION_PAUSE)

    except ImportError:
        return None
    except Exception:
        time.sleep(_ACTION_PAUSE)

    return _take_screenshot_b64()


def _press_key(key_text: str) -> None:
    """Press a key or combo, mapping macOS conventions."""
    import pyautogui

    _MAP = {
        "Return": "enter",    "return": "enter",
        "Escape": "esc",      "escape": "esc",
        "Tab": "tab",         "Backspace": "backspace",
        "Delete": "delete",
        "Page_Up": "pageup",  "Page_Down": "pagedown",
        "Home": "home",       "End": "end",
        "Up": "up",           "Down": "down",
        "Left": "left",       "Right": "right",
        "space": "space",
    }

    if "+" in key_text:
        parts = [_MAP.get(p, p.lower()) for p in key_text.split("+")]
        pyautogui.hotkey(*parts)
    else:
        pyautogui.press(_MAP.get(key_text, key_text.lower()))


# ---------------------------------------------------------------------------
# Screenshot helper
# ---------------------------------------------------------------------------

def _take_screenshot_b64() -> str | None:
    """Capture screen with macOS screencapture; return base64 PNG string."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        result = subprocess.run(
            ["screencapture", "-x", "-t", "png", path],
            capture_output=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    except Exception:
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _screen_size() -> tuple[int, int]:
    """Return (width, height) of the primary display."""
    try:
        import pyautogui
        return pyautogui.size()
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True, timeout=4,
        )
        for line in out.splitlines():
            if "Resolution:" in line:
                parts = line.strip().split()
                return int(parts[1]), int(parts[3])
    except Exception:
        pass
    return 1920, 1080
