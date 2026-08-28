"""
Computer Agent connector — Claude Computer Use API.

Exposes one tool: computer_use(task) — a full agentic loop that lets Claude
take screenshots, move the mouse, click, type, and scroll until the task is
complete. Uses pyautogui for mouse/keyboard control and macOS screencapture
for screenshots.

Safety: registered as a WRITE-tier tool so every invocation goes through the
HITL interrupt gate before executing. Requires explicit user approval.

Required env vars:
  ANTHROPIC_API_KEY  — Anthropic API key (separate from the Gemini key)

Optional env vars:
  CLAUDE_COMPUTER_MODEL  — defaults to claude-opus-4-8
"""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import time
from typing import Any

_MODEL = os.environ.get("CLAUDE_COMPUTER_MODEL", "claude-opus-4-8")
_MAX_ITER = 20
_ACTION_PAUSE = 0.4   # seconds between actions


def computer_use(task: str) -> str:
    """Execute a desktop task using Claude Computer Use.

    Claude takes control of the screen: takes screenshots, clicks, types, and
    navigates apps until the task is complete. Requires ANTHROPIC_API_KEY.
    Always confirm with the user before invoking — this is a write-tier action.
    """
    try:
        import anthropic
    except ImportError:
        return "[ERROR: anthropic not installed. Run: pip install anthropic]"

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "[ERROR: ANTHROPIC_API_KEY not set in environment]"

    screen_w, screen_h = _screen_size()
    client = anthropic.Anthropic(api_key=api_key)

    tools: list[dict[str, Any]] = [
        {
            "type": "computer_20241022",
            "name": "computer",
            "display_width_px": screen_w,
            "display_height_px": screen_h,
        }
    ]

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": task}
    ]

    from src import observability as _obs

    for _ in range(_MAX_ITER):
        response = client.beta.messages.create(
            model=_MODEL,
            max_tokens=4096,
            tools=tools,
            messages=messages,
            betas=["computer-use-2024-10-22"],
        )

        # Track cost in the api_usage table
        try:
            meta = response.usage
            _obs.log_api_call(
                _MODEL,
                getattr(meta, "input_tokens", 0),
                getattr(meta, "output_tokens", 0),
                source="computer_use",
            )
        except Exception:
            pass

        if response.stop_reason == "end_turn":
            texts = [
                b.text for b in response.content if hasattr(b, "text")
            ]
            return "\n".join(texts) or "Task completed."

        # Append model's turn
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result_content = _execute_action(block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_content,
            })

        if not tool_results:
            break

        messages.append({"role": "user", "content": tool_results})

    return f"[computer_use: stopped after {_MAX_ITER} iterations]"


# ---------------------------------------------------------------------------
# Action dispatcher
# ---------------------------------------------------------------------------

def _execute_action(action: dict[str, Any]) -> list[dict[str, Any]]:
    """Dispatch one computer action and return a screenshot as the result."""
    action_type = action.get("action", "")

    try:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05

        if action_type == "screenshot":
            pass  # just take screenshot below

        elif action_type == "mouse_move":
            x, y = action["coordinate"]
            pyautogui.moveTo(x, y, duration=0.15)

        elif action_type == "left_click":
            x, y = action["coordinate"]
            pyautogui.click(x, y)

        elif action_type == "right_click":
            x, y = action["coordinate"]
            pyautogui.rightClick(x, y)

        elif action_type == "middle_click":
            x, y = action["coordinate"]
            pyautogui.middleClick(x, y)

        elif action_type == "double_click":
            x, y = action["coordinate"]
            pyautogui.doubleClick(x, y)

        elif action_type == "left_click_drag":
            sx, sy = action["start_coordinate"]
            ex, ey = action["coordinate"]
            pyautogui.moveTo(sx, sy, duration=0.1)
            pyautogui.dragTo(ex, ey, duration=0.4, button="left")

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

        elif action_type == "cursor_position":
            cx, cy = pyautogui.position()
            return [{"type": "text", "text": f"Cursor at ({cx}, {cy})"}]

        time.sleep(_ACTION_PAUSE)

    except ImportError:
        return [{"type": "text", "text": "[ERROR: pyautogui not installed. Run: pip install pyautogui]"}]
    except Exception as exc:
        return [{"type": "text", "text": f"[Action error: {exc}]"}]

    return _take_screenshot()


def _press_key(key_text: str) -> None:
    """Press a key or key combo; translates Anthropic key names to pyautogui."""
    import pyautogui

    _MAP = {
        "Return": "enter", "return": "enter",
        "Escape": "esc",   "escape": "esc",
        "Tab": "tab",      "Backspace": "backspace",
        "Delete": "delete",
        "Page_Up": "pageup", "Page_Down": "pagedown",
        "Home": "home",    "End": "end",
        "Up": "up",        "Down": "down",
        "Left": "left",    "Right": "right",
        "F1": "f1", "F2": "f2", "F3": "f3", "F4": "f4",
        "F5": "f5", "F6": "f6", "F7": "f7", "F8": "f8",
        "F9": "f9", "F10": "f10", "F11": "f11", "F12": "f12",
        # macOS uses cmd for ctrl shortcuts
        "ctrl+c": "command+c",  "ctrl+v": "command+v",
        "ctrl+a": "command+a",  "ctrl+z": "command+z",
        "ctrl+x": "command+x",  "ctrl+s": "command+s",
        "ctrl+w": "command+w",  "ctrl+t": "command+t",
    }

    mapped = _MAP.get(key_text)
    if mapped:
        if "+" in mapped:
            pyautogui.hotkey(*mapped.split("+"))
        else:
            pyautogui.press(mapped)
    elif "+" in key_text:
        parts = [_MAP.get(p, p.lower()) for p in key_text.split("+")]
        pyautogui.hotkey(*parts)
    else:
        pyautogui.press(key_text.lower())


# ---------------------------------------------------------------------------
# Screenshot helper
# ---------------------------------------------------------------------------

def _take_screenshot() -> list[dict[str, Any]]:
    """Capture the screen with macOS screencapture; return base64 image block."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        result = subprocess.run(
            ["screencapture", "-x", "-t", "png", path],
            capture_output=True, timeout=5,
        )
        if result.returncode != 0:
            return [{"type": "text", "text": "[Screenshot failed]"}]

        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")

        return [{"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": data,
        }}]
    except Exception as exc:
        return [{"type": "text", "text": f"[Screenshot error: {exc}]"}]
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
    # Fallback via system_profiler on macOS
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True, timeout=4,
        )
        for line in out.splitlines():
            if "Resolution:" in line:
                parts = line.strip().split()
                return int(parts[1]), int(parts[3])
    except Exception:
        pass
    return 1920, 1080
