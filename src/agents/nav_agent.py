"""
Computer Navigation Agent — free-only stack.

Two execution paths, zero paid APIs:

  Browser path  — browser-use (MIT) + Gemini Flash free tier + Playwright
                  Best for: Gmail, Calendar, GitHub, Notion, Slack, finance, any website
                  Install: pip install browser-use && playwright install chromium

  Desktop path  — Gemini Flash vision (free tier) + pyautogui
                  Best for: native macOS apps (Finder, Terminal, Mail, custom apps)
                  Uses: screencapture + AppleScript AX tree + JSON action loop

Three use-cases share one run() call:
  1. Autonomous computer agent  — default
  2. Wearable AI (Frame glasses) — pass first_person_frame=<jpeg bytes>
  3. Robot learning              — pass robot_camera_frame=<jpeg bytes>, record=True

What to keep as direct connectors (nav agent is NOT the right tool):
  - Smart home / Home Assistant  — sub-300ms, event-driven
  - Apple Health streaming       — continuous sensor data
  - Push / iOS notifications     — event-driven, no screen
  - Car / wearable biometrics    — raw sensor streams
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Optional

import pyautogui
from google import genai
from google.genai import types as gtypes
from PIL import Image

logger = logging.getLogger(__name__)

# ── Gemini free-tier config ──────────────────────────────────────────────────
_GEMINI_KEY   = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
_VISION_MODEL = os.getenv("VISION_MODEL", "models/gemini-2.0-flash")   # free tier
_NAV_MAX_ITER = int(os.getenv("NAV_MAX_ITER", "20"))

# Resize target fed to Gemini (keeps token count + free-tier quota down)
_DISPLAY_W, _DISPLAY_H = 1280, 800

_TMP       = Path("/tmp")
_SHOT_RAW  = _TMP / "nav_raw.png"
_SHOT_FEED = _TMP / "nav_feed.png"

# Keywords whose presence in an action triggers HITL confirmation
_WRITE_KEYWORDS = frozenset({
    "send", "submit", "delete", "purchase", "pay", "transfer",
    "post", "publish", "confirm", "book", "schedule", "cancel",
    "sign", "order", "checkout", "remove", "unsubscribe",
})

# Tasks containing these strings are routed to the browser path
_BROWSER_SIGNALS = frozenset({
    "gmail", "google", "chrome", "safari", "browser", "website", "web",
    "http", "url", ".com", ".org", "notion.so", "github.com",
    "slack.com", "calendar", "drive", "docs", "sheets",
})

# App-specific prompting hints to reduce iterations
_APP_HINTS: dict[str, str] = {
    "Mail":          "Apple Mail: ⌘N = new, ⌘R = reply, ⌘⇧D = send.",
    "Safari":        "Safari: ⌘L = focus address bar, ⌘T = new tab.",
    "Google Chrome": "Chrome: ⌘L = address bar, ⌘T = new tab.",
    "Slack":         "Slack: ⌘K = jump to channel/DM, ⌘/ = shortcuts.",
    "Notion":        "Notion: ⌘P = search, / at line start = block menu.",
    "Calendar":      "Calendar: ⌘N = new event.",
    "Terminal":      "Terminal: type command then press Return.",
    "Finder":        "Finder: ⌘⇧G = Go To Folder, ⌘Space = Spotlight.",
    "Xcode":         "Xcode: ⌘B = build, ⌘R = run, ⌘⇧F = find in project.",
}


# ── Action schema (returned by Gemini vision as JSON) ────────────────────────

class _Act(str, Enum):
    click        = "click"
    double_click = "double_click"
    right_click  = "right_click"
    type_text    = "type_text"
    key_press    = "key_press"
    scroll       = "scroll"
    drag         = "drag"
    done         = "done"
    wait         = "wait"


_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action":    {"type": "string",
                      "enum": [a.value for a in _Act],
                      "description": "The action to execute."},
        "x":         {"type": "integer", "description": "X pixel coordinate (0–1280)."},
        "y":         {"type": "integer", "description": "Y pixel coordinate (0–800)."},
        "text":      {"type": "string",  "description": "Text to type or key combo (e.g. 'cmd+r')."},
        "direction": {"type": "string",  "enum": ["up", "down"],
                      "description": "Scroll direction."},
        "amount":    {"type": "integer", "description": "Scroll clicks (1-10)."},
        "start_x":   {"type": "integer", "description": "Drag start X."},
        "start_y":   {"type": "integer", "description": "Drag start Y."},
        "end_x":     {"type": "integer", "description": "Drag end X."},
        "end_y":     {"type": "integer", "description": "Drag end Y."},
        "result":    {"type": "string",  "description": "Summary when action=done."},
        "reason":    {"type": "string",  "description": "One-line reason for this action."},
    },
    "required": ["action"],
}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class NavResult:
    status: str              # "done" | "hitl_paused" | "max_iter" | "error"
    result: str
    steps: list[dict] = field(default_factory=list)
    pending_action: Optional[dict] = None


# ── Nav Agent ─────────────────────────────────────────────────────────────────

class NavAgent:
    """
    Computer navigation agent — 100% free stack.
    Gemini Flash vision (free tier) + browser-use (MIT) + pyautogui (MIT).
    """

    def __init__(self) -> None:
        self._gemini = genai.Client(api_key=_GEMINI_KEY)

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(
        self,
        task: str,
        context: str = "",
        *,
        hitl_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
        first_person_frame: Optional[bytes] = None,
        robot_camera_frame: Optional[bytes] = None,
        record: bool = False,
    ) -> NavResult:
        """
        Execute a task via screen navigation using only free resources.

        Routes automatically:
          - Web / browser tasks → browser-use + Gemini Flash
          - Native desktop tasks → Gemini vision loop + pyautogui
        """
        use_browser = self._is_browser_task(task)

        if use_browser:
            return await self._run_browser(task, context, hitl_callback, record)
        else:
            return await self._run_desktop(
                task, context,
                hitl_callback=hitl_callback,
                first_person_frame=first_person_frame,
                robot_camera_frame=robot_camera_frame,
                record=record,
            )

    # ── Browser path (browser-use + Gemini Flash, MIT/free) ──────────────────

    async def _run_browser(
        self,
        task: str,
        context: str,
        hitl_callback: Optional[Callable[[str], Awaitable[bool]]],
        record: bool,
    ) -> NavResult:
        """browser-use library with Gemini Flash backend."""
        try:
            # browser-use uses LangChain's LLM interface; use langchain-google-genai
            from browser_use import Agent as BrowserAgent
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            logger.warning(
                "browser-use not installed — falling back to desktop vision path. "
                "Run: pip install browser-use langchain-google-genai && playwright install chromium"
            )
            return await self._run_desktop(
                task, context,
                hitl_callback=hitl_callback,
                first_person_frame=None,
                robot_camera_frame=None,
                record=record,
            )

        full_task = task
        if context:
            full_task = f"{task}\n\nContext about the user:\n{context}"

        llm = ChatGoogleGenerativeAI(
            model=_VISION_MODEL.replace("models/", ""),  # langchain wants bare name
            google_api_key=_GEMINI_KEY,
        )

        agent = BrowserAgent(task=full_task, llm=llm)

        try:
            result = await agent.run(max_steps=_NAV_MAX_ITER)
            final_text = str(result) if result else "Browser task completed."
            return NavResult(status="done", result=final_text)
        except Exception as exc:
            logger.error("browser-use failed: %s", exc)
            return NavResult(status="error", result=str(exc))

    # ── Desktop vision path (Gemini Flash free tier + pyautogui) ─────────────

    async def _run_desktop(
        self,
        task: str,
        context: str,
        *,
        hitl_callback: Optional[Callable[[str], Awaitable[bool]]],
        first_person_frame: Optional[bytes],
        robot_camera_frame: Optional[bytes],
        record: bool,
    ) -> NavResult:
        """
        Perceive-decide-act loop.
          1. Capture screenshot + AX tree summary
          2. Ask Gemini Flash: 'what is the next action?' → JSON
          3. HITL gate for irreversible actions
          4. Execute via pyautogui
          5. Repeat
        """
        steps: list[dict] = []
        demos: list[dict] = []
        history: list[dict] = []     # conversation turns for context continuity

        system_prompt = self._build_system_prompt(task, context)

        for iteration in range(_NAV_MAX_ITER):
            # 1. Perceive
            shot_bytes  = self._capture_screenshot()
            app_name    = self._get_frontmost_app()
            ax_summary  = self._get_ax_summary(app_name)

            # 2. Build vision prompt
            parts = self._build_parts(
                shot_bytes, ax_summary, task, history,
                first_person_frame if iteration == 0 else None,
                robot_camera_frame if iteration == 0 else None,
            )

            # 3. Ask Gemini Flash for next action (structured JSON output)
            try:
                response = self._gemini.models.generate_content(
                    model=_VISION_MODEL,
                    contents=[gtypes.Content(role="user", parts=parts)],
                    config=gtypes.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=_ACTION_SCHEMA,
                        temperature=0.1,
                    ),
                )
                action = json.loads(response.text)
            except Exception as exc:
                logger.error("Gemini vision call failed (iter %d): %s", iteration, exc)
                return NavResult(status="error", result=str(exc), steps=steps)

            action_type = action.get("action", "wait")
            reason      = action.get("reason", "")
            logger.debug("NavAgent iter %d — %s (%s)", iteration, action_type, reason)

            # 4. Done?
            if action_type == _Act.done:
                if record:
                    self._save_demos(task, demos)
                return NavResult(
                    status="done",
                    result=action.get("result", "Task completed."),
                    steps=steps,
                )

            # 5. HITL gate for write/irreversible actions
            if hitl_callback and self._needs_confirmation(action):
                description = self._action_description(action)
                approved = await hitl_callback(description)
                if not approved:
                    return NavResult(
                        status="hitl_paused",
                        result=f"Cancelled: {description}",
                        steps=steps,
                        pending_action=action,
                    )

            # 6. Execute
            before_b64 = base64.b64encode(shot_bytes).decode() if record else ""
            outcome = await self._execute(action)
            await asyncio.sleep(0.35)   # UI settle time

            step = {
                "iteration": iteration,
                "action": action_type,
                "reason": reason,
                "outcome": outcome,
                "app": app_name,
            }
            steps.append(step)

            if record:
                demos.append({
                    "before_screenshot_b64": before_b64,
                    "action": action,
                    "app": app_name,
                    "timestamp": time.time(),
                })

            # Update conversation history for next iteration
            history.append({"action": action_type, "outcome": outcome, "app": app_name})

        if record:
            self._save_demos(task, demos)

        return NavResult(
            status="max_iter",
            result=f"Reached {_NAV_MAX_ITER}-step limit without completing the task.",
            steps=steps,
        )

    # ── Perceive ──────────────────────────────────────────────────────────────

    def _capture_screenshot(self) -> bytes:
        """screencapture → resize to _DISPLAY_W×_DISPLAY_H → PNG bytes."""
        subprocess.run(
            ["screencapture", "-x", "-t", "png", str(_SHOT_RAW)],
            capture_output=True, check=False,
        )
        img = Image.open(_SHOT_RAW)
        img = img.resize((_DISPLAY_W, _DISPLAY_H), Image.LANCZOS)
        img.save(_SHOT_FEED, format="PNG")
        return _SHOT_FEED.read_bytes()

    def _get_frontmost_app(self) -> str:
        script = (
            'tell application "System Events" to '
            'return name of first application process whose frontmost is true'
        )
        r = subprocess.run(["osascript", "-e", script],
                           capture_output=True, text=True, timeout=2)
        return r.stdout.strip() or "Unknown"

    def _get_ax_summary(self, app: str) -> str:
        """Fast AppleScript accessibility snapshot — Phase 2 will use full pyobjc AXUIElement."""
        script = f'''
        tell application "System Events"
            tell process "{app}"
                try
                    set fe to value of attribute "AXFocusedUIElement"
                    set feRole to role of fe
                    set feDesc to description of fe
                    return feRole & ": " & feDesc
                on error
                    return "no focused element"
                end try
            end tell
        end tell
        '''
        r = subprocess.run(["osascript", "-e", script],
                           capture_output=True, text=True, timeout=3)
        focused = r.stdout.strip()[:100] or "none"
        app_hint = _APP_HINTS.get(app, "")
        hint_part = f" | Hint: {app_hint}" if app_hint else ""
        return f"App: {app} | Focused: {focused}{hint_part}"

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_system_prompt(self, task: str, context: str) -> str:
        prompt = (
            "You are a macOS desktop automation agent. "
            "You receive a screenshot (resized to 1280×800) and control "
            "the computer via mouse and keyboard actions.\n\n"
            "Rules:\n"
            "- Return exactly one action per response as JSON matching the schema.\n"
            "- Coordinates (x, y) must be within 0–1280 (x) and 0–800 (y).\n"
            "- Before any irreversible action (send, submit, delete, purchase, book), "
            "  set action=wait and explain in 'reason' — the user will confirm.\n"
            "- When the task is fully complete, set action=done and summarise in 'result'.\n"
            "- Be precise with coordinates. Click the centre of the target element.\n"
        )
        if context:
            prompt += f"\nUser context:\n{context}\n"
        prompt += f"\nTask to complete: {task}"
        return prompt

    def _build_parts(
        self,
        shot_bytes: bytes,
        ax_summary: str,
        task: str,
        history: list[dict],
        first_person_frame: Optional[bytes],
        robot_camera_frame: Optional[bytes],
    ) -> list[gtypes.Part]:
        parts: list[gtypes.Part] = []

        # Wearable first-person context (first iteration only)
        if first_person_frame:
            parts.append(gtypes.Part.from_bytes(data=first_person_frame, mime_type="image/jpeg"))
            parts.append(gtypes.Part.from_text(text="[Wearable camera — what the user physically sees]\n"))

        # Robot camera (first iteration only)
        if robot_camera_frame:
            parts.append(gtypes.Part.from_bytes(data=robot_camera_frame, mime_type="image/jpeg"))
            parts.append(gtypes.Part.from_text(text="[Robot camera]\n"))

        # Current screenshot
        parts.append(gtypes.Part.from_bytes(data=shot_bytes, mime_type="image/png"))

        # Accessibility + history context
        state_text = f"Screen state: {ax_summary}\n"
        if history:
            recent = history[-4:]   # last 4 steps as context
            hist_lines = " → ".join(f"{s['action']}({s['app']})" for s in recent)
            state_text += f"Recent steps: {hist_lines}\n"
        state_text += "What is the next single action to complete the task? Return JSON only."

        parts.append(gtypes.Part.from_text(text=state_text))
        return parts

    # ── Execute ───────────────────────────────────────────────────────────────

    async def _execute(self, action: dict) -> str:
        t = action.get("action", "wait")

        if t == _Act.click:
            pyautogui.click(action["x"], action["y"])
            return f"Clicked ({action['x']}, {action['y']})."

        elif t == _Act.double_click:
            pyautogui.doubleClick(action["x"], action["y"])
            return f"Double-clicked ({action['x']}, {action['y']})."

        elif t == _Act.right_click:
            pyautogui.click(action["x"], action["y"], button="right")
            return f"Right-clicked ({action['x']}, {action['y']})."

        elif t == _Act.type_text:
            text = action.get("text", "")
            pyautogui.write(text, interval=0.02)
            return f"Typed: {text[:60]}{'…' if len(text) > 60 else ''}"

        elif t == _Act.key_press:
            chord = action.get("text", "")
            # Normalise: "cmd+r" → ["cmd", "r"]
            keys = [k.strip() for k in re.split(r"[+\-]", chord)]
            pyautogui.hotkey(*keys)
            return f"Pressed: {chord}."

        elif t == _Act.scroll:
            x, y = action.get("x", 640), action.get("y", 400)
            direction = action.get("direction", "down")
            amount = int(action.get("amount", 3))
            delta = amount if direction == "up" else -amount
            pyautogui.scroll(delta, x=x, y=y)
            return f"Scrolled {direction} {amount}× at ({x}, {y})."

        elif t == _Act.drag:
            pyautogui.mouseDown(action["start_x"], action["start_y"])
            await asyncio.sleep(0.1)
            pyautogui.moveTo(action["end_x"], action["end_y"], duration=0.35)
            pyautogui.mouseUp()
            return (f"Dragged ({action['start_x']},{action['start_y']}) "
                    f"→ ({action['end_x']},{action['end_y']}).")

        elif t == _Act.wait:
            await asyncio.sleep(1.0)
            return "Waited 1 second."

        return f"Unknown action '{t}'."

    # ── HITL helpers ──────────────────────────────────────────────────────────

    def _needs_confirmation(self, action: dict) -> bool:
        payload = json.dumps(action).lower()
        return any(kw in payload for kw in _WRITE_KEYWORDS)

    def _action_description(self, action: dict) -> str:
        t = action.get("action", "")
        if t == _Act.type_text:
            return f'Type: "{action.get("text", "")[:80]}"'
        if t == _Act.key_press:
            return f'Press shortcut: {action.get("text", "")}'
        if "x" in action and "y" in action:
            return f"{t} at ({action['x']}, {action['y']})"
        return str(action)

    # ── Routing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_browser_task(task: str) -> bool:
        """Heuristic: route to browser-use if the task is clearly web-based."""
        t = task.lower()
        return any(sig in t for sig in _BROWSER_SIGNALS)

    # ── Demo recording (robot learning) ──────────────────────────────────────

    def _save_demos(self, task: str, demos: list[dict]) -> None:
        out_dir = Path("data/demonstrations")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"demo_{int(time.time())}.json"
        slim = [{k: v for k, v in d.items() if k != "before_screenshot_b64"} for d in demos]
        out_file.write_text(json.dumps({"task": task, "steps": len(demos), "demos": slim}, indent=2))
        logger.info("NavAgent: saved %d demo steps → %s", len(demos), out_file)


# ── LangGraph tool wrapper ────────────────────────────────────────────────────

_agent: Optional[NavAgent] = None


def _get_agent() -> NavAgent:
    global _agent
    if _agent is None:
        _agent = NavAgent()
    return _agent


async def navigate_computer(
    task: str,
    context: str = "",
    hitl_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
) -> str:
    """
    LangGraph tool — navigate the computer to complete any task.

    Automatically routes:
      • Browser/web tasks  → browser-use (MIT) + Gemini Flash free tier
      • Native macOS apps  → Gemini vision loop (free) + pyautogui

    Use for: email, calendar, docs, GitHub, Notion, Slack, any app on screen.
    Do NOT use for: smart home, health streams, push notifications (direct connectors).

    Args:
        task:           Natural language task description.
        context:        Optional user context (name, preferences).
        hitl_callback:  Async fn(description) → bool; gates irreversible actions.

    Returns:
        Plain-text summary of what was accomplished.
    """
    result = await _get_agent().run(task, context, hitl_callback=hitl_callback)
    return f"[{result.status}] {result.result} ({len(result.steps)} steps)"
