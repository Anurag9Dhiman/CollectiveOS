"""
Computer Navigation Agent — free-only stack (Phase 2).

Two execution paths, zero paid APIs:

  Browser path  — browser-use (MIT) + Gemini Flash free tier + Playwright
                  Best for: Gmail, Calendar, GitHub, Notion, Slack, finance, any website
                  Install: pip install browser-use && playwright install chromium

  Desktop path  — Gemini Flash vision (free tier) + pyautogui
                  Phase 2 improvements vs Phase 1:
                    • Full AXUIElement accessibility tree via pyobjc (40-100× more context
                      than a single focused-element AppleScript call; falls back gracefully)
                    • Coordinate scaling: Gemini sees 1280×800; pyautogui uses logical
                      screen coords — scale factor computed lazily from pyautogui.size()
                    • OpenCV screen-diff verification: detects when an action had no effect
                      and informs Gemini so it can retry or take a different approach

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


def _compress_for_stream(png_bytes: bytes, quality: int = 40) -> str:
    """JPEG-compress a PNG frame and return it base64-encoded for SSE streaming."""
    import io as _io
    buf = _io.BytesIO()
    Image.open(_io.BytesIO(png_bytes)).save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ── Optional Phase-2 dependencies (graceful fallback if not installed) ────────
try:
    import ApplicationServices as _AS   # pyobjc-framework-ApplicationServices
    import AppKit as _AK                # pyobjc-framework-AppKit
    _PYOBJC_OK = True
except ImportError:
    _PYOBJC_OK = False
    logger.debug("pyobjc not available — AX tree falls back to AppleScript (Phase 1 mode)")

try:
    import cv2 as _cv2
    import numpy as _np
    _CV2_OK = True
except ImportError:
    _CV2_OK = False
    logger.debug("opencv-python not available — screen-change verification disabled")

# ── Gemini free-tier config ───────────────────────────────────────────────────
_GEMINI_KEY   = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
_VISION_MODEL = os.getenv("VISION_MODEL", "models/gemini-2.0-flash")   # free tier
_NAV_MAX_ITER = int(os.getenv("NAV_MAX_ITER", "20"))

# Gemini sees a screenshot resized to this resolution
_DISPLAY_W, _DISPLAY_H = 1280, 800

_TMP       = Path("/tmp")
_SHOT_RAW  = _TMP / "nav_raw.png"
_SHOT_FEED = _TMP / "nav_feed.png"

# ── Coordinate scaling ────────────────────────────────────────────────────────
# Gemini returns coords in 1280×800 screenshot space.
# pyautogui needs logical screen coords (e.g. 1512×982 on a Retina MacBook).
# We compute one scale factor per process and cache it.

_scale_cache: Optional[tuple[float, float]] = None


def _get_scale() -> tuple[float, float]:
    """
    Returns (sx, sy) where sx = _DISPLAY_W / screen_logical_width.
    Multiply by sx/sy to go screenshot→logical; divide to go logical→screenshot.
    """
    global _scale_cache
    if _scale_cache is None:
        try:
            sw, sh = pyautogui.size()
            _scale_cache = (_DISPLAY_W / sw, _DISPLAY_H / sh)
        except Exception:
            _scale_cache = (1.0, 1.0)
    return _scale_cache


def _to_screen(sx_coord: int, sy_coord: int) -> tuple[int, int]:
    """Screenshot coords → actual logical screen coords for pyautogui."""
    sx, sy = _get_scale()
    return int(sx_coord / sx), int(sy_coord / sy)


# ── AX tree constants ─────────────────────────────────────────────────────────
_INTERACTIVE_ROLES = frozenset({
    "AXButton", "AXTextField", "AXTextArea", "AXLink",
    "AXCheckBox", "AXRadioButton", "AXMenuItem", "AXComboBox",
    "AXPopUpButton", "AXSlider", "AXCell", "AXStaticText",
    "AXMenuBarItem", "AXTabGroup", "AXSearchField",
})
_MAX_AX_ELEMS = 60
_MAX_AX_DEPTH = 5

# Keywords whose presence in an action triggers HITL confirmation
_WRITE_KEYWORDS = frozenset({
    "send", "submit", "delete", "purchase", "pay", "transfer",
    "post", "publish", "confirm", "book", "schedule", "cancel",
    "sign", "order", "checkout", "remove", "unsubscribe",
})

# Tasks containing these strings are routed to the browser path.
# Covers all web-navigable services that were previously separate connectors.
_BROWSER_SIGNALS = frozenset({
    # Explicit web intent
    "browser", "website", "web", "http", "url", "search the web",
    # URL patterns
    ".com", ".org", ".io", ".net",
    # Google services
    "gmail", "google", "google.com", "calendar", "drive", "docs", "sheets",
    "google calendar", "google drive", "google docs",
    # Productivity
    "notion", "notion.so", "todoist", "todoist.com", "trello", "linear",
    "jira", "confluence", "airtable",
    # Code
    "github", "github.com", "gitlab", "bitbucket", "stackoverflow",
    # Communication
    "slack", "slack.com", "discord", "discord.com", "telegram", "teams",
    # Social / media
    "twitter", "x.com", "linkedin", "reddit", "instagram", "facebook",
    "youtube", "youtube.com",
    # Shopping / finance
    "amazon", "amazon.com", "stripe", "paypal", "venmo", "robinhood",
    # Maps / travel
    "maps", "google maps", "airbnb", "booking.com", "uber", "lyft",
    # Browser apps
    "chrome", "safari", "firefox",
})

# App-specific keyboard-shortcut hints injected into each iteration's state text.
# Gemini uses these to prefer reliable shortcuts over visual hunting.
_APP_HINTS: dict[str, str] = {
    "Safari": (
        "⌘L: address bar | ⌘T: new tab | ⌘W: close tab | ⌘R: reload | "
        "⌘F: find on page | ⌘[: back | ⌘]: forward | ⌘⇧T: reopen tab | "
        "Space: scroll down | ⇧Space: scroll up | ⌘A: select all text in field"
    ),
    "Google Chrome": (
        "⌘L: address bar | ⌘T: new tab | ⌘W: close tab | ⌘R: reload | "
        "⌘F: find on page | ⌘[: back | ⌘]: forward | ⌘⇧T: reopen tab | "
        "⌘⇧J: downloads | ⌘⇧N: incognito | Space: scroll down"
    ),
    "Firefox": (
        "⌘L: address bar | ⌘T: new tab | ⌘W: close tab | ⌘R: reload | "
        "⌘F: find on page | ⌘[: back | ⌘]: forward | Space: scroll down"
    ),
    "Mail": (
        "⌘N: new message | ⌘R: reply | ⌘⇧R: reply all | ⌘⇧F: forward | "
        "⌘⇧D: send | ⌘⌫: delete | ⌘⇧U: mark as unread | "
        "⌘1: inbox | ⌘↑/↓: prev/next message | Space: scroll / next unread"
    ),
    "Finder": (
        "⌘⇧G: go to folder | ⌘⇧N: new folder | ⌘⌫: move to trash | "
        "Return: rename selected | Space: quick look | ⌘I: get info | "
        "⌘↑: enclosing folder | ⌘↓: open | ⌘1/2/3/4: icon/list/column/gallery | "
        "⌘F: find | ⌘A: select all"
    ),
    "Terminal": (
        "⌘T: new tab | ⌘N: new window | ⌘W: close tab | ⌘K: clear | "
        "ctrl+C: interrupt | ctrl+D: EOF | ctrl+L: clear screen | "
        "ctrl+A: line start | ctrl+E: line end | ↑/↓: command history | "
        "ctrl+R: search history | ⌘+: increase font"
    ),
    "iTerm2": (
        "⌘T: new tab | ⌘D: split vertically | ⌘⇧D: split horizontally | "
        "⌘W: close | ⌘K: clear buffer | ctrl+C: interrupt | ctrl+L: clear | "
        "⌘⇧I: broadcast input to panes | ⌘↑/↓/←/→: navigate panes"
    ),
    "Xcode": (
        "⌘B: build | ⌘R: run | ⌘.: stop | ⌘⇧K: clean | ⌘⇧O: open quickly | "
        "⌘L: go to line | ctrl+I: re-indent | ⌘/: comment/uncomment | "
        "⌘⇧F: find in project | ⌘⌃←/→: back/forward navigation"
    ),
    "Visual Studio Code": (
        "⌘P: quick open | ⌘⇧P: command palette | ⌘B: sidebar | "
        "⌘`: terminal | ⌘/: toggle comment | ⌘D: select next match | "
        "⌘⇧L: select all matches | ⌥↑/↓: move line | ⌘⇧F: find in files | "
        "⌘K ⌘S: shortcuts | ⌘⇧X: extensions"
    ),
    "Slack": (
        "⌘K: jump to channel or DM | ⌘⇧K: direct messages | "
        "⌘[: back | ⌘]: forward | ⌘F: search | ⌘⇧M: activity | "
        "⌘N: new message | E: edit last message (in message field) | "
        "↑: edit last message | ⌘⌥K: create channel"
    ),
    "Notion": (
        "⌘P: search / navigate | /: insert block at cursor | "
        "⌘⇧0-6: heading levels | ⌘⇧8: bulleted list | ⌘⇧9: numbered list | "
        "⌘⇧C: toggle code block | ⌘B: bold | ⌘I: italic | ⌘⇧L: link | "
        "⌘D: duplicate block | ⌘⌫: delete block"
    ),
    "Calendar": (
        "⌘N: new event | ⌘T: go to today | ⌘1: day view | ⌘2: week view | "
        "⌘3: month view | ⌘4: year view | ←/→: prev/next period | "
        "⌘⌫: delete event | ⌘E: edit event | ⌘F: search"
    ),
    "Notes": (
        "⌘N: new note | ⌘⌫: delete note | ⌘B: bold | ⌘I: italic | "
        "⌘U: underline | ⌘⇧L: checklist | ⌘⇧H: heading | ⌘F: find | "
        "⌘L: lock note | ⌘⇧A: add attachment"
    ),
    "Reminders": (
        "⌘N: new reminder | Return: confirm | ⌘⌫: delete | "
        "⌘I: get info / edit | Space: mark complete | ⌘⇧N: new list"
    ),
    "Messages": (
        "⌘N: new message | Return: send | ⇧Return: newline | "
        "⌘F: find | ⌘⌫: delete conversation | ⌘I: details"
    ),
    "FaceTime": (
        "⌘N: new call | Space: mute/unmute | ⌘M: mute | ⌘⇧F: full screen"
    ),
    "Spotify": (
        "Space: play/pause | ⌘→: next | ⌘←: previous | ⌘↑/↓: volume | "
        "⌘L: like song | ⌘F: search | ⌘1/2/3: home/search/library"
    ),
    "System Settings": (
        "⌘F: search settings | ⌘,: preferences (in most apps) | "
        "⌫: back | Return: select"
    ),
    "System Preferences": (
        "⌘F: search | ⌘L: main view | ←/→: back/forward | Return: open pane"
    ),
    "Preview": (
        "⌘⇧A: select all | ⌘C: copy | ⌘F: find | ⌘+/-: zoom | "
        "⌘⇧O: open | ⌘P: print | ⌘S: save | Space/⇧Space: scroll"
    ),
    "TextEdit": (
        "⌘B: bold | ⌘I: italic | ⌘U: underline | ⌘A: select all | "
        "⌘F: find | ⌘⇧T: rich/plain text toggle | ⌘S: save"
    ),
    "Photos": (
        "Return: open | ⌫: delete | ⌘⇧F: full screen | I: info | "
        "←/→: prev/next | ⌘+/-: zoom | ⌘E: edit | ⌘⇧E: export"
    ),
    "Music": (
        "Space: play/pause | ⌘→: next | ⌘←: previous | ⌘↑/↓: volume | "
        "⌘L: show in library | ⌘F: search | ⌘I: get info"
    ),
}


# ── Action schema (returned by Gemini vision as JSON) ─────────────────────────

class _Act(str, Enum):
    click          = "click"
    double_click   = "double_click"
    right_click    = "right_click"
    type_text      = "type_text"
    key_press      = "key_press"
    scroll         = "scroll"
    drag           = "drag"
    read_clipboard = "read_clipboard"
    done           = "done"
    wait           = "wait"


_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action":    {"type": "string",
                      "enum": [a.value for a in _Act],
                      "description": "The action to execute."},
        "x":         {"type": "integer", "description": "X pixel coordinate in screenshot space (0–1280)."},
        "y":         {"type": "integer", "description": "Y pixel coordinate in screenshot space (0–800)."},
        "text":      {"type": "string",  "description": "Text to type or key combo (e.g. 'cmd+r')."},
        "direction": {"type": "string",  "enum": ["up", "down"],
                      "description": "Scroll direction."},
        "amount":    {"type": "integer", "description": "Scroll clicks (1-10)."},
        "start_x":   {"type": "integer", "description": "Drag start X (screenshot space)."},
        "start_y":   {"type": "integer", "description": "Drag start Y (screenshot space)."},
        "end_x":     {"type": "integer", "description": "Drag end X (screenshot space)."},
        "end_y":     {"type": "integer", "description": "Drag end Y (screenshot space)."},
        "result":    {"type": "string",  "description": "Summary when action=done."},
        "reason":    {"type": "string",  "description": "One-line reason for this action."},
    },
    "required": ["action"],
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class NavResult:
    status: str              # "done" | "hitl_paused" | "max_iter" | "error"
    result: str
    steps: list[dict] = field(default_factory=list)
    pending_action: Optional[dict] = None
    verified: bool = True    # False only when verification explicitly failed all retries


# ── Nav Agent ─────────────────────────────────────────────────────────────────

class NavAgent:
    """
    Computer navigation agent — 100% free stack, Phase 2.
    Gemini Flash vision (free) + browser-use (MIT) + pyautogui (MIT).
    Phase 2 adds: pyobjc AX tree, OpenCV verification, coordinate scaling.
    """

    def __init__(self, genai_client: object | None = None) -> None:
        # Accept an injected client so tests can substitute a fake before the
        # real genai.Client is constructed. Created lazily when None.
        self._gemini = genai_client

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini is None:
            self._gemini = genai.Client(api_key=_GEMINI_KEY)
        return self._gemini

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
        if self._is_browser_task(task):
            return await self._run_browser(task, context, hitl_callback, record)
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

        from src import computer_stream as _cs
        run_id = _cs.begin_run(task)

        full_task = f"{task}\n\nUser context:\n{context}" if context else task
        llm = ChatGoogleGenerativeAI(
            model=_VISION_MODEL.replace("models/", ""),
            google_api_key=_GEMINI_KEY,
        )
        agent = BrowserAgent(task=full_task, llm=llm)
        try:
            result = await agent.run(max_steps=_NAV_MAX_ITER)
            result_str = str(result) or "Browser task completed."
            _cs.end_run(run_id, result_str, 0)
            nav_result = NavResult(status="done", result=result_str)
            _save_nav_memory(task, [], nav_result.result)
            return nav_result
        except Exception as exc:
            logger.error("browser-use failed: %s", exc)
            _cs.end_run(run_id, str(exc), 0)
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
        Perceive-decide-act loop (Phase 2).
          1. Capture screenshot + full AXUIElement tree
          2. Ask Gemini Flash for next action → JSON (structured output)
          3. HITL gate for irreversible actions
          4. Execute via pyautogui (with coordinate scaling)
          5. OpenCV verification: warn Gemini if screen didn't change
          6. Repeat
        """
        from src import computer_stream as _cs

        steps: list[dict] = []
        demos: list[dict] = []
        history: list[dict] = []
        stuck_count = 0
        verify_retries = 0   # how many times we've sent the agent back after a failed verify

        run_id = _cs.begin_run(task)

        # Pre-run planning: one text-only Gemini call to produce an ordered step list.
        # Emitted to the stream so the Computer panel can show the plan before execution.
        plan = self._plan_task(task, context)
        if plan:
            _cs.emit_plan(run_id, plan)
            logger.debug("NavAgent plan for %r: %s", task[:60], plan)

        system_prompt = self._build_system_prompt(task, context, plan=plan)

        for iteration in range(_NAV_MAX_ITER):
            if _cs.should_stop():
                _cs.end_run(run_id, "Stopped by user.", iteration)
                return NavResult(status="error", result="Stopped by user.", steps=steps)
            # 1. Perceive
            shot_bytes = self._capture_screenshot()
            app_name   = self._get_frontmost_app()
            ax_context = self._get_ax_tree(app_name)   # Phase 2: full AX tree

            # 2. Build prompt
            parts = self._build_parts(
                shot_bytes, ax_context, history,
                first_person_frame if iteration == 0 else None,
                robot_camera_frame if iteration == 0 else None,
                stuck_count=stuck_count,
            )

            # 3. Ask Gemini
            try:
                response = self._get_gemini_client().models.generate_content(
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
                logger.exception("Gemini vision call failed (iter %d)", iteration)
                _cs.end_run(run_id, str(exc), iteration)
                return NavResult(status="error", result=str(exc), steps=steps)

            action_type = action.get("action", "wait")
            reason      = action.get("reason", "")
            logger.debug("NavAgent iter %d — %s (%s)", iteration, action_type, reason)

            # 4. Done? Verify before committing.
            if action_type == _Act.done:
                claimed = action.get("result", "Task completed.")
                final_shot = self._capture_screenshot()
                verified, note = self._verify_completion(task, claimed, final_shot)

                if not verified and verify_retries < 2:
                    verify_retries += 1
                    logger.debug("NavAgent verification failed (retry %d): %s", verify_retries, note)
                    _cs.emit_action(run_id, iteration,
                                    {"action": "verify_failed", "reason": note}, None)
                    history.append({
                        "action": "verify_failed",
                        "outcome": (
                            f"VERIFICATION FAILED: {note} "
                            f"— task is not complete yet. Continue and try again."
                        ),
                        "app": app_name,
                        "screen_changed": False,
                    })
                    continue   # loop back — agent will see the failure note

                if record:
                    self._save_demos(task, demos)
                _cs.end_run(run_id, claimed, iteration)
                nav_result = NavResult(status="done", result=claimed,
                                       steps=steps, verified=verified)
                _save_nav_memory(task, steps, nav_result.result)
                return nav_result

            # 5. HITL gate
            if hitl_callback and self._needs_confirmation(action):
                description = self._action_description(action)
                if not await hitl_callback(description):
                    return NavResult(
                        status="hitl_paused",
                        result=f"Cancelled: {description}",
                        steps=steps,
                        pending_action=action,
                    )

            # 6. Execute
            before_b64 = base64.b64encode(shot_bytes).decode() if record else ""
            outcome = await self._execute(action)

            # 7. Wait for screen to stabilise (adaptive — up to 2 s)
            after_bytes = await self._wait_for_stable_screen()
            changed = self._verify_screen_change(shot_bytes, after_bytes)
            if not changed and action_type not in (_Act.type_text, _Act.key_press, _Act.wait):
                stuck_count += 1
                outcome += " [WARNING: screen unchanged — element may not be clickable or UI is loading]"
                logger.debug("NavAgent iter %d — screen unchanged after %s (stuck=%d)", iteration, action_type, stuck_count)
            else:
                stuck_count = 0

            # Compress post-action screenshot for live panel streaming
            try:
                screenshot_b64 = _compress_for_stream(after_bytes)
            except Exception:
                screenshot_b64 = None

            step = {
                "iteration": iteration,
                "action": action_type,
                "reason": reason,
                "outcome": outcome,
                "app": app_name,
                "screen_changed": changed,
            }
            steps.append(step)
            _cs.emit_action(run_id, iteration, action, screenshot_b64)

            if record:
                demos.append({
                    "before_screenshot_b64": before_b64,
                    "action": action,
                    "app": app_name,
                    "timestamp": time.time(),
                })

            history.append({
                "action": action_type,
                "outcome": outcome,
                "app": app_name,
                "screen_changed": changed,
            })

        if record:
            self._save_demos(task, demos)

        msg = f"Reached {_NAV_MAX_ITER}-step limit without completing the task."
        _cs.end_run(run_id, msg, _NAV_MAX_ITER)
        return NavResult(status="max_iter", result=msg, steps=steps)

    # ── Perceive ─────────────────────────────────────────────────────────────

    def _capture_screenshot(self) -> bytes:
        """screencapture (physical pixels) → resize to 1280×800 → PNG bytes."""
        subprocess.run(
            ["screencapture", "-x", "-t", "png", str(_SHOT_RAW)],
            capture_output=True, check=False,
        )
        img = Image.open(_SHOT_RAW)
        img = img.resize((_DISPLAY_W, _DISPLAY_H), Image.LANCZOS)
        img.save(_SHOT_FEED, format="PNG")
        return _SHOT_FEED.read_bytes()

    async def _wait_for_stable_screen(
        self,
        max_wait: float = 2.0,
        poll_interval: float = 0.2,
    ) -> bytes:
        """Poll until two consecutive screenshots look the same (screen has settled).

        Fast actions (keypress, click with instant response) settle in ~200 ms.
        Page loads, app launches, and animations are waited out up to max_wait seconds.

        Returns the final stable (or last captured) screenshot bytes.
        Falls back to a single-poll capture when OpenCV is unavailable.
        """
        await asyncio.sleep(poll_interval)
        prev = self._capture_screenshot()

        if not _CV2_OK:
            return prev   # no comparison possible — single poll is enough

        elapsed = poll_interval
        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            current = self._capture_screenshot()
            if not self._verify_screen_change(prev, current):
                return current   # stable: two frames matched
            prev = current

        logger.debug("_wait_for_stable_screen: screen still changing after %.1fs", max_wait)
        return prev

    def _get_frontmost_app(self) -> str:
        if _PYOBJC_OK:
            try:
                app = _AK.NSWorkspace.sharedWorkspace().frontmostApplication()
                return app.localizedName() or "Unknown"
            except Exception:
                pass
        # AppleScript fallback
        script = (
            'tell application "System Events" to '
            'return name of first application process whose frontmost is true'
        )
        r = subprocess.run(["osascript", "-e", script],
                           capture_output=True, text=True, timeout=2)
        return r.stdout.strip() or "Unknown"

    def _get_ax_tree(self, app: str) -> str:
        """
        Phase 2: Full AXUIElement tree via pyobjc.
        Falls back to AppleScript focused-element summary when pyobjc is unavailable.

        Returns a compact string of interactive elements with their center coordinates
        already scaled to the 1280×800 screenshot space, e.g.:
          AXButton[Send]@(940,752) AXTextField[To:]@(400,200) AXLink[Inbox]@(80,340)
        """
        if _PYOBJC_OK:
            try:
                return self._pyobjc_ax_tree(app)
            except Exception as exc:
                logger.debug("pyobjc AX tree failed (%s) — falling back to AppleScript", exc)

        return self._applescript_ax_summary(app)

    def _pyobjc_ax_tree(self, app: str) -> str:
        """Walk the live AXUIElement tree for the frontmost application."""
        pid = _AK.NSWorkspace.sharedWorkspace().frontmostApplication().processIdentifier()
        app_elem = _AS.AXUIElementCreateApplication(pid)

        sx, sy = _get_scale()   # screenshot / screen_logical
        results: list[str] = []
        self._walk_ax(app_elem, depth=0, results=results, sx=sx, sy=sy)

        hint = _APP_HINTS.get(app, "")
        tree_str = " ".join(results[:_MAX_AX_ELEMS]) or "no interactive elements found"
        hint_part = f" | Hint: {hint}" if hint else ""

        dialog_warning = self._detect_modal_dialog(app_elem)
        prefix = f"⚠️ {dialog_warning} | " if dialog_warning else ""
        return f"{prefix}App: {app} | AX elements: {tree_str}{hint_part}"

    def _detect_modal_dialog(self, app_elem: object) -> str:
        """Return a warning string if a modal dialog/sheet/alert is blocking the UI.

        Inspects AXWindows for AXSheet, AXAlert, or AXDialog containers and
        collects their title + button labels so Gemini knows exactly how to dismiss.
        Returns an empty string when no modal is found.
        """
        try:
            windows = self._ax_attr(app_elem, "AXWindows") or []
            for win in windows:
                # Sheets attach as children of AXWindow; alerts may be top-level
                candidates = [win] + list(self._ax_attr(win, "AXChildren") or [])
                for elem in candidates:
                    role = self._ax_attr(elem, "AXRole") or ""
                    if role not in ("AXSheet", "AXAlert", "AXDialog"):
                        continue

                    title = (self._ax_attr(elem, "AXTitle") or
                             self._ax_attr(elem, "AXDescription") or "")

                    # Collect button labels (up to 3 levels deep)
                    btn_labels: list[str] = []

                    def _collect_btns(node: object, d: int = 0) -> None:
                        if d > 3 or len(btn_labels) >= 5:
                            return
                        if self._ax_attr(node, "AXRole") == "AXButton":
                            lbl = self._ax_attr(node, "AXTitle") or ""
                            if lbl:
                                btn_labels.append(lbl)
                        for ch in (self._ax_attr(node, "AXChildren") or []):
                            _collect_btns(ch, d + 1)

                    _collect_btns(elem)

                    btn_str = " / ".join(btn_labels) if btn_labels else "OK"
                    title_str = f'"{title}"' if title else role
                    return (
                        f"MODAL DIALOG {title_str} is blocking the screen. "
                        f"Buttons available: [{btn_str}]. "
                        f"Dismiss it FIRST — press Return (default) or Escape (cancel) "
                        f"before continuing the task."
                    )
        except Exception:
            pass
        return ""

    def _walk_ax(
        self, elem: object, depth: int,
        results: list[str], sx: float, sy: float,
    ) -> None:
        if depth > _MAX_AX_DEPTH or len(results) >= _MAX_AX_ELEMS:
            return

        role  = self._ax_attr(elem, "AXRole") or ""
        title = (self._ax_attr(elem, "AXTitle") or
                 self._ax_attr(elem, "AXDescription") or
                 self._ax_attr(elem, "AXPlaceholderValue") or "")
        value = str(self._ax_attr(elem, "AXValue") or "")
        pos   = self._ax_attr(elem, "AXPosition")
        size  = self._ax_attr(elem, "AXSize")

        if role in _INTERACTIVE_ROLES and pos and size and (title or value):
            # Center in logical screen coords, then scale to screenshot space
            cx = int((pos.x + size.width  / 2) * sx)
            cy = int((pos.y + size.height / 2) * sy)
            label = (title or value)[:35].replace("\n", " ")
            results.append(f"{role}[{label}]@({cx},{cy})")

        for child in (self._ax_attr(elem, "AXChildren") or []):
            self._walk_ax(child, depth + 1, results, sx, sy)

    @staticmethod
    def _ax_attr(elem: object, attr: str) -> object:
        try:
            err, val = _AS.AXUIElementCopyAttributeValue(elem, attr, None)
            return val if err == 0 else None
        except Exception:
            return None

    def _applescript_ax_summary(self, app: str) -> str:
        """Phase 1 AppleScript fallback — focused element + dialog detection."""
        # Dialog check: count sheets attached to the frontmost window
        dialog_script = f'''
        tell application "System Events"
            tell process "{app}"
                try
                    if (count of sheets) > 0 then
                        set sh to sheet 1
                        set shDesc to ""
                        try
                            set shDesc to description of sh
                        end try
                        set btnNames to name of every button of sh
                        return "DIALOG: " & shDesc & " | Buttons: " & (btnNames as string)
                    end if
                end try
                return ""
            end tell
        end tell
        '''
        dr = subprocess.run(["osascript", "-e", dialog_script],
                            capture_output=True, text=True, timeout=3)
        dialog_info = dr.stdout.strip()

        focus_script = f'''
        tell application "System Events"
            tell process "{app}"
                try
                    set fe to value of attribute "AXFocusedUIElement"
                    return (role of fe) & ": " & (description of fe)
                on error
                    return "no focused element"
                end try
            end tell
        end tell
        '''
        r = subprocess.run(["osascript", "-e", focus_script],
                           capture_output=True, text=True, timeout=3)
        focused  = r.stdout.strip()[:100] or "none"
        app_hint = _APP_HINTS.get(app, "")
        hint_part = f" | Hint: {app_hint}" if app_hint else ""

        if dialog_info:
            return (
                f"⚠️ MODAL DIALOG blocking screen — {dialog_info}. "
                f"Dismiss FIRST (Return = default, Escape = cancel). "
                f"| App: {app} | Focused: {focused}{hint_part}"
            )
        return f"App: {app} | Focused: {focused}{hint_part}"

    # ── OpenCV screen-change verification ─────────────────────────────────────

    def _verify_screen_change(self, before: bytes, after: bytes) -> bool:
        """
        Returns True if the screen visibly changed between the two PNG frames.
        Uses mean absolute pixel difference on grayscale; threshold = 0.5/255.
        Falls back to True (assume success) when opencv is not installed.
        """
        if not _CV2_OK:
            return True
        try:
            b = _cv2.imdecode(_np.frombuffer(before, _np.uint8), _cv2.IMREAD_GRAYSCALE)
            a = _cv2.imdecode(_np.frombuffer(after,  _np.uint8), _cv2.IMREAD_GRAYSCALE)
            return float(_cv2.absdiff(b, a).mean()) > 0.5
        except Exception:
            return True

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _plan_task(self, task: str, context: str) -> list[str]:
        """One-shot Gemini text call that breaks the task into 3–7 ordered steps.

        Returns a list of step strings (e.g. ["Open Safari", "Navigate to gmail.com", ...]).
        Falls back to an empty list on any error so the loop degrades gracefully.
        """
        planning_prompt = (
            "You are a macOS desktop automation planner.\n"
            "Given a task and optional context, output a JSON array of 3–7 short, ordered steps "
            "that a human would follow to complete the task using a keyboard and mouse.\n"
            "Each step must be a single imperative sentence (≤ 15 words).\n"
            "Return ONLY a JSON array of strings — no markdown, no explanation.\n\n"
            f"Task: {task}\n"
        )
        if context:
            planning_prompt += f"Context: {context}\n"

        try:
            resp = self._get_gemini_client().models.generate_content(
                model=_VISION_MODEL,
                contents=planning_prompt,
                config=gtypes.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=400,
                ),
            )
            steps = json.loads(resp.text)
            if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
                return steps[:7]
        except Exception as exc:
            logger.debug("Task planning skipped: %s", exc)
        return []

    def _verify_completion(
        self, task: str, claimed_result: str, shot_bytes: bytes
    ) -> tuple[bool, str]:
        """Post-completion sanity check: did the task actually succeed?

        Sends the final screenshot + task description to Gemini Vision.
        Returns (verified, note). Fails open — returns (True, "") on any error
        so a transient API failure never blocks a successfully completed task.

        Conservative bias: Gemini is instructed to return verified=True when
        uncertain; only flag False when the screenshot clearly shows failure
        (error message, unchanged state, wrong content).
        """
        prompt = (
            f"Task: {task}\n"
            f"Claimed result: {claimed_result}\n\n"
            "Look at this screenshot. Does the evidence clearly show the task succeeded?\n"
            "Return JSON: {\"verified\": bool, \"note\": \"one sentence\"}\n"
            "Rules:\n"
            "- verified=true when uncertain — only false when the screenshot CLEARLY shows "
            "failure (error message, wrong page, task state unchanged).\n"
            "- Keep note under 20 words."
        )
        try:
            resp = self._get_gemini_client().models.generate_content(
                model=_VISION_MODEL,
                contents=[
                    gtypes.Part.from_bytes(data=shot_bytes, mime_type="image/png"),
                    gtypes.Part.from_text(text=prompt),
                ],
                config=gtypes.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=120,
                ),
            )
            data = json.loads(resp.text)
            return bool(data.get("verified", True)), str(data.get("note", ""))
        except Exception as exc:
            logger.debug("Completion verification skipped: %s", exc)
            return True, ""   # fail open

    def _build_system_prompt(self, task: str, context: str, plan: list[str] | None = None) -> str:
        prompt = (
            "You are a macOS desktop automation agent. "
            "Your job is to complete the given task by controlling the screen exactly as a human would.\n\n"
            "You receive a screenshot (1280×800) and an AX elements list.\n"
            "AX elements format: Role[label]@(cx,cy) — cx,cy are centres in screenshot space.\n\n"
            "Rules:\n"
            "- Return exactly one action per response as JSON matching the schema.\n"
            "- Coordinates (x, y) must be in screenshot space: x∈[0,1280], y∈[0,800].\n"
            "- Prefer AX element centres over visually estimated positions — they are exact.\n"
            "- If a previous step reports [WARNING: screen unchanged], try a different approach "
            "  (different coordinates, scroll to reveal the element, or use a keyboard shortcut).\n"
            "- Use keyboard shortcuts whenever possible — they are faster and more reliable than clicking.\n"
            "- When the task is complete, set action=done and summarise what was accomplished in 'result'.\n"
            "- If the task requires extracting specific text (an email address, URL, tracking number, etc.), "
            "  select and copy it first (⌘C), then use action=read_clipboard to capture the exact text, "
            "  and include it verbatim in your done result.\n"
            "- If the task is impossible (app not installed, page not found, wrong credentials), "
            "  set action=done and explain why in 'result'.\n"
        )
        if plan:
            plan_lines = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan))
            prompt += f"\nExecution plan (follow this order):\n{plan_lines}\n"
        if context:
            prompt += f"\nUser context:\n{context}\n"
        prompt += f"\nTask: {task}"
        return prompt

    def _build_parts(
        self,
        shot_bytes: bytes,
        ax_context: str,
        history: list[dict],
        first_person_frame: Optional[bytes],
        robot_camera_frame: Optional[bytes],
        stuck_count: int = 0,
    ) -> list[gtypes.Part]:
        parts: list[gtypes.Part] = []

        if first_person_frame:
            parts.append(gtypes.Part.from_bytes(data=first_person_frame, mime_type="image/jpeg"))
            parts.append(gtypes.Part.from_text(text="[Wearable camera — user's physical view]\n"))

        if robot_camera_frame:
            parts.append(gtypes.Part.from_bytes(data=robot_camera_frame, mime_type="image/jpeg"))
            parts.append(gtypes.Part.from_text(text="[Robot camera]\n"))

        parts.append(gtypes.Part.from_bytes(data=shot_bytes, mime_type="image/png"))

        state_text = f"Screen state: {ax_context}\n"
        if history:
            recent = history[-4:]
            hist_lines = " → ".join(
                f"{s['action']}({'✓' if s.get('screen_changed', True) else '✗'})"
                for s in recent
            )
            state_text += f"Recent steps: {hist_lines}\n"
        state_text += "What is the next single action? Return JSON only."

        if stuck_count >= 3:
            state_text += (
                f"\n⚠️ STUCK: screen unchanged for {stuck_count} consecutive steps. "
                "Switch to a completely different strategy:\n"
                "1. Spotlight search (key_press cmd+space) then type what you need.\n"
                "2. Tab key to cycle focus to the target element.\n"
                "3. Right-click instead of click.\n"
                "4. Scroll to reveal a hidden element.\n"
                "5. Press Escape and start fresh from a different entry point."
            )

        parts.append(gtypes.Part.from_text(text=state_text))
        return parts

    # ── Execute (with coordinate scaling) ────────────────────────────────────

    async def _execute(self, action: dict) -> str:
        """Execute an action. Coordinates from Gemini are in 1280×800 screenshot space;
        _to_screen() converts them to logical screen coords for pyautogui."""
        t = action.get("action", "wait")

        if t == _Act.click:
            x, y = _to_screen(action["x"], action["y"])
            pyautogui.click(x, y)
            return f"Clicked ({action['x']},{action['y']}) → screen ({x},{y})."

        elif t == _Act.double_click:
            x, y = _to_screen(action["x"], action["y"])
            pyautogui.doubleClick(x, y)
            return f"Double-clicked ({action['x']},{action['y']}) → screen ({x},{y})."

        elif t == _Act.right_click:
            x, y = _to_screen(action["x"], action["y"])
            pyautogui.click(x, y, button="right")
            return f"Right-clicked ({action['x']},{action['y']}) → screen ({x},{y})."

        elif t == _Act.type_text:
            text = action.get("text", "")
            # Use clipboard paste for reliable macOS text input.
            # pyautogui.write() drops special characters, accents, and emoji.
            subprocess.run(["pbcopy"], input=text.encode("utf-8"),
                           capture_output=True, check=False)
            pyautogui.hotkey("cmd", "v")
            return f"Typed via clipboard: {text[:60]}{'…' if len(text) > 60 else ''}"

        elif t == _Act.key_press:
            chord = action.get("text", "")
            keys = [k.strip() for k in re.split(r"[+\-]", chord)]
            pyautogui.hotkey(*keys)
            return f"Pressed: {chord}."

        elif t == _Act.scroll:
            x, y = _to_screen(action.get("x", 640), action.get("y", 400))
            direction = action.get("direction", "down")
            amount = int(action.get("amount", 3))
            delta = amount if direction == "up" else -amount
            pyautogui.scroll(delta, x=x, y=y)
            return f"Scrolled {direction} {amount}× at screen ({x},{y})."

        elif t == _Act.drag:
            sx1, sy1 = _to_screen(action["start_x"], action["start_y"])
            sx2, sy2 = _to_screen(action["end_x"],   action["end_y"])
            pyautogui.mouseDown(sx1, sy1)
            await asyncio.sleep(0.1)
            pyautogui.moveTo(sx2, sy2, duration=0.35)
            pyautogui.mouseUp()
            return f"Dragged screen ({sx1},{sy1}) → ({sx2},{sy2})."

        elif t == _Act.wait:
            await asyncio.sleep(1.0)
            return "Waited 1 second."

        elif t == _Act.read_clipboard:
            r = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=3)
            content = r.stdout.strip()
            if not content:
                return "Clipboard is empty."
            return f"Clipboard content: {content[:500]}{'…' if len(content) > 500 else ''}"

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
            return f"{t} at screenshot ({action['x']},{action['y']})"
        return str(action)

    # ── Routing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_browser_task(task: str) -> bool:
        t = task.lower()
        return any(sig in t for sig in _BROWSER_SIGNALS)

    # ── Demo recording ────────────────────────────────────────────────────────

    def _save_demos(self, task: str, demos: list[dict]) -> None:
        out_dir = Path("data/demonstrations")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"demo_{int(time.time())}.json"
        slim = [{k: v for k, v in d.items() if k != "before_screenshot_b64"} for d in demos]
        out_file.write_text(
            json.dumps({"task": task, "steps": len(demos), "demos": slim}, indent=2)
        )
        logger.info("NavAgent: saved %d demo steps → %s", len(demos), out_file)


# ── Navigation memory helpers ─────────────────────────────────────────────────

def _save_nav_memory(task: str, steps: list[dict], result: str) -> None:
    """Persist a concise navigation pattern to memory after a successful run.

    Stored with source='nav' — searchable by the nav agent but excluded from
    the user-facing Memory panel (which shows only source='fact' rows).
    Runs in the calling thread; errors are swallowed so they never block a task.
    """
    try:
        apps = list({s["app"] for s in steps if s.get("app") and s["app"] != "Unknown"})
        path = "browser" if not steps else "desktop"
        memo_parts = [f"Nav: '{task[:100]}'", f"path={path}"]
        if apps:
            memo_parts.append(f"apps={','.join(apps[:4])}")
        memo_parts.append(f"steps={len(steps)}")
        memo_parts.append(f"result={result[:150]}")
        memo = " | ".join(memo_parts)

        from src.memory import save_nav_pattern
        save_nav_pattern(memo)
        logger.debug("Nav memory saved: %s", memo[:80])
    except Exception as exc:
        logger.debug("Nav memory save skipped: %s", exc)


def _get_nav_context(task: str) -> str:
    """Retrieve the most relevant past navigation patterns for *task*.

    Returns a formatted string ready to inject as context, or '' when the DB
    is unavailable or no relevant patterns exist yet.
    """
    try:
        from src.memory import search_nav_patterns
        patterns = search_nav_patterns(task, limit=3)
        if patterns:
            return f"Past navigation patterns for similar tasks:\n{patterns}"
    except Exception as exc:
        logger.debug("Nav context retrieval skipped: %s", exc)
    return ""


# ── Wearable frame context store ──────────────────────────────────────────────
# Single-user system: one frame at a time stored here.
# Set by wearable_stream before the agent runs; consumed once by navigate_computer.

_first_person_frame: Optional[bytes] = None


def set_first_person_frame(frame: Optional[bytes]) -> None:
    """Store a wearable camera frame to be used as first_person_frame on next navigate_computer call."""
    global _first_person_frame
    _first_person_frame = frame


def get_and_clear_first_person_frame() -> Optional[bytes]:
    """Return the stored wearable frame and clear it (consume-once semantics)."""
    global _first_person_frame
    frame, _first_person_frame = _first_person_frame, None
    return frame


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
    _first_person_frame: Optional[bytes] = None,
) -> str:
    """
    LangGraph tool — navigate the computer to complete any task.

    Automatically routes:
      • Browser/web tasks  → browser-use (MIT) + Gemini Flash free tier
      • Native macOS apps  → Gemini vision loop (free) + pyautogui

    Injects past navigation patterns as context so the agent improves over time.
    _first_person_frame: internal; not in tool schema. Injected by the sync
    shim from the wearable frame store when triggered by Frame glasses.
    """
    # Enrich context with relevant past navigation patterns
    nav_ctx = _get_nav_context(task)
    full_context = "\n\n".join(filter(None, [context, nav_ctx]))

    result = await _get_agent().run(
        task, full_context,
        hitl_callback=hitl_callback,
        first_person_frame=_first_person_frame,
    )
    return f"[{result.status}] {result.result} ({len(result.steps)} steps)"
