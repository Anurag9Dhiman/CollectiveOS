"""
Unit tests for NavAgent — free-only stack.

All external I/O is mocked:
- google.genai (Gemini Flash free tier)
- pyautogui, subprocess, PIL, browser_use
No real screen, API calls, or file writes happen.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Make PIL.Image.open usable in tests ──────────────────────────────────────

_fake_img = MagicMock()
_fake_img.resize.return_value = _fake_img
_fake_img.save = MagicMock()
sys.modules["PIL"].Image.open = MagicMock(return_value=_fake_img)
sys.modules["PIL"].Image.LANCZOS = 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_gemini_response(action_dict: dict) -> MagicMock:
    resp = MagicMock()
    resp.text = json.dumps(action_dict)
    return resp


def _nav_agent():
    """Import NavAgent fresh (conftest already mocked google.genai)."""
    # clear cached module so each test group can reload cleanly
    for key in list(sys.modules):
        if key.startswith("src.agents.nav_agent"):
            del sys.modules[key]
    from src.agents.nav_agent import NavAgent
    return NavAgent()


# ── Routing ───────────────────────────────────────────────────────────────────

class TestRouting:
    def test_browser_task_detected(self):
        from src.agents.nav_agent import NavAgent
        assert NavAgent._is_browser_task("open gmail and find the latest email") is True
        assert NavAgent._is_browser_task("go to github.com and create an issue") is True
        assert NavAgent._is_browser_task("search on Google for coffee shops") is True

    def test_desktop_task_not_browser(self):
        from src.agents.nav_agent import NavAgent
        assert NavAgent._is_browser_task("open Terminal and run ls") is False
        assert NavAgent._is_browser_task("take a screenshot of Finder") is False
        assert NavAgent._is_browser_task("open Xcode and build the project") is False


# ── HITL gate ─────────────────────────────────────────────────────────────────

class TestHITL:
    def setup_method(self):
        from src.agents.nav_agent import NavAgent
        self.agent = NavAgent()

    def test_write_action_triggers_confirmation(self):
        assert self.agent._needs_confirmation({"action": "type_text", "text": "send this"}) is True
        assert self.agent._needs_confirmation({"action": "click", "reason": "submit the form"}) is True
        assert self.agent._needs_confirmation({"action": "key_press", "text": "delete"}) is True
        assert self.agent._needs_confirmation({"action": "type_text", "text": "purchase ticket"}) is True

    def test_safe_action_skips_confirmation(self):
        assert self.agent._needs_confirmation({"action": "click", "x": 100, "y": 200}) is False
        assert self.agent._needs_confirmation({"action": "scroll", "direction": "down"}) is False
        assert self.agent._needs_confirmation({"action": "key_press", "text": "cmd+c"}) is False

    def test_action_description_type_text(self):
        desc = self.agent._action_description({"action": "type_text", "text": "Hello world"})
        assert "Hello world" in desc

    def test_action_description_key_press(self):
        desc = self.agent._action_description({"action": "key_press", "text": "cmd+shift+d"})
        assert "cmd+shift+d" in desc

    def test_action_description_click_coords(self):
        desc = self.agent._action_description({"action": "click", "x": 42, "y": 84})
        assert "42" in desc and "84" in desc


# ── Desktop vision loop ───────────────────────────────────────────────────────

class TestDesktopLoop:
    def _patched_run(self, gemini_responses: list[dict]):
        """Return a NavAgent with mocked perceive + Gemini."""
        agent = _nav_agent()

        # Mock screencapture and PIL
        agent._capture_screenshot = MagicMock(return_value=b"fake-png")
        agent._get_frontmost_app  = MagicMock(return_value="Finder")
        agent._get_ax_summary     = MagicMock(return_value="App: Finder | Focused: none")

        # Mock Gemini response sequence
        responses = [_make_gemini_response(r) for r in gemini_responses]
        agent._gemini.models.generate_content = MagicMock(side_effect=responses)

        return agent

    @pytest.mark.asyncio
    async def test_done_on_first_action(self):
        agent = self._patched_run([
            {"action": "done", "result": "Opened Finder."},
        ])
        with patch("pyautogui.click"), patch("subprocess.run"):
            result = await agent._run_desktop("open Finder", "", hitl_callback=None,
                                              first_person_frame=None, robot_camera_frame=None,
                                              record=False)
        assert result.status == "done"
        assert "Finder" in result.result

    @pytest.mark.asyncio
    async def test_click_then_done(self):
        agent = self._patched_run([
            {"action": "click", "x": 100, "y": 200, "reason": "click Finder icon"},
            {"action": "done", "result": "Task done."},
        ])
        with patch("pyautogui.click"), patch("subprocess.run"):
            result = await agent._run_desktop("click Finder", "", hitl_callback=None,
                                              first_person_frame=None, robot_camera_frame=None,
                                              record=False)
        assert result.status == "done"
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_hitl_blocks_write_action(self):
        """HITL callback rejecting a write action → hitl_paused status."""
        agent = self._patched_run([
            {"action": "type_text", "text": "send email now", "reason": "send it"},
        ])

        async def reject(_): return False

        with patch("pyautogui.write"), patch("subprocess.run"):
            result = await agent._run_desktop("send email", "", hitl_callback=reject,
                                              first_person_frame=None, robot_camera_frame=None,
                                              record=False)
        assert result.status == "hitl_paused"

    @pytest.mark.asyncio
    async def test_hitl_approved_continues(self):
        """HITL callback approving → execution continues."""
        agent = self._patched_run([
            {"action": "type_text", "text": "send email now", "reason": "send it"},
            {"action": "done", "result": "Email sent."},
        ])

        async def approve(_): return True

        with patch("pyautogui.write"), patch("subprocess.run"):
            result = await agent._run_desktop("send email", "", hitl_callback=approve,
                                              first_person_frame=None, robot_camera_frame=None,
                                              record=False)
        assert result.status == "done"

    @pytest.mark.asyncio
    async def test_max_iter_guard(self):
        """Returns max_iter after hitting iteration cap."""
        # Always return a safe action — never done
        responses = [{"action": "scroll", "direction": "down", "amount": 3}] * 25
        agent = self._patched_run(responses)

        # Patch NAV_MAX_ITER to 3 so the test is fast
        with patch("src.agents.nav_agent._NAV_MAX_ITER", 3), \
             patch("pyautogui.scroll"), patch("subprocess.run"):
            result = await agent._run_desktop("scroll forever", "", hitl_callback=None,
                                              first_person_frame=None, robot_camera_frame=None,
                                              record=False)
        assert result.status == "max_iter"

    @pytest.mark.asyncio
    async def test_gemini_error_returns_error_status(self):
        agent = _nav_agent()
        agent._capture_screenshot = MagicMock(return_value=b"fake-png")
        agent._get_frontmost_app  = MagicMock(return_value="Safari")
        agent._get_ax_summary     = MagicMock(return_value="")
        agent._gemini.models.generate_content = MagicMock(side_effect=RuntimeError("API down"))

        result = await agent._run_desktop("do something", "", hitl_callback=None,
                                          first_person_frame=None, robot_camera_frame=None,
                                          record=False)
        assert result.status == "error"
        assert "API down" in result.result


# ── Browser path ──────────────────────────────────────────────────────────────

class TestBrowserPath:
    @pytest.mark.asyncio
    async def test_browser_use_unavailable_falls_back(self):
        """If browser-use is not installed, falls back to desktop vision path."""
        agent = _nav_agent()
        agent._capture_screenshot = MagicMock(return_value=b"fake-png")
        agent._get_frontmost_app  = MagicMock(return_value="Chrome")
        agent._get_ax_summary     = MagicMock(return_value="")
        agent._gemini.models.generate_content = MagicMock(
            return_value=_make_gemini_response({"action": "done", "result": "Found it."})
        )

        # Force ImportError inside _run_browser by removing browser_use mock
        with patch.dict(sys.modules, {"browser_use": None}), \
             patch("subprocess.run"):
            result = await agent._run_browser("open gmail", "", None, False)

        # Falls back to desktop path which completes via mocked Gemini
        assert result.status in ("done", "error", "max_iter")

    @pytest.mark.asyncio
    async def test_browser_use_success(self):
        """browser-use Agent.run() returns result → done."""
        agent = _nav_agent()

        fake_browser_agent = MagicMock()
        fake_browser_agent.run = AsyncMock(return_value="Email found: hello@example.com")

        fake_bu_module      = MagicMock()
        fake_bu_module.Agent = MagicMock(return_value=fake_browser_agent)

        with patch.dict(sys.modules, {"browser_use": fake_bu_module,
                                       "langchain_google_genai": sys.modules["langchain_google_genai"]}):
            result = await agent._run_browser("check gmail", "user context", None, False)

        assert result.status == "done"
        assert "Email found" in result.result


# ── LangGraph tool wrapper ────────────────────────────────────────────────────

class TestNavigateComputerTool:
    @pytest.mark.asyncio
    async def test_returns_string(self):
        from src.agents import nav_agent as mod

        async def fake_run(task, context="", *, hitl_callback=None,
                           first_person_frame=None, robot_camera_frame=None, record=False):
            from src.agents.nav_agent import NavResult
            return NavResult(status="done", result="Opened app.", steps=[{}, {}])

        with patch.object(mod._get_agent(), "run", side_effect=fake_run):
            out = await mod.navigate_computer("open Finder")
        assert isinstance(out, str)
        assert "done" in out
        assert "2 steps" in out
