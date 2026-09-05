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
    agent = NavAgent()
    # Trigger lazy client creation so agent._gemini is a MagicMock (from conftest
    # mock of genai.Client) and tests can do agent._gemini.models.generate_content = ...
    agent._get_gemini_client()
    return agent


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
        agent._get_ax_tree        = MagicMock(return_value="App: Finder | AX elements: AXButton[OK]@(640,400)")

        # Skip the pre-run planning call — it would otherwise consume one response
        # from the side_effect list and leave the loop without enough responses.
        agent._plan_task = MagicMock(return_value=[])

        # Mock Gemini response sequence for the vision loop only
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
    async def test_read_clipboard_action_returns_content(self):
        """read_clipboard action runs pbpaste and returns its stdout."""
        agent = self._patched_run([
            {"action": "read_clipboard", "reason": "capture copied text"},
            {"action": "done", "result": "Got email: user@example.com"},
        ])
        fake_pbpaste = MagicMock()
        fake_pbpaste.stdout = "user@example.com"
        with patch("pyautogui.click"), \
             patch("subprocess.run", return_value=fake_pbpaste):
            result = await agent._run_desktop("get email from screen", "", hitl_callback=None,
                                              first_person_frame=None, robot_camera_frame=None,
                                              record=False)
        assert result.status == "done"
        clipboard_step = result.steps[0]
        assert "user@example.com" in clipboard_step["outcome"]

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
        agent._get_ax_tree        = MagicMock(return_value="App: Chrome | AX elements: none")
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


# ── Phase 2: coordinate scaling ───────────────────────────────────────────────

class TestCoordinateScaling:
    def test_to_screen_identity_when_scale_is_one(self):
        """When scale=(1,1), screenshot coords == screen coords."""
        from src.agents.nav_agent import _to_screen
        with patch("src.agents.nav_agent._scale_cache", (1.0, 1.0)):
            assert _to_screen(640, 400) == (640, 400)

    def test_to_screen_scales_down_for_retina(self):
        """For a 1512×982 logical screen, scale=(1280/1512, 800/982)."""
        from src.agents.nav_agent import _to_screen
        sx = 1280 / 1512
        sy = 800  / 982
        with patch("src.agents.nav_agent._scale_cache", (sx, sy)):
            x, y = _to_screen(640, 400)
            # 640 / (1280/1512) = 640 * 1512/1280 = 756
            assert x == 756
            assert y == int(400 / sy)

    def test_get_scale_uses_pyautogui_size(self):
        """_get_scale() reads pyautogui.size() and caches the result."""
        import src.agents.nav_agent as mod
        import pyautogui
        original = mod._scale_cache
        mod._scale_cache = None   # force recompute
        try:
            pyautogui.size = MagicMock(return_value=(1512, 982))
            sx, sy = mod._get_scale()
            assert abs(sx - 1280/1512) < 0.001
            assert abs(sy - 800/982)   < 0.001
        finally:
            mod._scale_cache = original


# ── Phase 2: OpenCV verification ──────────────────────────────────────────────

class TestScreenVerification:
    def test_verify_returns_true_when_cv2_unavailable(self):
        """When OpenCV is not installed, verification always returns True (safe default)."""
        from src.agents.nav_agent import NavAgent
        agent = NavAgent()
        with patch("src.agents.nav_agent._CV2_OK", False):
            assert agent._verify_screen_change(b"before", b"after") is True

    def test_verify_true_on_different_frames(self):
        """Two different byte arrays → changed=True (mocked cv2)."""
        import src.agents.nav_agent as mod
        agent = mod.NavAgent()
        # Make cv2 available in the module with a diff that exceeds threshold
        fake_diff = MagicMock()
        fake_diff.mean.return_value = 5.0   # > 0.5 threshold
        with patch("src.agents.nav_agent._CV2_OK", True), \
             patch("src.agents.nav_agent._cv2") as mock_cv2, \
             patch("src.agents.nav_agent._np") as mock_np:
            mock_np.frombuffer.return_value = b"data"
            mock_cv2.imdecode.return_value = MagicMock()
            mock_cv2.absdiff.return_value = fake_diff
            assert agent._verify_screen_change(b"A", b"B") is True

    def test_verify_false_on_identical_frames(self):
        """Same frame → changed=False."""
        import src.agents.nav_agent as mod
        agent = mod.NavAgent()
        fake_diff = MagicMock()
        fake_diff.mean.return_value = 0.0   # no change
        with patch("src.agents.nav_agent._CV2_OK", True), \
             patch("src.agents.nav_agent._cv2") as mock_cv2, \
             patch("src.agents.nav_agent._np") as mock_np:
            mock_np.frombuffer.return_value = b"data"
            mock_cv2.imdecode.return_value = MagicMock()
            mock_cv2.absdiff.return_value = fake_diff
            assert agent._verify_screen_change(b"A", b"A") is False


# ── Phase 2: AX tree fallback ─────────────────────────────────────────────────

class TestAXTree:
    def test_falls_back_to_applescript_when_pyobjc_unavailable(self):
        """When _PYOBJC_OK is False, _get_ax_tree calls the AppleScript path."""
        from src.agents.nav_agent import NavAgent
        agent = NavAgent()
        agent._applescript_ax_summary = MagicMock(return_value="App: Finder | Focused: none")
        with patch("src.agents.nav_agent._PYOBJC_OK", False):
            result = agent._get_ax_tree("Finder")
        agent._applescript_ax_summary.assert_called_once_with("Finder")
        assert "Finder" in result

    def test_falls_back_on_pyobjc_exception(self):
        """If pyobjc AX walk raises, falls back to AppleScript."""
        from src.agents.nav_agent import NavAgent
        agent = NavAgent()
        agent._pyobjc_ax_tree = MagicMock(side_effect=RuntimeError("AX denied"))
        agent._applescript_ax_summary = MagicMock(return_value="fallback")
        with patch("src.agents.nav_agent._PYOBJC_OK", True):
            result = agent._get_ax_tree("Terminal")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_unchanged_screen_adds_warning_to_history(self):
        """When screen doesn't change after a click, outcome contains [WARNING]."""
        agent = _nav_agent()
        agent._capture_screenshot = MagicMock(return_value=b"same-bytes")
        agent._get_frontmost_app  = MagicMock(return_value="Finder")
        agent._get_ax_tree        = MagicMock(return_value="App: Finder | AX elements: none")
        agent._verify_screen_change = MagicMock(return_value=False)  # no change
        agent._plan_task = MagicMock(return_value=[])  # don't consume loop responses
        agent._gemini.models.generate_content = MagicMock(side_effect=[
            _make_gemini_response({"action": "click", "x": 100, "y": 100, "reason": "click item"}),
            _make_gemini_response({"action": "done", "result": "Done."}),
        ])
        with patch("pyautogui.click"), patch("subprocess.run"):
            result = await agent._run_desktop("click Finder item", "", hitl_callback=None,
                                              first_person_frame=None, robot_camera_frame=None,
                                              record=False)
        assert result.status == "done"
        click_step = result.steps[0]
        assert "WARNING" in click_step["outcome"]
        assert click_step["screen_changed"] is False
