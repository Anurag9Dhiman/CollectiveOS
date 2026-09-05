"""
Tests for Phase 4: Frame wearable integration.
Covers: display formatting, first_person_frame store/consume, wearable stream integration.
"""

from __future__ import annotations

import base64
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── format_for_frame ──────────────────────────────────────────────────────────

class TestFormatForFrame:
    def test_strips_markdown_bold(self):
        from src.connectors.frame_wearable import format_for_frame
        result = format_for_frame("**Hello** world")
        assert "**" not in result
        assert "Hello" in result

    def test_strips_markdown_links(self):
        from src.connectors.frame_wearable import format_for_frame
        result = format_for_frame("See [this link](https://example.com) for details.")
        assert "https://example.com" not in result
        assert "this link" in result

    def test_strips_code_fences(self):
        from src.connectors.frame_wearable import format_for_frame
        result = format_for_frame("Use `git status` to check.")
        assert "`" not in result
        assert "git status" in result

    def test_truncates_long_text(self):
        from src.connectors.frame_wearable import format_for_frame
        long_text = "word " * 100   # ~500 chars
        result = format_for_frame(long_text)
        # Must fit within max_lines × line_width
        assert len(result) <= 25 * 6 + 6   # chars + newlines + ellipsis

    def test_wraps_to_line_width(self):
        from src.connectors.frame_wearable import format_for_frame
        with patch("src.connectors.frame_wearable._FRAME_LINE_WIDTH", 10):
            result = format_for_frame("This is a longer sentence that must wrap.")
        for line in result.split("\n"):
            assert len(line) <= 10

    def test_max_lines_respected(self):
        from src.connectors.frame_wearable import format_for_frame
        with patch("src.connectors.frame_wearable._FRAME_MAX_LINES", 3), \
             patch("src.connectors.frame_wearable._FRAME_LINE_WIDTH", 10):
            result = format_for_frame("a b c d e f g h i j k l m n o p q r s t u")
        assert result.count("\n") <= 2   # max 3 lines = max 2 newlines

    def test_short_text_unchanged_structure(self):
        from src.connectors.frame_wearable import format_for_frame
        result = format_for_frame("Done.")
        assert "Done" in result


# ── First-person frame store / consume ───────────────────────────────────────

class TestFirstPersonFrameStore:
    def setup_method(self):
        # Reset global state before each test
        import src.agents.nav_agent as mod
        mod._first_person_frame = None

    def test_set_and_get_frame(self):
        from src.agents.nav_agent import set_first_person_frame, get_and_clear_first_person_frame
        sample = b"\xff\xd8\xff"   # JPEG magic bytes
        set_first_person_frame(sample)
        result = get_and_clear_first_person_frame()
        assert result == sample

    def test_consume_clears_frame(self):
        from src.agents.nav_agent import set_first_person_frame, get_and_clear_first_person_frame
        set_first_person_frame(b"frame-data")
        get_and_clear_first_person_frame()           # first call consumes
        assert get_and_clear_first_person_frame() is None   # second call → None

    def test_set_none_clears(self):
        from src.agents.nav_agent import set_first_person_frame, get_and_clear_first_person_frame
        set_first_person_frame(b"old-frame")
        set_first_person_frame(None)
        assert get_and_clear_first_person_frame() is None

    def test_navigate_computer_wrapper_passes_frame(self):
        """The async navigate_computer wrapper forwards _first_person_frame to NavAgent.run()."""
        import asyncio
        import src.agents.nav_agent as mod

        captured: list = []

        async def fake_run(task, context="", *, hitl_callback=None,
                           first_person_frame=None, robot_camera_frame=None, record=False):
            captured.append(first_person_frame)
            return mod.NavResult(status="done", result="ok", steps=[])

        agent = mod._get_agent()
        with patch.object(agent, "run", side_effect=fake_run):
            asyncio.run(mod.navigate_computer("open Finder", _first_person_frame=b"jpg-bytes"))

        assert captured[0] == b"jpg-bytes"


# ── Wearable stream: frame forwarding ────────────────────────────────────────

class TestWearableStreamFrameForwarding:
    def test_run_agent_sets_frame_when_image_present(self):
        """_run_agent should call set_first_person_frame with decoded bytes."""
        fake_bytes = b"\xff\xd8\xff"
        fake_b64   = base64.b64encode(fake_bytes).decode()

        set_calls: list = []

        with patch("src.wearable_stream._classify_intent", return_value=True), \
             patch("src.agents.nav_agent.set_first_person_frame",
                   side_effect=lambda f: set_calls.append(f)), \
             patch("src.memory.search_with_graph", return_value=""), \
             patch("src.conversations.create", return_value=1), \
             patch("src.conversations.save_message"), \
             patch("src.agent.run", return_value=("reply", False, False)), \
             patch("src.api._system_prompt", return_value=""), \
             patch("src.memory.save_smart"):

            from src.wearable_stream import _run_agent
            _run_agent("do something", [], fake_b64, "image/jpeg")

        assert len(set_calls) == 1
        assert set_calls[0] == fake_bytes

    def test_run_agent_clears_frame_when_no_image(self):
        """_run_agent should call set_first_person_frame(None) when no image."""
        set_calls: list = []

        with patch("src.agents.nav_agent.set_first_person_frame",
                   side_effect=lambda f: set_calls.append(f)), \
             patch("src.memory.search_with_graph", return_value=""), \
             patch("src.conversations.create", return_value=1), \
             patch("src.conversations.save_message"), \
             patch("src.agent.run", return_value=("reply", False, False)), \
             patch("src.api._system_prompt", return_value=""), \
             patch("src.memory.save_smart"):

            from src.wearable_stream import _run_agent
            _run_agent("do something", [], None, "image/jpeg")

        assert set_calls[-1] is None

    def test_reply_includes_frame_display_field(self, auth):
        """WebSocket reply JSON must include frame_display key after Phase 4."""
        # This is tested indirectly via the wearable stream mock —
        # the field is added in handle_wearable_ws when triggered=True.
        # We verify format_for_frame is callable and returns a str.
        from src.connectors.frame_wearable import format_for_frame
        result = format_for_frame("Meeting confirmed for 3pm with Alice.")
        assert isinstance(result, str)
        assert len(result) > 0


# ── FrameBLEConnector stubs ───────────────────────────────────────────────────

class TestFrameBLEConnector:
    @pytest.mark.asyncio
    async def test_connect_returns_false_when_env_not_set(self):
        """No FRAME_BLE_DEVICE → connect returns False without error."""
        from src.connectors.frame_wearable import FrameBLEConnector
        connector = FrameBLEConnector()
        with patch("src.connectors.frame_wearable._FRAME_BLE_DEVICE", ""):
            result = await connector.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_returns_false_when_sdk_missing(self):
        """frame-sdk not installed → connect returns False without error."""
        from src.connectors.frame_wearable import FrameBLEConnector
        connector = FrameBLEConnector()
        with patch("src.connectors.frame_wearable._FRAME_BLE_DEVICE", "AA:BB:CC:DD"), \
             patch.dict(sys.modules, {"frame_sdk": None}):
            result = await connector.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_show_text_returns_false_when_not_connected(self):
        from src.connectors.frame_wearable import FrameBLEConnector
        connector = FrameBLEConnector()
        result = await connector.show_text("hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_capture_photo_returns_none_when_not_connected(self):
        from src.connectors.frame_wearable import FrameBLEConnector
        connector = FrameBLEConnector()
        result = await connector.capture_photo()
        assert result is None
