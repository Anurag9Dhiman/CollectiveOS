"""Tests for Phase 16: Nav agent memory — save patterns, retrieve context."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call


# ── memory.py additions ───────────────────────────────────────────────────────

class TestSaveNavPattern:
    def test_saves_with_nav_source(self):
        inserted = []

        class FakeCur:
            def execute(self, sql, params):
                inserted.append(params)
            def __enter__(self): return self
            def __exit__(self, *a): pass

        class FakeConn:
            def cursor(self): return FakeCur()
            def commit(self): pass
            def close(self): pass

        with patch("src.memory._embed", return_value=[0.1] * 3072), \
             patch("src.memory.connect", return_value=FakeConn()), \
             patch("src.memory.default_user_id", return_value=1):
            from src.memory import save_nav_pattern
            save_nav_pattern("Nav: 'check Gmail' | path=browser | result=done")

        assert len(inserted) == 1
        # params: (user_id, pattern, embedding, now) — 'nav' is a SQL literal
        assert "Nav:" in inserted[0][1]   # pattern string is in position 1

    def test_swallows_db_errors(self):
        with patch("src.memory._embed", side_effect=Exception("no embed")):
            from src.memory import save_nav_pattern
            save_nav_pattern("any pattern")   # must not raise


class TestSearchNavPatterns:
    def test_returns_empty_when_no_rows(self):
        class FakeCur:
            def execute(self, *a): pass
            def fetchall(self): return []
            def __enter__(self): return self
            def __exit__(self, *a): pass

        class FakeConn:
            def cursor(self): return FakeCur()
            def close(self): pass

        with patch("src.memory._embed", return_value=[0.1] * 3072), \
             patch("src.memory.connect", return_value=FakeConn()), \
             patch("src.memory.default_user_id", return_value=1):
            from src.memory import search_nav_patterns
            result = search_nav_patterns("check Gmail")

        assert result == ""

    def test_returns_patterns_as_text(self):
        from datetime import datetime, timezone
        ts = datetime(2026, 1, 10, tzinfo=timezone.utc)
        fake_rows = [
            ("Nav: 'check Gmail' | path=browser | result=found 3 emails", ts),
        ]

        class FakeCur:
            def execute(self, *a): pass
            def fetchall(self): return fake_rows
            def __enter__(self): return self
            def __exit__(self, *a): pass

        class FakeConn:
            def cursor(self): return FakeCur()
            def close(self): pass

        with patch("src.memory._embed", return_value=[0.1] * 3072), \
             patch("src.memory.connect", return_value=FakeConn()), \
             patch("src.memory.default_user_id", return_value=1):
            from src.memory import search_nav_patterns
            result = search_nav_patterns("check Gmail")

        assert "Gmail" in result
        assert "2026-01-10" in result

    def test_returns_empty_string_on_db_error(self):
        with patch("src.memory._embed", side_effect=Exception("no db")):
            from src.memory import search_nav_patterns
            assert search_nav_patterns("task") == ""

    def test_empty_query_returns_empty(self):
        from src.memory import search_nav_patterns
        assert search_nav_patterns("") == ""
        assert search_nav_patterns("   ") == ""


# ── nav_agent.py helpers ──────────────────────────────────────────────────────

class TestSaveNavMemory:
    def test_saves_browser_pattern(self):
        saved = []
        with patch("src.memory.save_nav_pattern", side_effect=lambda p: saved.append(p)):
            from src.agents.nav_agent import _save_nav_memory
            _save_nav_memory("check Gmail", [], "Found 3 emails")

        assert len(saved) == 1
        assert "Gmail" in saved[0]
        assert "browser" in saved[0]

    def test_saves_desktop_pattern_with_apps(self):
        saved = []
        steps = [
            {"action": "click", "app": "Finder", "screen_changed": True},
            {"action": "type_text", "app": "Finder", "screen_changed": True},
        ]
        with patch("src.memory.save_nav_pattern", side_effect=lambda p: saved.append(p)):
            from src.agents.nav_agent import _save_nav_memory
            _save_nav_memory("open a file in Finder", steps, "File opened")

        assert "Finder" in saved[0]
        assert "desktop" in saved[0]
        assert "steps=2" in saved[0]

    def test_swallows_memory_errors(self):
        with patch("src.memory.save_nav_pattern", side_effect=Exception("db down")):
            from src.agents.nav_agent import _save_nav_memory
            _save_nav_memory("task", [], "result")   # must not raise

    def test_excludes_unknown_apps(self):
        saved = []
        steps = [{"action": "click", "app": "Unknown", "screen_changed": True}]
        with patch("src.memory.save_nav_pattern", side_effect=lambda p: saved.append(p)):
            from src.agents.nav_agent import _save_nav_memory
            _save_nav_memory("task", steps, "done")

        assert "Unknown" not in saved[0]


class TestGetNavContext:
    def test_returns_empty_when_no_patterns(self):
        with patch("src.memory.search_nav_patterns", return_value=""):
            from src.agents.nav_agent import _get_nav_context
            result = _get_nav_context("any task")
        assert result == ""

    def test_wraps_patterns_with_header(self):
        patterns = "[2026-01-10] Nav: 'check Gmail' | result=done"
        with patch("src.memory.search_nav_patterns", return_value=patterns):
            from src.agents.nav_agent import _get_nav_context
            result = _get_nav_context("check email")
        assert "Past navigation patterns" in result
        assert "Gmail" in result

    def test_swallows_import_errors(self):
        with patch("src.memory.search_nav_patterns", side_effect=Exception("no module")):
            from src.agents.nav_agent import _get_nav_context
            result = _get_nav_context("task")
        assert result == ""


# ── Integration: navigate_computer injects context ───────────────────────────

class TestNavigateComputerContext:
    def test_context_injected_into_run(self):
        """navigate_computer must pass enriched context (nav patterns + caller context) to run()."""
        import asyncio
        from src.agents.nav_agent import NavResult

        captured_context = []

        async def fake_run(task, context, **kwargs):
            captured_context.append(context)
            return NavResult(status="done", result="ok", steps=[])

        with patch("src.agents.nav_agent._get_agent") as mock_get_agent, \
             patch("src.agents.nav_agent._get_nav_context",
                   return_value="Past navigation patterns for similar tasks:\n[2026] Nav: old"):
            agent = MagicMock()
            agent.run = fake_run
            mock_get_agent.return_value = agent

            from src.agents.nav_agent import navigate_computer
            asyncio.run(navigate_computer("check email", context="user is busy"))

        assert len(captured_context) == 1
        ctx = captured_context[0]
        assert "user is busy" in ctx
        assert "Past navigation patterns" in ctx

    def test_empty_nav_context_not_included(self):
        import asyncio
        from src.agents.nav_agent import NavResult

        captured_context = []

        async def fake_run(task, context, **kwargs):
            captured_context.append(context)
            return NavResult(status="done", result="ok", steps=[])

        with patch("src.agents.nav_agent._get_agent") as mock_get_agent, \
             patch("src.agents.nav_agent._get_nav_context", return_value=""):
            agent = MagicMock()
            agent.run = fake_run
            mock_get_agent.return_value = agent

            from src.agents.nav_agent import navigate_computer
            asyncio.run(navigate_computer("open Finder", context=""))

        assert captured_context[0] == ""   # no spurious newlines or headers
