"""
Tests for Phase 15: Activity log — log_event, list_events, REST endpoint,
and wiring into scheduler / output_bus / watchers.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock, call

import pytest


# ── activity.py unit tests ────────────────────────────────────────────────────

class TestLogEvent:
    def test_log_event_swallows_db_errors(self):
        """log_event must never raise even when DB is unavailable."""
        with patch("src.activity.connect", side_effect=Exception("no db")):
            from src.activity import log_event
            log_event("routine", "Morning briefing", "ok result")   # must not raise

    def test_log_event_truncates_long_body(self):
        """Body longer than 1000 chars should be truncated before insert."""
        inserted = []

        class FakeCur:
            def execute(self, sql, params):
                inserted.append(params)
            def __enter__(self): return self
            def __exit__(self, *a): pass

        class FakeConn:
            def cursor(self): return FakeCur()
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch("src.activity.connect", return_value=FakeConn()):
            from src.activity import log_event
            log_event("routine", "title", "x" * 2000)

        body = inserted[0][2]
        assert len(body) <= 1000


class TestListEvents:
    def test_list_events_returns_empty_on_db_error(self):
        with patch("src.activity.connect", side_effect=Exception("no db")):
            from src.activity import list_events
            assert list_events() == []

    def test_list_events_maps_rows_to_dicts(self):
        from datetime import datetime, timezone
        ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        fake_rows = [
            (1, ts, "routine", "Morning briefing", "done"),
            (2, ts, "watcher", "BTC alert", "triggered"),
        ]

        class FakeCur:
            def execute(self, *a): pass
            def fetchall(self): return fake_rows
            def __enter__(self): return self
            def __exit__(self, *a): pass

        class FakeConn:
            def cursor(self): return FakeCur()
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch("src.activity.connect", return_value=FakeConn()):
            from src.activity import list_events
            result = list_events(limit=10, days=7)

        assert len(result) == 2
        assert result[0]["event_type"] == "routine"
        assert result[0]["icon"] == "⏱"
        assert result[1]["icon"] == "👁"

    def test_list_events_icon_for_notification(self):
        from datetime import datetime, timezone
        ts = datetime(2026, 1, 15, tzinfo=timezone.utc)

        class FakeCur:
            def execute(self, *a): pass
            def fetchall(self): return [(3, ts, "notification", "Alert", "body")]
            def __enter__(self): return self
            def __exit__(self, *a): pass

        class FakeConn:
            def cursor(self): return FakeCur()
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch("src.activity.connect", return_value=FakeConn()):
            from src.activity import list_events
            result = list_events()
        assert result[0]["icon"] == "🔔"


# ── Wiring: scheduler logs routine events ────────────────────────────────────

class TestSchedulerWiring:
    def test_routine_run_calls_log_event(self):
        logged = []
        with patch("src.assistant_starter.run", return_value="All done"), \
             patch("src.api._system_prompt", return_value=""), \
             patch("src.memory.search", return_value=""), \
             patch("src.routines.record_run"), \
             patch("src.output_bus.deliver"), \
             patch("src.scheduler._send_notification"), \
             patch("src.scheduler._send_telegram"), \
             patch("src.activity.log_event", side_effect=lambda *a: logged.append(a)):

            from src.scheduler import _run_routine
            _run_routine(1, "Morning briefing", "give me a briefing", "none")

        assert len(logged) == 1
        assert logged[0][0] == "routine"
        assert "Morning briefing" in logged[0][1]

    def test_routine_error_still_logs(self):
        logged = []
        with patch("src.assistant_starter.run", side_effect=Exception("timeout")), \
             patch("src.api._system_prompt", return_value=""), \
             patch("src.memory.search", return_value=""), \
             patch("src.routines.record_run"), \
             patch("src.output_bus.deliver"), \
             patch("src.scheduler._send_notification"), \
             patch("src.scheduler._send_telegram"), \
             patch("src.activity.log_event", side_effect=lambda *a: logged.append(a)):

            from src.scheduler import _run_routine
            _run_routine(1, "Morning briefing", "give me a briefing", "none")

        assert len(logged) == 1
        assert "error" in logged[0][2].lower()


# ── Wiring: output_bus logs notification deliveries ──────────────────────────

class TestOutputBusWiring:
    def test_deliver_logs_notification_event(self):
        logged = []
        with patch("src.output_bus._send_notification"), \
             patch("src.activity.log_event", side_effect=lambda *a: logged.append(a)):

            from src.output_bus import deliver
            deliver("My Routine", "Here is your briefing.", channel="notification")

        assert len(logged) == 1
        assert logged[0][0] == "notification"
        assert "My Routine" in logged[0][1]

    def test_deliver_skips_log_for_api_channel(self):
        logged = []
        with patch("src.activity.log_event", side_effect=lambda *a: logged.append(a)):
            from src.output_bus import deliver
            deliver("title", "body", channel="api")

        assert logged == []


# ── Wiring: watchers log triggered events ────────────────────────────────────

class TestWatchersWiring:
    def test_triggered_watcher_logs_event(self):
        logged = []
        watcher = {
            "id": 1, "name": "BTC Alert", "prompt": "what is BTC price?",
            "condition": "above 50000", "notify_via": "notification",
        }
        with patch("src.assistant_starter.run", return_value="BTC is $55000"), \
             patch("src.api._system_prompt", return_value=""), \
             patch("src.memory.search", return_value=""), \
             patch("src.watchers._check_condition", return_value=True), \
             patch("src.watchers._record_check"), \
             patch("src.output_bus.deliver"), \
             patch("src.activity.log_event", side_effect=lambda *a: logged.append(a)):

            from src.watchers import evaluate
            evaluate(watcher)

        # At least one watcher log_event call
        watcher_logs = [l for l in logged if l[0] == "watcher"]
        assert len(watcher_logs) >= 1
        assert "BTC Alert" in watcher_logs[0][1]

    def test_untriggered_watcher_does_not_log(self):
        logged = []
        watcher = {
            "id": 2, "name": "Rain Alert", "prompt": "is it raining?",
            "condition": "raining", "notify_via": "notification",
        }
        with patch("src.assistant_starter.run", return_value="No rain today"), \
             patch("src.api._system_prompt", return_value=""), \
             patch("src.memory.search", return_value=""), \
             patch("src.watchers._check_condition", return_value=False), \
             patch("src.watchers._record_check"), \
             patch("src.activity.log_event", side_effect=lambda *a: logged.append(a)):

            from src.watchers import evaluate
            evaluate(watcher)

        watcher_logs = [l for l in logged if l[0] == "watcher"]
        assert watcher_logs == []


# ── REST endpoint ─────────────────────────────────────────────────────────────

class TestActivityEndpoint:
    def test_returns_list(self, auth):
        with patch("src.activity.list_events", return_value=[]):
            r = auth.get("/activity")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_returns_events(self, auth):
        fake = [
            {"id": 1, "created_at": "2026-01-15T10:00:00+00:00",
             "event_type": "routine", "icon": "⏱", "title": "Briefing", "body": "done"},
        ]
        with patch("src.activity.list_events", return_value=fake):
            r = auth.get("/activity")
        assert r.json()[0]["title"] == "Briefing"

    def test_passes_days_param(self, auth):
        captured = []
        def fake_list(limit=100, days=7):
            captured.append(days)
            return []
        with patch("src.activity.list_events", side_effect=fake_list):
            auth.get("/activity?days=30")
        assert captured[0] == 30

    def test_requires_auth(self):
        from fastapi.testclient import TestClient
        from src.api import app
        client = TestClient(app)
        r = client.get("/activity")
        assert r.status_code in (401, 403)
