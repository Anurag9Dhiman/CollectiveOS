"""
Tests for Phase 5: Robot learning stream and demonstration manager.
"""

from __future__ import annotations

import asyncio
import base64
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Demonstration manager ─────────────────────────────────────────────────────

class TestDemoManager:
    def _write_demo(self, tmp_dir: Path, name: str, task: str, n_steps: int) -> Path:
        data = {
            "task": task,
            "steps": n_steps,
            "demos": [
                {"action": {"action": "click", "x": 100, "y": 200}, "app": "Finder", "timestamp": 1.0}
                for _ in range(n_steps)
            ],
        }
        p = tmp_dir / f"{name}.json"
        p.write_text(json.dumps(data))
        return p

    def test_list_demos_empty_when_no_dir(self, tmp_path):
        with patch("src.connectors.demonstrations._DEMO_DIR", tmp_path / "nonexistent"):
            from src.connectors.demonstrations import list_demos
            assert list_demos() == []

    def test_list_demos_returns_metadata(self, tmp_path):
        self._write_demo(tmp_path, "demo_1000", "click the button", 3)
        with patch("src.connectors.demonstrations._DEMO_DIR", tmp_path):
            from src.connectors.demonstrations import list_demos
            result = list_demos()
        assert len(result) == 1
        assert result[0]["task"] == "click the button"
        assert result[0]["steps"] == 3
        assert result[0]["id"] == "demo_1000"

    def test_list_demos_respects_limit(self, tmp_path):
        for i in range(5):
            self._write_demo(tmp_path, f"demo_{1000 + i}", f"task {i}", 1)
        with patch("src.connectors.demonstrations._DEMO_DIR", tmp_path):
            from src.connectors.demonstrations import list_demos
            result = list_demos(limit=2)
        assert len(result) == 2

    def test_get_demo_returns_data(self, tmp_path):
        self._write_demo(tmp_path, "demo_9999", "open finder", 2)
        with patch("src.connectors.demonstrations._DEMO_DIR", tmp_path):
            from src.connectors.demonstrations import get_demo
            data = get_demo("demo_9999")
        assert data is not None
        assert data["task"] == "open finder"

    def test_get_demo_returns_none_when_missing(self, tmp_path):
        with patch("src.connectors.demonstrations._DEMO_DIR", tmp_path):
            from src.connectors.demonstrations import get_demo
            assert get_demo("demo_does_not_exist") is None

    def test_summarize_policy_empty(self, tmp_path):
        with patch("src.connectors.demonstrations._DEMO_DIR", tmp_path / "none"):
            from src.connectors.demonstrations import summarize_policy
            summary = summarize_policy()
        assert summary["total_demos"] == 0
        assert summary["total_steps"] == 0
        assert summary["action_counts"] == {}

    def test_summarize_policy_counts_actions(self, tmp_path):
        self._write_demo(tmp_path, "demo_1", "click a", 2)
        self._write_demo(tmp_path, "demo_2", "click b", 3)
        with patch("src.connectors.demonstrations._DEMO_DIR", tmp_path):
            from src.connectors.demonstrations import summarize_policy
            summary = summarize_policy()
        assert summary["total_demos"] == 2
        assert summary["total_steps"] == 5
        assert summary["action_counts"]["click"] == 5

    def test_summarize_policy_includes_tasks(self, tmp_path):
        self._write_demo(tmp_path, "demo_1", "open mail", 1)
        self._write_demo(tmp_path, "demo_2", "click send", 1)
        with patch("src.connectors.demonstrations._DEMO_DIR", tmp_path):
            from src.connectors.demonstrations import summarize_policy
            summary = summarize_policy()
        assert "open mail" in summary["tasks"] or "click send" in summary["tasks"]


# ── Robot stream handler ──────────────────────────────────────────────────────

class TestRobotStreamHandler:
    def _make_ws(self, messages: list[str]) -> MagicMock:
        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()
        # yield messages then raise WebSocketDisconnect
        from fastapi import WebSocketDisconnect
        receive_values = [AsyncMock(return_value=m)() for m in messages]
        receive_values.append(asyncio.coroutine(lambda: (_ for _ in ()).throw(WebSocketDisconnect()))())
        ws.receive_text = AsyncMock(side_effect=[m for m in messages] + [WebSocketDisconnect()])
        return ws

    @pytest.mark.asyncio
    async def test_ping_pong(self):
        from fastapi import WebSocketDisconnect
        from src.robot_stream import handle_robot_ws

        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()
        ws.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "ping"}),
            WebSocketDisconnect(),
        ])

        await handle_robot_ws(ws, "token")

        sent = [json.loads(c.args[0]) for c in ws.send_text.call_args_list]
        types = [m["type"] for m in sent]
        assert "pong" in types

    @pytest.mark.asyncio
    async def test_frame_message_stores_frame(self):
        from fastapi import WebSocketDisconnect
        from src.robot_stream import handle_robot_ws

        fake_b64 = base64.b64encode(b"\xff\xd8\xff").decode()
        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()
        ws.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "frame", "image_b64": fake_b64}),
            WebSocketDisconnect(),
        ])

        await handle_robot_ws(ws, "token")

        sent = [json.loads(c.args[0]) for c in ws.send_text.call_args_list]
        acks = [m for m in sent if m["type"] == "ack"]
        assert any("Frame" in m.get("message", "") for m in acks)

    @pytest.mark.asyncio
    async def test_task_message_triggers_nav_agent(self):
        from fastapi import WebSocketDisconnect
        from src.robot_stream import handle_robot_ws

        fake_result = {"status": "done", "result": "opened door", "steps": 2, "demo_path": "/tmp/demo_1.json"}

        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()
        ws.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "task", "text": "open the door"}),
            WebSocketDisconnect(),
        ])

        with patch("src.robot_stream._run_robot_task", return_value=fake_result):
            await handle_robot_ws(ws, "token")

        sent = [json.loads(c.args[0]) for c in ws.send_text.call_args_list]
        reply = next((m for m in sent if m["type"] == "reply"), None)
        assert reply is not None
        assert reply["triggered"] is True
        assert reply["text"] == "opened door"

    @pytest.mark.asyncio
    async def test_empty_task_returns_error(self):
        from fastapi import WebSocketDisconnect
        from src.robot_stream import handle_robot_ws

        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()
        ws.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "task", "text": "   "}),
            WebSocketDisconnect(),
        ])

        await handle_robot_ws(ws, "token")

        sent = [json.loads(c.args[0]) for c in ws.send_text.call_args_list]
        errors = [m for m in sent if m["type"] == "error"]
        assert len(errors) == 1

    @pytest.mark.asyncio
    async def test_unknown_message_type_returns_error(self):
        from fastapi import WebSocketDisconnect
        from src.robot_stream import handle_robot_ws

        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()
        ws.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "unsupported_type"}),
            WebSocketDisconnect(),
        ])

        await handle_robot_ws(ws, "token")

        sent = [json.loads(c.args[0]) for c in ws.send_text.call_args_list]
        errors = [m for m in sent if m["type"] == "error"]
        assert len(errors) == 1


# ── _run_robot_task: robot camera forwarded to nav agent ─────────────────────

class TestRunRobotTask:
    def test_robot_frame_passed_to_nav_agent(self):
        """robot_camera_frame is forwarded and record=True is set."""
        import src.agents.nav_agent as mod

        captured: dict = {}

        async def fake_run(task, context="", *, hitl_callback=None,
                           first_person_frame=None, robot_camera_frame=None, record=False):
            captured["robot_camera_frame"] = robot_camera_frame
            captured["record"] = record
            return mod.NavResult(status="done", result="ok", steps=[])

        agent = mod._get_agent()
        with patch.object(agent, "run", side_effect=fake_run):
            from src.robot_stream import _run_robot_task
            _run_robot_task("open the door", [], b"robot-jpeg-bytes")

        assert captured["robot_camera_frame"] == b"robot-jpeg-bytes"
        assert captured["record"] is True

    def test_no_frame_still_runs(self):
        """Task without a frame still runs (record=True, frame=None)."""
        import src.agents.nav_agent as mod

        async def fake_run(task, context="", *, hitl_callback=None,
                           first_person_frame=None, robot_camera_frame=None, record=False):
            return mod.NavResult(status="done", result="done", steps=[])

        agent = mod._get_agent()
        with patch.object(agent, "run", side_effect=fake_run):
            from src.robot_stream import _run_robot_task
            result = _run_robot_task("click button", [], None)

        assert result["status"] == "done"


# ── API endpoints ─────────────────────────────────────────────────────────────

class TestDemoAPIEndpoints:
    def test_list_demonstrations_endpoint(self, auth, tmp_path):
        with patch("src.connectors.demonstrations._DEMO_DIR", tmp_path):
            r = auth.get("/demonstrations")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_policy_endpoint(self, auth, tmp_path):
        with patch("src.connectors.demonstrations._DEMO_DIR", tmp_path / "none"):
            r = auth.get("/demonstrations/policy")
        assert r.status_code == 200
        data = r.json()
        assert "total_demos" in data
        assert "action_counts" in data

    def test_get_demonstration_404(self, auth, tmp_path):
        with patch("src.connectors.demonstrations._DEMO_DIR", tmp_path):
            r = auth.get("/demonstrations/demo_does_not_exist")
        assert r.status_code == 404

    def test_get_demonstration_returns_data(self, auth, tmp_path):
        (tmp_path / "demo_42.json").write_text(
            json.dumps({"task": "test task", "steps": 1, "demos": []})
        )
        with patch("src.connectors.demonstrations._DEMO_DIR", tmp_path):
            r = auth.get("/demonstrations/demo_42")
        assert r.status_code == 200
        assert r.json()["task"] == "test task"
