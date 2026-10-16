"""
Tests for Phase 14: Tasks REST API endpoints.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestTasksListEndpoint:
    def test_list_returns_empty_list_on_db_error(self, auth):
        with patch("src.orchestrator.list_tasks", side_effect=Exception("no db")):
            r = auth.get("/tasks")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_returns_tasks(self, auth):
        fake = [
            {"id": 1, "description": "Write a report", "status": "completed", "created_at": "2026-01-01T10:00"},
            {"id": 2, "description": "Send email", "status": "running", "created_at": "2026-01-01T11:00"},
        ]
        with patch("src.orchestrator.list_tasks", return_value=fake):
            r = auth.get("/tasks")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        assert data[0]["id"] == 1

    def test_list_passes_limit(self, auth):
        captured = []
        def fake_list(limit=10):
            captured.append(limit)
            return []
        with patch("src.orchestrator.list_tasks", side_effect=fake_list):
            auth.get("/tasks?limit=5")
        assert captured[0] == 5

    def test_list_requires_auth(self):
        from fastapi.testclient import TestClient
        from src.api import app
        client = TestClient(app)
        r = client.get("/tasks")
        assert r.status_code in (401, 403)


class TestTaskGetEndpoint:
    def test_get_existing_task(self, auth):
        fake = {
            "id": 7,
            "description": "Book restaurant",
            "status": "completed",
            "created_at": "2026-01-01T09:00",
            "steps": [
                {"id": 1, "tool": "search_web", "args": {}, "output": "results", "status": "completed"},
            ],
        }
        with patch("src.orchestrator.get_task", return_value=fake):
            r = auth.get("/tasks/7")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == 7
        assert len(data["steps"]) == 1

    def test_get_missing_task_returns_404(self, auth):
        with patch("src.orchestrator.get_task", return_value={}):
            r = auth.get("/tasks/9999")
        assert r.status_code == 404


class TestTaskCancelEndpoint:
    def test_cancel_pending_task(self, auth):
        with patch("src.orchestrator.cancel_task", return_value="Task #3 cancelled."):
            r = auth.post("/tasks/3/cancel")
        assert r.status_code == 200
        assert "cancelled" in r.json()["message"]

    def test_cancel_already_done_task(self, auth):
        with patch("src.orchestrator.cancel_task",
                   return_value="Task #5 could not be cancelled (already finished or not found)."):
            r = auth.post("/tasks/5/cancel")
        assert r.status_code == 200
        assert "could not" in r.json()["message"]
