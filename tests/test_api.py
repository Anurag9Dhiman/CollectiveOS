"""Integration-style tests for src/api.py via FastAPI TestClient."""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client):
        with patch("src.api._cache") as mock_cache:
            mock_cache.ping.return_value = True
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    def test_chat_requires_token(self, client):
        resp = client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 401

    def test_chat_rejects_wrong_token(self, client):
        resp = client.post(
            "/chat",
            json={"message": "hello"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_ask_rejects_missing_token(self, client):
        resp = client.get("/ask?q=hello")
        assert resp.status_code == 401

    def test_ask_accepts_query_param_token(self, client):
        resp = client.get("/ask?q=hello&token=test-token")
        assert resp.status_code == 200

    def test_chat_accepts_valid_bearer(self, auth):
        resp = auth.post("/chat", json={"message": "hello"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------

class TestAsk:
    def test_ask_returns_plain_text(self, auth):
        resp = auth.client.get("/ask?q=hello&token=test-token") \
               if hasattr(auth, "client") else \
               auth.get("/ask?q=hello&token=test-token")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")

    def test_ask_rejects_empty_q(self, client):
        resp = client.get("/ask?q=&token=test-token")
        assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------

class TestChat:
    def test_chat_returns_reply(self, auth):
        resp = auth.post("/chat", json={"message": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert "reply" in data
        assert "conversation_id" in data

    def test_chat_rejects_empty_message(self, auth):
        resp = auth.post("/chat", json={"message": "   "})
        assert resp.status_code == 400

    def test_chat_returns_conversation_id(self, auth):
        resp = auth.post("/chat", json={"message": "hello"})
        assert isinstance(resp.json()["conversation_id"], int)

    def test_chat_interrupted_field_present(self, auth):
        resp = auth.post("/chat", json={"message": "send email"})
        assert "interrupted" in resp.json()

    def test_chat_rejects_invalid_image_mime(self, auth):
        resp = auth.post("/chat", json={
            "message": "look at this",
            "image_b64": "abc123",
            "image_mime": "text/plain",
        })
        assert resp.status_code == 400

    def test_chat_accepts_conversation_id(self, auth):
        resp = auth.post("/chat", json={"message": "hello", "conversation_id": 42})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /chat/approve
# ---------------------------------------------------------------------------

class TestApprove:
    def test_approve_returns_reply(self, auth):
        resp = auth.post("/chat/approve", json={"conversation_id": 1, "approved": True})
        assert resp.status_code == 200
        assert "reply" in resp.json()

    def test_cancel_returns_reply(self, auth):
        resp = auth.post("/chat/approve", json={"conversation_id": 1, "approved": False})
        assert resp.status_code == 200
        assert "reply" in resp.json()

    def test_approve_requires_auth(self, client):
        resp = client.post("/chat/approve", json={"conversation_id": 1, "approved": True})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /conversations
# ---------------------------------------------------------------------------

class TestConversations:
    def test_list_conversations_requires_auth(self, client):
        resp = client.get("/conversations")
        assert resp.status_code == 401

    def test_list_conversations_returns_list(self, auth):
        resp = auth.get("/conversations")
        assert resp.status_code == 200
        assert "conversations" in resp.json()
        assert isinstance(resp.json()["conversations"], list)


# ---------------------------------------------------------------------------
# /conversations/search (Phase 11: full-text search)
# ---------------------------------------------------------------------------

class TestSearchConversations:
    def test_search_requires_auth(self, client):
        resp = client.get("/conversations/search?q=hello")
        assert resp.status_code == 401

    def test_search_returns_hits_key(self, auth, monkeypatch):
        import src.conversations as conv
        monkeypatch.setattr(conv, "search_messages", lambda q, limit=20: [], raising=False)
        resp = auth.get("/conversations/search?q=test")
        assert resp.status_code == 200
        body = resp.json()
        assert "hits" in body
        assert "query" in body

    def test_search_rejects_empty_query(self, auth):
        resp = auth.get("/conversations/search?q=")
        assert resp.status_code == 422

    def test_search_forwards_limit(self, auth, monkeypatch):
        import src.conversations as conv
        captured = {}
        def fake_search(q, limit=20):
            captured["limit"] = limit
            return []
        monkeypatch.setattr(conv, "search_messages", fake_search, raising=False)
        auth.get("/conversations/search?q=test&limit=5")
        assert captured.get("limit") == 5


# ---------------------------------------------------------------------------
# /routines
# ---------------------------------------------------------------------------

class TestRoutines:
    def test_list_routines_requires_auth(self, client):
        resp = client.get("/routines")
        assert resp.status_code == 401

    def test_list_routines_returns_list(self, auth, monkeypatch):
        import src.routines as _routines
        monkeypatch.setattr(_routines, "list_all", lambda: [], raising=False)
        resp = auth.get("/routines")
        assert resp.status_code == 200
        assert "routines" in resp.json()

    def test_create_routine_validates_cron(self, auth, monkeypatch):
        import src.routines as _routines
        monkeypatch.setattr(_routines, "create", lambda *a, **kw: {"id": 1}, raising=False)
        # CronTrigger is globally mocked; patch from_crontab to raise for bad input
        with patch("apscheduler.triggers.cron.CronTrigger.from_crontab",
                   side_effect=ValueError("bad cron")):
            resp = auth.post("/routines", json={
                "name": "test",
                "prompt": "check email",
                "schedule": "not-a-cron",
            })
        assert resp.status_code == 400

    def test_create_routine_validates_notify_via(self, auth, monkeypatch):
        import src.routines as _routines
        monkeypatch.setattr(_routines, "create", lambda *a, **kw: {"id": 1}, raising=False)
        resp = auth.post("/routines", json={
            "name": "test",
            "prompt": "check email",
            "schedule": "0 8 * * *",
            "notify_via": "carrier_pigeon",
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /permissions
# ---------------------------------------------------------------------------

class TestPermissions:
    def test_get_permissions_requires_auth(self, client):
        resp = client.get("/permissions")
        assert resp.status_code == 401

    def test_get_permissions_returns_dict(self, auth, monkeypatch):
        import src.permissions as perm
        monkeypatch.setattr(perm, "list_all", lambda: {}, raising=False)
        resp = auth.get("/permissions")
        assert resp.status_code == 200
        assert "permissions" in resp.json()


# ---------------------------------------------------------------------------
# /briefing
# ---------------------------------------------------------------------------

class TestBriefing:
    def test_get_briefing_requires_auth(self, client):
        resp = client.get("/briefing")
        assert resp.status_code == 401

    def test_get_briefing_returns_data(self, auth, monkeypatch):
        import src.briefing as briefing
        monkeypatch.setattr(briefing, "generate", lambda: {
            "date": "today", "sections": {}, "briefing": "Good morning!", "generated_at": "now"
        }, raising=False)
        resp = auth.get("/briefing")
        assert resp.status_code == 200

    def test_get_briefing_schedule(self, auth, monkeypatch):
        import src.briefing as briefing
        monkeypatch.setattr(briefing, "get_config", lambda: {"enabled": False, "hour": 8}, raising=False)
        resp = auth.get("/briefing/schedule")
        assert resp.status_code == 200

    def test_post_briefing_schedule(self, auth, monkeypatch):
        import src.briefing as briefing
        monkeypatch.setattr(briefing, "set_config", lambda p: {"enabled": True, "hour": 7}, raising=False)
        monkeypatch.setattr(briefing, "register_job", lambda s: None, raising=False)
        resp = auth.post("/briefing/schedule", json={"enabled": True, "hour": 7})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /robot
# ---------------------------------------------------------------------------

class TestRobot:
    def test_robot_status_requires_auth(self, client):
        resp = client.get("/robot/status")
        assert resp.status_code == 401

    def test_robot_status_returns_state(self, auth, monkeypatch):
        import src.connectors.ros2_sim as sim
        monkeypatch.setattr(sim, "get_state", lambda: {"room": "hallway", "battery": 90}, raising=False)
        resp = auth.get("/robot/status")
        assert resp.status_code == 200
