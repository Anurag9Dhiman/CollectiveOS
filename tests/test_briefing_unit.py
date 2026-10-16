"""Tests for src/briefing.py — unit tests with mocked connectors and Gemini."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import src.briefing as briefing


@pytest.fixture(autouse=True)
def _tmp_config(tmp_path, monkeypatch):
    """Redirect the config file to a temp path so tests don't pollute the repo."""
    config_file = str(tmp_path / "briefing_config.json")
    monkeypatch.setattr(briefing, "_CONFIG_FILE", config_file)
    yield config_file


class TestGetConfig:
    def test_returns_defaults_when_no_file(self):
        cfg = briefing.get_config()
        assert cfg["enabled"] is False
        assert cfg["hour"] == 8
        assert cfg["minute"] == 0
        assert cfg["timezone"] == "UTC"
        assert cfg["notify_via"] == "notification"

    def test_merges_saved_values(self, _tmp_config):
        with open(_tmp_config, "w") as f:
            json.dump({"enabled": True, "hour": 7}, f)
        cfg = briefing.get_config()
        assert cfg["enabled"] is True
        assert cfg["hour"] == 7
        assert cfg["minute"] == 0  # still default

    def test_handles_corrupt_file(self, _tmp_config):
        with open(_tmp_config, "w") as f:
            f.write("not valid json{{{")
        cfg = briefing.get_config()
        assert cfg["hour"] == 8  # falls back to defaults


class TestSetConfig:
    def test_saves_and_returns_merged_config(self, _tmp_config):
        result = briefing.set_config({"enabled": True, "hour": 9})
        assert result["enabled"] is True
        assert result["hour"] == 9

    def test_clamps_hour_to_valid_range(self):
        result = briefing.set_config({"hour": 99})
        assert result["hour"] == 23

    def test_clamps_hour_below_zero(self):
        result = briefing.set_config({"hour": -5})
        assert result["hour"] == 0

    def test_clamps_minute_to_valid_range(self):
        result = briefing.set_config({"minute": 100})
        assert result["minute"] == 59

    def test_persists_to_file(self, _tmp_config):
        briefing.set_config({"hour": 6})
        with open(_tmp_config) as f:
            saved = json.load(f)
        assert saved["hour"] == 6


class TestScheduleEnabled:
    def test_false_by_default(self):
        assert briefing.schedule_enabled() is False

    def test_true_when_enabled(self):
        briefing.set_config({"enabled": True})
        assert briefing.schedule_enabled() is True

    def test_false_when_disabled(self):
        briefing.set_config({"enabled": False})
        assert briefing.schedule_enabled() is False


class TestFallbackText:
    def test_includes_date(self):
        sections = {"date": "Monday, January 1, 2026"}
        text = briefing._fallback_text(sections)
        assert "Monday, January 1, 2026" in text

    def test_includes_health_when_present(self):
        sections = {"date": "today", "health": "Readiness 85, HRV 45"}
        text = briefing._fallback_text(sections)
        assert "Readiness 85, HRV 45" in text

    def test_includes_memory_when_present(self):
        sections = {"date": "today", "memory": "User prefers dark roast coffee"}
        text = briefing._fallback_text(sections)
        assert "User prefers dark roast coffee" in text

    def test_skips_none_sections(self):
        sections = {"date": "today", "health": None, "memory": None}
        text = briefing._fallback_text(sections)
        assert "None" not in text
        assert "Good morning" in text

    def test_truncates_long_health(self):
        sections = {"date": "today", "health": "x" * 500}
        text = briefing._fallback_text(sections)
        assert len(text) < 400


class TestSynthesize:
    def test_calls_fallback_when_no_api_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "")
        sections = {"date": "today"}
        result = briefing._synthesize(sections)
        assert "Good morning" in result

    def test_calls_gemini_when_key_present(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        mock_resp = MagicMock()
        mock_resp.text = "Good morning! Here is your briefing."
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp
        with patch("google.genai.Client", return_value=mock_client):
            sections = {"date": "today", "health": None, "memory": None}
            result = briefing._synthesize(sections)
        assert "Good morning" in result

    def test_falls_back_when_gemini_raises(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        with patch("google.genai.Client", side_effect=RuntimeError("API error")):
            sections = {"date": "today"}
            result = briefing._synthesize(sections)
        assert isinstance(result, str)
        assert len(result) > 0


class TestGenerate:
    def test_returns_required_keys(self):
        with patch.object(briefing, "_get_health", return_value=None), \
             patch.object(briefing, "_get_memory_context", return_value=None), \
             patch.object(briefing, "_synthesize", return_value="Test briefing"):
            result = briefing.generate()
        assert "date" in result
        assert "sections" in result
        assert "briefing" in result
        assert "generated_at" in result

    def test_briefing_text_is_from_synthesize(self):
        with patch.object(briefing, "_get_health", return_value=None), \
             patch.object(briefing, "_get_memory_context", return_value=None), \
             patch.object(briefing, "_synthesize", return_value="Custom briefing text"):
            result = briefing.generate()
        assert result["briefing"] == "Custom briefing text"

    def test_sections_include_data_from_connectors(self):
        with patch.object(briefing, "_get_health", return_value="Readiness 90"), \
             patch.object(briefing, "_get_memory_context", return_value="Prefers coffee"), \
             patch.object(briefing, "_synthesize", return_value="x"):
            result = briefing.generate()
        assert result["sections"]["health"] == "Readiness 90"
        assert result["sections"]["memory"] == "Prefers coffee"

    def test_get_health_handles_connector_exception(self):
        """_get_health wraps connector errors and returns None so generate() never crashes."""
        with patch("src.connectors.health.health_get_readiness", side_effect=RuntimeError("no device")):
            result = briefing._get_health()
        assert result is None


class TestDeliver:
    def test_calls_output_bus_deliver(self):
        with patch.object(briefing, "generate", return_value={
                "briefing": "Good morning!",
                "date": "today",
                "sections": {},
                "generated_at": "now",
             }), \
             patch("src.output_bus.deliver") as mock_deliver:
            briefing.deliver()
        mock_deliver.assert_called_once()

    def test_uses_configured_channel(self):
        briefing.set_config({"notify_via": "telegram"})
        with patch.object(briefing, "generate", return_value={
                "briefing": "Morning!",
                "date": "today",
                "sections": {},
                "generated_at": "now",
             }), \
             patch("src.output_bus.deliver") as mock_deliver:
            briefing.deliver()
        _, kwargs = mock_deliver.call_args
        assert kwargs.get("channel") == "telegram"
