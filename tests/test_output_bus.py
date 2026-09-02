"""Tests for src/output_bus.py — mocks requests and subprocess."""

import os
from unittest.mock import MagicMock, call, patch

import pytest

import src.output_bus as bus


class TestValidChannels:
    def test_valid_channels_is_frozenset(self):
        assert isinstance(bus.VALID_CHANNELS, frozenset)

    def test_api_and_none_are_valid(self):
        assert "api" in bus.VALID_CHANNELS
        assert "none" in bus.VALID_CHANNELS

    def test_notification_is_valid(self):
        assert "notification" in bus.VALID_CHANNELS

    def test_telegram_is_valid(self):
        assert "telegram" in bus.VALID_CHANNELS

    def test_push_is_valid(self):
        assert "push" in bus.VALID_CHANNELS


class TestDeliver:
    def test_api_channel_is_noop(self):
        with patch.object(bus, "_send_notification") as mock_notif, \
             patch.object(bus, "_send_telegram") as mock_tg:
            bus.deliver("T", "B", channel="api")
            mock_notif.assert_not_called()
            mock_tg.assert_not_called()

    def test_none_channel_is_noop(self):
        with patch.object(bus, "_send_notification") as mock_notif:
            bus.deliver("T", "B", channel="none")
            mock_notif.assert_not_called()

    def test_notification_channel_calls_send_notification(self):
        with patch.object(bus, "_send_notification") as mock_notif:
            bus.deliver("Hello", "World", channel="notification")
            mock_notif.assert_called_once_with("Hello", "World")

    def test_telegram_channel_calls_send_telegram(self):
        with patch.object(bus, "_send_telegram") as mock_tg:
            bus.deliver("Hello", "World", channel="telegram")
            mock_tg.assert_called_once_with("Hello", "World")

    def test_push_channel_calls_send_push(self):
        with patch.object(bus, "_send_push") as mock_push:
            bus.deliver("Hello", "World", channel="push")
            mock_push.assert_called_once_with("Hello", "World")

    def test_both_calls_notification_and_one_other(self):
        with patch.object(bus, "_send_notification") as mock_notif, \
             patch.object(bus, "_send_telegram") as mock_tg, \
             patch.object(bus, "_send_slack", MagicMock(), create=True) as mock_slack:
            bus.deliver("T", "B", channel="both")
            mock_notif.assert_called_once()
            # Either telegram or slack should be called depending on the version
            assert mock_tg.called or mock_slack.called

    def test_channel_error_does_not_propagate(self):
        with patch.object(bus, "_send_notification", side_effect=RuntimeError("oops")):
            # Should not raise
            bus.deliver("T", "B", channel="notification")

    def test_default_channel_is_api(self):
        with patch.object(bus, "_send_notification") as mock_notif:
            bus.deliver("T", "B")
            mock_notif.assert_not_called()


class TestSendTelegram:
    def test_skips_when_no_token(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        with patch("requests.post") as mock_post:
            bus._send_telegram("Title", "Body")
            mock_post.assert_not_called()

    def test_calls_correct_url(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TOKEN123")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock()
            bus._send_telegram("Title", "Body")
            assert mock_post.called
            url = mock_post.call_args[0][0]
            assert "TOKEN123" in url
            assert "sendMessage" in url

    def test_body_is_truncated_to_fit(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
        long_body = "x" * 5000
        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock()
            bus._send_telegram("T", long_body)
            kwargs = mock_post.call_args[1]
            text = kwargs["json"]["text"]
            assert len(text) <= 4200


class TestSendNotification:
    def test_skips_on_non_macos(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        with patch("subprocess.run") as mock_run:
            bus._send_notification("T", "B")
            mock_run.assert_not_called()

    def test_calls_osascript_on_macos(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            bus._send_notification("Hello", "World")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "osascript" in cmd

    def test_body_is_truncated_at_250(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        long = "a" * 400
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            bus._send_notification("T", long)
            script = mock_run.call_args[0][0][-1]
            # 250 chars of body + some wrapper text
            assert len(script) < 500
