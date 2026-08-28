"""
Output bus — single point for routing agent output to delivery channels.

All delivery logic that was previously scattered across scheduler.py
and individual connectors lives here. Callers pass (title, body, channel)
and the bus handles dispatch, including fan-out for "both".

Supported channels:
  notification  — Mac notification via osascript (macOS only)
  telegram      — Telegram bot message to TELEGRAM_CHAT_ID
  push          — iOS APNs push notification
  both          — Mac notification + Telegram simultaneously
  api           — no side effect; caller handles the text (default)
  none          — silently discard

Adding a new channel: add a branch in deliver() and a private _send_*()
helper below. No other files need changing.
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess

log = logging.getLogger("collectiveos.output_bus")

VALID_CHANNELS = frozenset({"notification", "telegram", "push", "both", "api", "none"})


def deliver(title: str, body: str, channel: str = "api") -> None:
    """Route (title, body) to the specified delivery channel(s).

    Errors in individual channels are logged and swallowed so one broken
    channel never silences another.
    """
    if channel in ("api", "none"):
        return

    if channel in ("notification", "both"):
        try:
            _send_notification(title, body)
        except Exception as exc:
            log.warning("Mac notification failed: %s", exc)

    if channel in ("telegram", "both"):
        try:
            _send_telegram(title, body)
        except Exception as exc:
            log.warning("Telegram delivery failed: %s", exc)

    if channel == "push":
        try:
            _send_push(title, body)
        except Exception as exc:
            log.warning("iOS push delivery failed: %s", exc)


# ---------------------------------------------------------------------------
# Channel implementations
# ---------------------------------------------------------------------------

def _send_notification(title: str, body: str) -> None:
    if platform.system() != "Darwin":
        log.debug("Mac notification skipped — not macOS")
        return
    safe_title = title.replace('"', "'")
    safe_body = body[:250].replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{safe_body}" with title "{safe_title}"'
    subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10, check=False)


def _send_telegram(title: str, body: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        log.warning("Telegram delivery skipped — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        return
    import requests
    max_body = 4000 - len(title) - 10
    text = f"*{title}*\n\n{body[:max_body]}"
    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        timeout=15,
    )


def _send_push(title: str, body: str) -> None:
    from src.connectors.ios_push import push_notification
    push_notification(title=title, body=body[:200])
