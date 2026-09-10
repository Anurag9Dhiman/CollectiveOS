"""
screen_watcher.py
-----------------
Proactive screen awareness — periodically captures a screenshot and asks
the LLM whether anything on screen needs the user's attention.

If the LLM finds something worth surfacing, a Slack notification is sent.
Nothing is sent when everything looks normal — no noise.

Configuration (environment variables):
  SCREEN_WATCHER_ENABLED    — "1" or "true" to enable (default: disabled)
  SCREEN_WATCHER_INTERVAL   — poll interval in seconds (default: 300 = 5 min)
  SCREEN_WATCHER_MODEL      — Gemini vision model (default: VISION_MODEL env or
                               gemini-1.5-flash-latest)

The watcher registers itself with APScheduler via register_job(), called from
scheduler.start() on server startup alongside the briefing job.
"""

from __future__ import annotations

import base64
import logging
import os

log = logging.getLogger(__name__)

_ENABLED_ENVS = {"1", "true", "yes", "on"}
_DEFAULT_INTERVAL = 300  # seconds

_SYSTEM_PROMPT = """You are a proactive personal assistant monitor.
You are shown a screenshot of the user's Mac screen.
Your job: decide whether anything on this screen needs the user's attention RIGHT NOW.

Things worth flagging (be selective — only flag genuinely actionable items):
- A calendar event starting within the next 10 minutes
- A build, CI run, or test that has failed (red status)
- An unread Slack direct message or @mention visible on screen
- A terminal showing an error or a blocking prompt
- A battery warning (below 10% and not charging)
- A dialog or alert box waiting for a response
- A download or long-running process that has finished or failed

Things NOT worth flagging:
- Normal app windows with nothing urgent
- Background content the user can see when they look
- Minor UI badges or general notification counts

Reply in this exact JSON format (no markdown, no extra text):
{
  "alert": true | false,
  "summary": "one-sentence description of what needs attention",
  "urgency": "high" | "medium" | "low"
}

If nothing needs attention, set alert to false and leave summary empty.
"""


def _is_enabled() -> bool:
    return os.environ.get("SCREEN_WATCHER_ENABLED", "").lower() in _ENABLED_ENVS


def _interval() -> int:
    try:
        return int(os.environ.get("SCREEN_WATCHER_INTERVAL", _DEFAULT_INTERVAL))
    except ValueError:
        return _DEFAULT_INTERVAL


def _capture_screenshot() -> bytes | None:
    """Capture the primary display and return raw PNG bytes."""
    try:
        import subprocess
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        subprocess.run(
            ["screencapture", "-x", "-t", "png", tmp],
            check=True, capture_output=True, timeout=10,
        )
        with open(tmp, "rb") as f:
            data = f.read()
        _os.unlink(tmp)
        return data
    except Exception as exc:
        log.warning("screen_watcher: screenshot failed: %s", exc)
        return None


def _ask_llm(png_bytes: bytes) -> dict | None:
    """Send the screenshot to Gemini Vision and parse the JSON response."""
    import json
    from google import genai
    from google.genai import types

    model = os.environ.get(
        "VISION_MODEL",
        os.environ.get("SCREEN_WATCHER_MODEL", "gemini-1.5-flash-latest"),
    )
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        log.warning("screen_watcher: GEMINI_API_KEY not set")
        return None

    try:
        client = genai.Client(api_key=api_key)
        b64 = base64.b64encode(png_bytes).decode()
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part(
                    inline_data=types.Blob(mime_type="image/png", data=b64)
                ),
                types.Part(text="Analyse this screenshot and reply in the JSON format specified."),
            ],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=256,
            ),
        )
        text = (response.text or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as exc:
        log.warning("screen_watcher: LLM call failed: %s", exc)
        return None


def _check_screen() -> None:
    """Single poll cycle — called by APScheduler."""
    if not _is_enabled():
        return

    log.debug("screen_watcher: checking screen")
    png = _capture_screenshot()
    if not png:
        return

    result = _ask_llm(png)
    if not result or not result.get("alert"):
        return

    summary = result.get("summary", "Something on your screen needs attention.")
    urgency = result.get("urgency", "medium")
    urgency_prefix = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(urgency, "")

    try:
        from src import output_bus
        output_bus.deliver(
            title="Screen alert",
            body=f"{urgency_prefix} {summary}",
            channel="slack",
        )
        log.info("screen_watcher: alerted — %s", summary)
    except Exception as exc:
        log.warning("screen_watcher: delivery failed: %s", exc)


def register_job(scheduler) -> None:
    """Register the screen-watcher poll with an APScheduler instance."""
    if not _is_enabled():
        log.info("screen_watcher: disabled (set SCREEN_WATCHER_ENABLED=1 to enable)")
        return

    interval = _interval()
    scheduler.add_job(
        _check_screen,
        "interval",
        seconds=interval,
        id="screen_watcher",
        replace_existing=True,
    )
    log.info("screen_watcher: registered — polling every %d s", interval)


def status() -> dict:
    """Return current watcher configuration (for the API)."""
    return {
        "enabled": _is_enabled(),
        "interval_seconds": _interval(),
        "model": os.environ.get(
            "VISION_MODEL",
            os.environ.get("SCREEN_WATCHER_MODEL", "gemini-1.5-flash-latest"),
        ),
    }
