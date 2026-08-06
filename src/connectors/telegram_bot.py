"""
Telegram connector — read messages and send replies via the Bot API.

Setup
-----
1. Message @BotFather on Telegram → /newbot → copy the token.
2. Add to .env:  TELEGRAM_BOT_TOKEN=123456:ABCdef...
3. Message your bot once in Telegram so it has your chat_id.
4. Optionally add TELEGRAM_CHAT_ID=<your chat id> to .env so the assistant
   can message you directly without needing an explicit chat_id.

The bot token authenticates all requests. No OAuth flow needed.

Read  tool : telegram_get_messages
Write tool : telegram_send   (confirm recipient + content before calling)
"""

import datetime
import os

import requests

_BASE = "https://api.telegram.org/bot"

# Module-level offset — avoids re-delivering the same updates within a server
# session. Resets to 0 on restart, which is fine (re-reads pending messages).
_offset: int = 0


def _token() -> str:
    t = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not t:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. "
            "Create a bot with @BotFather, copy the token, "
            "and add it to your .env file."
        )
    return t


def _call(method: str, **params) -> dict:
    url = f"{_BASE}{_token()}/{method}"
    resp = requests.post(url, json=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_messages(limit: int = 10) -> str:
    """
    Return recent messages sent to this Telegram bot.

    Polls the Telegram Bot API for pending updates, then confirms them so
    the same messages aren't returned on the next call. Returns the last
    `limit` messages (default 10) with sender name, chat_id, and text.

    Requires TELEGRAM_BOT_TOKEN in the environment.
    """
    global _offset

    try:
        data = _call("getUpdates", offset=_offset, limit=limit, timeout=0)
    except RuntimeError as e:
        return str(e)
    except requests.RequestException as e:
        return f"Telegram API error: {e}"

    if not data.get("ok"):
        return f"Telegram error: {data.get('description', 'unknown')}"

    updates = data.get("result", [])
    if not updates:
        return "No new messages in your Telegram bot inbox."

    # Advance offset so these updates are confirmed on the next call
    _offset = updates[-1]["update_id"] + 1

    lines = []
    for u in updates:
        msg = u.get("message") or u.get("edited_message") or u.get("channel_post")
        if not msg:
            continue

        sender   = msg.get("from") or msg.get("sender_chat") or {}
        name     = " ".join(filter(None, [
            sender.get("first_name", ""),
            sender.get("last_name", ""),
        ])).strip() or sender.get("username") or sender.get("title") or "Unknown"
        chat_id  = msg.get("chat", {}).get("id", "")
        text     = msg.get("text") or msg.get("caption") or "(non-text)"
        ts       = datetime.datetime.fromtimestamp(
            msg.get("date", 0),
            tz=datetime.timezone.utc,
        ).astimezone().strftime("%b %-d %H:%M")

        lines.append(f"[{ts}] {name} (chat_id={chat_id}): {text}")

    if not lines:
        return "No text messages found in recent updates."

    return f"{len(lines)} message(s):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def send_message(chat_id: str, text: str) -> str:
    """
    Send a message to a Telegram chat.

    Always confirm the recipient (chat_id) and message text with the user
    before calling. Use Markdown formatting if helpful.

    - chat_id: Telegram chat id — visible in get_messages output, or set
               TELEGRAM_CHAT_ID in .env for your personal chat.
    - text:    Message to send (plain text, up to 4096 chars).
    """
    # Fall back to env-configured personal chat_id if not specified
    resolved = str(chat_id).strip() or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not resolved:
        return (
            "No chat_id provided and TELEGRAM_CHAT_ID is not set in .env. "
            "Pass an explicit chat_id, or add your personal chat_id to .env."
        )

    try:
        result = _call("sendMessage", chat_id=resolved, text=text[:4096])
    except RuntimeError as e:
        return str(e)
    except requests.RequestException as e:
        return f"Telegram API error: {e}"

    if result.get("ok"):
        return f"Telegram message sent to chat {resolved}."
    return f"Failed to send: {result.get('description', 'unknown error')}"
