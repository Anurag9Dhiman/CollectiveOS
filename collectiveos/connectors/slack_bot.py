"""
Slack connector — list channels, read messages, send messages.

Env vars:
  SLACK_BOT_TOKEN — Bot User OAuth Token (xoxb-...)
                    OAuth scopes needed:
                      channels:read, groups:read       (list channels)
                      channels:history, groups:history,
                      im:history, mpim:history         (read messages)
                      chat:write                       (send messages)
                      users:read                       (resolve usernames)
                      im:write                         (open DM conversations)
"""

import datetime
import os

import requests

_BASE = "https://slack.com/api/"
_user_cache: dict[str, str] = {}


def _headers() -> dict[str, str]:
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _check_token() -> str | None:
    if not os.environ.get("SLACK_BOT_TOKEN"):
        return "SLACK_BOT_TOKEN is not set. Add it to .env and restart."
    return None


def _get(method: str, params: dict | None = None) -> dict:
    resp = requests.get(f"{_BASE}{method}", headers=_headers(), params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "unknown Slack error"))
    return data


def _post(method: str, body: dict) -> dict:
    resp = requests.post(f"{_BASE}{method}", headers=_headers(), json=body, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "unknown Slack error"))
    return data


def _ts_to_str(ts: str) -> str:
    try:
        return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


def _get_user_name(user_id: str) -> str:
    if not user_id:
        return "unknown"
    if user_id in _user_cache:
        return _user_cache[user_id]
    try:
        data = _get("users.info", {"user": user_id})
        user = data.get("user", {})
        name = user.get("real_name") or user.get("name", user_id)
        _user_cache[user_id] = name
        return name
    except Exception:
        _user_cache[user_id] = user_id
        return user_id


def _resolve_channel(name: str) -> tuple[str, str | None]:
    """Return (channel_id, error). Accepts C/G/D IDs or #channel-name."""
    name = name.lstrip("#").strip()
    if not name:
        return "", "Channel name or ID is required."
    # Already looks like a Slack channel/DM/group ID
    if len(name) >= 9 and name[0].upper() in ("C", "G", "D", "W"):
        return name, None
    # Look up by name
    try:
        data = _get("conversations.list", {
            "types": "public_channel,private_channel",
            "limit": 200,
            "exclude_archived": "true",
        })
        for ch in data.get("channels", []):
            if ch.get("name", "").lower() == name.lower():
                return ch["id"], None
        return "", f"Channel not found: #{name}. Use slack_list_channels to see available channels."
    except Exception as exc:
        return "", f"Channel lookup error: {exc}"


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def slack_list_channels(limit: int = 30) -> str:
    """List Slack channels the bot has access to."""
    err = _check_token()
    if err:
        return err
    try:
        data = _get("conversations.list", {
            "types": "public_channel,private_channel",
            "limit": min(limit, 200),
            "exclude_archived": "true",
        })
        channels = data.get("channels", [])
        if not channels:
            return "No channels found. Make sure the bot is added to at least one channel."
        lines = []
        for ch in channels[:limit]:
            marker = "🔒" if ch.get("is_private") else "#"
            members = ch.get("num_members", "?")
            topic = (ch.get("topic") or {}).get("value") or ""
            topic_str = f" — {topic}" if topic else ""
            lines.append(f"{marker}{ch['name']} (ID: {ch['id']}, {members} members){topic_str}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Slack list channels error: {exc}"


def slack_read_messages(channel: str, limit: int = 20) -> str:
    """Read recent messages from a Slack channel, by name or ID."""
    err = _check_token()
    if err:
        return err
    channel_id, resolve_err = _resolve_channel(channel)
    if resolve_err:
        return resolve_err
    try:
        data = _get("conversations.history", {"channel": channel_id, "limit": min(limit, 100)})
        messages = data.get("messages", [])
        if not messages:
            return f"No messages found in {channel}."
        display = channel.lstrip("#")
        lines = [f"Recent messages in #{display}:"]
        for msg in reversed(messages):  # oldest first
            user_id = msg.get("user", "")
            name = _get_user_name(user_id) if user_id else "bot"
            text = msg.get("text", "(no text)")
            ts = _ts_to_str(msg.get("ts", ""))
            lines.append(f"  [{ts}] {name}: {text}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Slack read messages error: {exc}"


def slack_send_message(channel: str, text: str) -> str:
    """Send a message to a Slack channel or DM. Always confirm with the user first."""
    err = _check_token()
    if err:
        return err
    channel = channel.strip()
    # If it looks like a user ID (U...), open a DM channel first
    if channel and channel[0].upper() == "U" and len(channel) >= 9:
        try:
            dm = _post("conversations.open", {"users": channel})
            channel_id = dm["channel"]["id"]
        except Exception as exc:
            return f"Could not open DM with user {channel}: {exc}"
    else:
        channel_id, resolve_err = _resolve_channel(channel)
        if resolve_err:
            return resolve_err
    try:
        result = _post("chat.postMessage", {"channel": channel_id, "text": text})
        ts = _ts_to_str(result.get("ts", ""))
        return f"Message sent at {ts}."
    except Exception as exc:
        return f"Slack send message error: {exc}"
