"""
iMessage connector — read texts and send messages via Mac's Messages app.

Read:  queries ~/Library/Messages/chat.db (SQLite).
       Requires Full Disk Access for the process running this server.
       System Settings → Privacy & Security → Full Disk Access → add Terminal
       (or your IDE / the uvicorn process).

Send:  osascript tells Messages.app to send — no extra permissions needed
       beyond Automation access (macOS will prompt once on first use).
"""

import datetime
import os
import platform
import shlex
import shutil
import sqlite3
import subprocess
import tempfile

_CHAT_DB = os.path.expanduser("~/Library/Messages/chat.db")

# Seconds between Apple epoch (2001-01-01) and Unix epoch (1970-01-01)
_APPLE_EPOCH_OFFSET = 978307200


def _require_macos() -> str | None:
    if platform.system() != "Darwin":
        return "iMessage tools only work on macOS."
    return None


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_messages(contact: str = "", limit: int = 10, days: int = 3) -> str:
    """
    Return recent iMessages from your Mac's Messages history.

    - contact: phone number (+1…) or Apple ID email to filter to one thread.
               Leave blank to see the most recent messages across all chats.
    - limit:   maximum number of messages to return (default 10).
    - days:    how many days back to look (default 3).

    Requires Full Disk Access for the terminal / server process.
    """
    err = _require_macos()
    if err:
        return err

    if not os.path.exists(_CHAT_DB):
        return "Messages database not found at ~/Library/Messages/chat.db."

    # Copy to a temp file to avoid SQLite "database is locked" on the live DB.
    tmp = tempfile.mktemp(suffix=".db")
    try:
        shutil.copy2(_CHAT_DB, tmp)
        return _query_messages(tmp, contact.strip(), limit, days)
    except PermissionError:
        return (
            "Permission denied reading chat.db. "
            "Grant Full Disk Access to Terminal (or your server process) in "
            "System Settings → Privacy & Security → Full Disk Access."
        )
    except Exception as e:
        return f"Error reading messages: {e}"
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _query_messages(db_path: str, contact: str, limit: int, days: int) -> str:
    cutoff_apple = (
        datetime.datetime.now(datetime.timezone.utc).timestamp()
        - _APPLE_EPOCH_OFFSET
        - days * 86400
    ) * 1e9  # nanoseconds

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            m.text,
            m.is_from_me,
            m.date          AS apple_ts,
            h.id            AS contact_id
        FROM  message m
        LEFT JOIN handle h ON h.ROWID = m.handle_id
        WHERE m.text IS NOT NULL
          AND m.text != ''
          AND m.date > ?
          {contact_filter}
        ORDER BY m.date DESC
        LIMIT ?
    """

    if contact:
        contact_filter = "AND h.id LIKE ?"
        params = (cutoff_apple, f"%{contact}%", limit)
    else:
        contact_filter = ""
        params = (cutoff_apple, limit)

    rows = conn.execute(
        query.format(contact_filter=contact_filter), params
    ).fetchall()
    conn.close()

    if not rows:
        label = f" with {contact}" if contact else ""
        return f"No messages found{label} in the last {days} day(s)."

    lines = []
    for row in reversed(rows):  # chronological order
        ts = datetime.datetime.fromtimestamp(
            row["apple_ts"] / 1e9 + _APPLE_EPOCH_OFFSET,
            tz=datetime.timezone.utc,
        ).astimezone().strftime("%a %b %-d %H:%M")

        sender = "Me" if row["is_from_me"] else (row["contact_id"] or "Unknown")
        lines.append(f"[{ts}] {sender}: {row['text']}")

    header = f"Messages (last {days}d" + (f", {contact}" if contact else "") + "):"
    return header + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

def send_message(to: str, message: str) -> str:
    """
    Send an iMessage to a phone number (+1…) or Apple ID email.

    Only call this after the user has confirmed the recipient and the full
    message text. Messages.app will prompt for Automation permission on first use.
    """
    err = _require_macos()
    if err:
        return err

    # Sanitise to prevent osascript injection
    safe_to  = to.strip().replace('"', "").replace("\\", "")
    safe_msg = message.replace("\\", "\\\\").replace('"', '\\"')

    if not safe_to:
        return "Recipient cannot be empty."

    script = (
        'tell application "Messages"\n'
        '  set targetService to 1st service whose service type = iMessage\n'
        f'  set targetBuddy to buddy "{safe_to}" of targetService\n'
        f'  send "{safe_msg}" to targetBuddy\n'
        'end tell'
    )

    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=20,
    )

    if result.returncode != 0:
        err_msg = result.stderr.strip()
        # Common failure: contact not found in iMessage
        if "Invalid parameter" in err_msg or "buddy" in err_msg.lower():
            return (
                f"Could not find '{to}' as an iMessage contact. "
                "Make sure the number/email is registered with iMessage and "
                "you have an existing conversation in Messages.app."
            )
        return f"Failed to send iMessage: {err_msg}"

    return f"iMessage sent to {to}."
