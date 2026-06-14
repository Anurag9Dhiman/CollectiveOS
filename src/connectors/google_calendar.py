"""
Google Calendar connector — read-only.

Uses the shared OAuth handler in google_auth.py. On first run a browser
opens for consent; subsequent runs reuse token.json silently.
"""

import datetime

from googleapiclient.discovery import build

from src.connectors.google_auth import get_credentials


def get_calendar_events(days_ahead: int = 7) -> str:
    """Return upcoming calendar events as a plain-text list."""
    service = build("calendar", "v3", credentials=get_credentials())

    now = datetime.datetime.utcnow()
    time_min = now.isoformat() + "Z"
    time_max = (now + datetime.timedelta(days=days_ahead)).isoformat() + "Z"

    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = result.get("items", [])
    if not events:
        return f"No events in the next {days_ahead} day(s)."

    lines = []
    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date", ""))
        title = event.get("summary", "(no title)")
        location = event.get("location", "")
        line = f"- {start}  {title}"
        if location:
            line += f"  [{location}]"
        lines.append(line)

    return "\n".join(lines)
