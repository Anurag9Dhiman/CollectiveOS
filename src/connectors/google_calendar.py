"""
Google Calendar connector — read and create events.

Uses the shared OAuth handler in google_auth.py. On first run a browser
opens for consent; subsequent runs reuse token.json silently.

Note: scope was upgraded from calendar.readonly to calendar.events.
If you have an existing token.json, delete it and re-run to re-authorise.
"""

import datetime
import os

from googleapiclient.discovery import build

from src.connectors.google_auth import get_credentials

_DEFAULT_TZ = os.environ.get("TIMEZONE", "UTC")


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


def create_event(
    title: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
    timezone: str = "",
) -> str:
    """
    Create a Google Calendar event and return a confirmation with the event link.

    - title:          event title / summary
    - start_datetime: ISO 8601 string, e.g. "2024-07-10T14:00:00"
    - end_datetime:   ISO 8601 string, e.g. "2024-07-10T15:00:00"
    - description:    optional event body text
    - timezone:       IANA tz name (e.g. "America/New_York"). Defaults to the
                      TIMEZONE env var, or UTC if unset.
    """
    tz = timezone or _DEFAULT_TZ
    service = build("calendar", "v3", credentials=get_credentials())

    body: dict = {
        "summary": title,
        "start": {"dateTime": start_datetime, "timeZone": tz},
        "end":   {"dateTime": end_datetime,   "timeZone": tz},
    }
    if description:
        body["description"] = description

    event = service.events().insert(calendarId="primary", body=body).execute()

    link = event.get("htmlLink", "")
    event_id = event.get("id", "")
    return (
        f"Event created: '{title}'\n"
        f"  Start: {start_datetime}  ({tz})\n"
        f"  End:   {end_datetime}\n"
        f"  ID:    {event_id}\n"
        f"  Link:  {link}"
    )
