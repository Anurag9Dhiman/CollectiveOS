"""
Google Calendar connector — read-only.

First run: opens a browser for OAuth consent and saves token.json next to
credentials.json. Every run after that reuses the saved token silently.

Prerequisites (one-time setup):
  1. Go to https://console.cloud.google.com/
  2. Create a project (or pick an existing one).
  3. Enable the Google Calendar API for that project.
  4. Go to APIs & Services > Credentials > Create Credentials > OAuth client ID.
     Application type: Desktop app.
  5. Download the JSON file and save it as credentials.json in the repo root.
  6. pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

import datetime
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

# Resolve paths relative to the repo root (two levels up from this file).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")
CREDENTIALS_FILE = os.path.join(_ROOT, "credentials.json")
TOKEN_FILE = os.path.join(_ROOT, "token.json")


def _get_service():
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def get_calendar_events(days_ahead: int = 7) -> str:
    """Return upcoming calendar events as a plain-text list."""
    service = _get_service()

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
