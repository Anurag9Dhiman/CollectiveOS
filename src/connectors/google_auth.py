"""
Shared Google OAuth handler.

All Google connectors (Calendar, Gmail, Drive, ...) import get_credentials()
from here so one token.json covers every scope. When you add a new connector,
add its scope to SCOPES and delete token.json so the user re-authorises once.
"""

import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    # calendar.events covers read + create/edit events (not calendar settings).
    # Upgraded from calendar.readonly to support create_event.
    # Delete token.json and re-authorise once after this change.
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")
CREDENTIALS_FILE = os.path.join(_ROOT, "credentials.json")
TOKEN_FILE = os.path.join(_ROOT, "token.json")


def get_credentials() -> Credentials:
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

    return creds
