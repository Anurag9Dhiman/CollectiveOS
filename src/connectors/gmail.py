"""
Gmail connector — read-only.

Uses the shared OAuth handler in google_auth.py (same token.json as Calendar).
"""

import base64
import re

from googleapiclient.discovery import build

from src.connectors.google_auth import get_credentials


def _service():
    return build("gmail", "v1", credentials=get_credentials())


def _decode_body(payload: dict) -> str:
    """Extract plain-text body from a message payload."""
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

    for part in payload.get("parts", []):
        text = _decode_body(part)
        if text:
            return text

    return ""


def _header(headers: list, name: str) -> str:
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def _summarise_message(msg: dict) -> str:
    headers = msg["payload"]["headers"]
    subject = _header(headers, "Subject") or "(no subject)"
    sender  = _header(headers, "From")
    date    = _header(headers, "Date")
    snippet = msg.get("snippet", "")
    # Trim snippet to keep output compact
    snippet = re.sub(r"\s+", " ", snippet).strip()[:200]
    return f"From: {sender}\nDate: {date}\nSubject: {subject}\nSnippet: {snippet}"


def get_recent_emails(max_results: int = 10) -> str:
    """Return the most recent emails from the inbox."""
    svc = _service()
    resp = svc.users().messages().list(
        userId="me", labelIds=["INBOX"], maxResults=max_results
    ).execute()

    messages = resp.get("messages", [])
    if not messages:
        return "No emails found in your inbox."

    summaries = []
    for m in messages:
        full = svc.users().messages().get(
            userId="me", id=m["id"], format="full"
        ).execute()
        summaries.append(_summarise_message(full))

    return "\n\n---\n\n".join(summaries)


def search_emails(query: str, max_results: int = 5) -> str:
    """Search emails using a Gmail query string (e.g. 'from:alice subject:invoice')."""
    svc = _service()
    resp = svc.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()

    messages = resp.get("messages", [])
    if not messages:
        return f"No emails found for query: {query!r}"

    summaries = []
    for m in messages:
        full = svc.users().messages().get(
            userId="me", id=m["id"], format="full"
        ).execute()
        summaries.append(_summarise_message(full))

    return "\n\n---\n\n".join(summaries)
