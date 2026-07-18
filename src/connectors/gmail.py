"""
Gmail connector — read inbox and compose drafts.

Uses the shared OAuth handler in google_auth.py (same token.json as Calendar).
Scope: gmail.readonly (read) + gmail.compose (create drafts / send).
Delete token.json and re-authorise once after the scope change.
"""

import base64
import email.mime.text
import email.mime.multipart
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


# ---------------------------------------------------------------------------
# Write (confirm with user before calling)
# ---------------------------------------------------------------------------

def _build_raw(to: str, subject: str, body: str, reply_to_id: str = "") -> tuple[str, str]:
    """
    Encode a MIME message to base64url. Returns (raw_b64, thread_id).
    thread_id is empty string when not a reply.
    """
    svc = _service()
    msg = email.mime.text.MIMEText(body, "plain")
    msg["To"]      = to
    msg["Subject"] = subject
    thread_id      = ""

    if reply_to_id:
        original = svc.users().messages().get(
            userId="me", id=reply_to_id, format="metadata",
            metadataHeaders=["Subject", "Message-ID", "References"]
        ).execute()
        hdrs = original.get("payload", {}).get("headers", [])

        def _h(name: str) -> str:
            return next((x["value"] for x in hdrs if x["name"].lower() == name.lower()), "")

        if not subject:
            orig_subj = _h("Subject")
            msg["Subject"] = f"Re: {orig_subj}" if not orig_subj.startswith("Re:") else orig_subj

        msg_id = _h("Message-ID")
        if msg_id:
            msg["In-Reply-To"] = msg_id
            refs = _h("References")
            msg["References"] = f"{refs} {msg_id}".strip()

        thread_id = original.get("threadId", "")

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    return raw, thread_id


def create_draft(
    to: str,
    subject: str,
    body: str,
    reply_to_id: str = "",
) -> str:
    """
    Create a Gmail draft. Does NOT send — the user reviews and sends from Gmail.

    - to:          Recipient email address.
    - subject:     Subject line. If reply_to_id is given and subject is blank,
                   'Re: <original subject>' is set automatically.
    - body:        Plain-text email body.
    - reply_to_id: Optional — message id to reply to (keeps the email in the
                   same thread). Use the id field from get_recent_emails.
    """
    svc = _service()
    raw, thread_id = _build_raw(to, subject, body, reply_to_id)

    msg_body: dict = {"raw": raw}
    if thread_id:
        msg_body["threadId"] = thread_id

    draft = svc.users().drafts().create(
        userId="me", body={"message": msg_body}
    ).execute()

    final_subject = subject or "(auto from thread)"
    return (
        f"Draft created — not sent. Open Gmail to review and send.\n"
        f"  To:       {to}\n"
        f"  Subject:  {final_subject}\n"
        f"  Draft id: {draft.get('id', '')}"
    )


def send_email(
    to: str,
    subject: str,
    body: str,
    reply_to_id: str = "",
) -> str:
    """
    Send an email immediately via Gmail.

    Only call this after the user has explicitly approved the full content —
    recipient, subject, and body. For safety, prefer create_draft first.

    - to:          Recipient email address.
    - subject:     Email subject.
    - body:        Plain-text body.
    - reply_to_id: Optional — message id to reply to (keeps thread).
    """
    svc = _service()
    raw, thread_id = _build_raw(to, subject, body, reply_to_id)

    send_body: dict = {"raw": raw}
    if thread_id:
        send_body["threadId"] = thread_id

    sent = svc.users().messages().send(userId="me", body=send_body).execute()
    return (
        f"Email sent.\n"
        f"  To:         {to}\n"
        f"  Subject:    {subject}\n"
        f"  Message id: {sent.get('id', '')}"
    )
