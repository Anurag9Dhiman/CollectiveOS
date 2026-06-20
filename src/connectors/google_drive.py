"""
Google Drive connector — read-only.

Uses the shared OAuth handler in google_auth.py (same token.json).
Delete token.json and re-run once after adding this connector so Google
grants the drive.readonly scope.
"""

import io

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from src.connectors.google_auth import get_credentials

# Plain-text MIME types Drive can export to.
_EXPORT_MAP = {
    "application/vnd.google-apps.document":     ("text/plain", ".txt"),
    "application/vnd.google-apps.spreadsheet":  ("text/csv",   ".csv"),
    "application/vnd.google-apps.presentation": ("text/plain", ".txt"),
}


def _service():
    return build("drive", "v3", credentials=get_credentials())


def list_files(max_results: int = 20, query: str = "") -> str:
    """
    List files in Google Drive.
    Optionally filter with a Drive query string, e.g.:
      "name contains 'budget'"
      "mimeType='application/vnd.google-apps.document'"
      "modifiedTime > '2024-01-01'"
    """
    q = "trashed = false"
    if query:
        q += f" and ({query})"

    resp = _service().files().list(
        q=q,
        pageSize=max_results,
        fields="files(id, name, mimeType, modifiedTime, size)",
        orderBy="modifiedTime desc",
    ).execute()

    files = resp.get("files", [])
    if not files:
        return "No files found."

    lines = []
    for f in files:
        size = f.get("size", "")
        size_str = f" ({int(size):,} bytes)" if size else ""
        lines.append(
            f"- [{f['id']}] {f['name']}{size_str}  (modified: {f.get('modifiedTime', '')[:10]})"
        )
    return "\n".join(lines)


def read_file(file_id: str) -> str:
    """
    Read the text content of a Drive file by its ID.
    Works for Google Docs, Sheets, Slides (exported as plain text/CSV)
    and plain text / markdown files stored in Drive.
    """
    svc = _service()
    meta = svc.files().get(fileId=file_id, fields="name, mimeType").execute()
    mime = meta.get("mimeType", "")
    name = meta.get("name", file_id)

    buf = io.BytesIO()

    if mime in _EXPORT_MAP:
        export_mime, _ = _EXPORT_MAP[mime]
        req = svc.files().export_media(fileId=file_id, mimeType=export_mime)
    else:
        req = svc.files().get_media(fileId=file_id)

    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()

    content = buf.getvalue().decode("utf-8", errors="ignore")

    # Cap at 4 000 chars to stay within prompt budget.
    if len(content) > 4000:
        content = content[:4000] + "\n\n[… truncated — file is longer]"

    return f"File: {name}\n\n{content}"
