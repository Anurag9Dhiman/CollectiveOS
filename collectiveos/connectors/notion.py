"""
Notion connector — search, read, create, and append to Notion pages.

Env vars:
  NOTION_API_KEY  — Integration token from https://www.notion.so/my-integrations
                    Grant the integration access to pages/databases you want to use.
"""

import os
from typing import Any

import requests

_BASE = "https://api.notion.com/v1"
_VERSION = "2022-06-28"


def _headers() -> dict[str, str]:
    token = os.environ.get("NOTION_API_KEY", "")
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": _VERSION,
        "Content-Type": "application/json",
    }


def _check_token() -> str | None:
    if not os.environ.get("NOTION_API_KEY"):
        return "NOTION_API_KEY is not set. Add it to .env and restart."
    return None


def _get(path: str, params: dict | None = None) -> dict:
    resp = requests.get(f"{_BASE}{path}", headers=_headers(), params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict) -> dict:
    resp = requests.post(f"{_BASE}{path}", headers=_headers(), json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _patch(path: str, body: dict) -> dict:
    resp = requests.patch(f"{_BASE}{path}", headers=_headers(), json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _rich_text_to_str(rich_texts: list[dict]) -> str:
    return "".join(rt.get("plain_text", "") for rt in rich_texts)


def _block_to_str(block: dict) -> str:
    btype = block.get("type", "")
    data = block.get(btype, {})
    text = _rich_text_to_str(data.get("rich_text", []))
    if btype == "divider":
        return "---"
    if btype == "child_page":
        return f"[Page: {data.get('title', '(untitled)')}]"
    if not text:
        return ""
    prefix = {
        "heading_1": "# ",
        "heading_2": "## ",
        "heading_3": "### ",
        "bulleted_list_item": "• ",
        "numbered_list_item": "1. ",
        "to_do": f"[{'x' if data.get('checked') else ' '}] ",
        "quote": "> ",
        "code": "```\n",
    }.get(btype, "")
    suffix = "\n```" if btype == "code" else ""
    return prefix + text + suffix


def _page_title(page: dict) -> str:
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            return _rich_text_to_str(prop.get("title", []))
    return "(untitled)"


def _db_title(db: dict) -> str:
    return _rich_text_to_str(db.get("title", []))


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def notion_search(query: str, filter_type: str = "") -> str:
    """Search Notion for pages and databases matching a query."""
    err = _check_token()
    if err:
        return err
    try:
        body: dict[str, Any] = {"query": query.strip(), "page_size": 10}
        if filter_type in ("page", "database"):
            body["filter"] = {"value": filter_type, "property": "object"}
        data = _post("/search", body)
        results = data.get("results", [])
        if not results:
            return f"No Notion results found for: {query!r}"
        lines = []
        for r in results:
            obj_type = r.get("object", "")
            title = _page_title(r) if obj_type == "page" else _db_title(r)
            page_id = r.get("id", "")
            url = r.get("url", "")
            lines.append(f"[{obj_type}] {title}\n  ID: {page_id}\n  URL: {url}")
        return "\n\n".join(lines)
    except Exception as exc:
        return f"Notion search error: {exc}"


def notion_read_page(page_id: str) -> str:
    """Read the full content of a Notion page, returned as plain text."""
    err = _check_token()
    if err:
        return err
    page_id = page_id.strip().replace("-", "")
    try:
        page = _get(f"/pages/{page_id}")
        title = _page_title(page)
        blocks_data = _get(f"/blocks/{page_id}/children", {"page_size": 100})
        lines = [f"# {title}", ""]
        for block in blocks_data.get("results", []):
            line = _block_to_str(block)
            if line:
                lines.append(line)
        return "\n".join(lines).strip()
    except Exception as exc:
        return f"Notion read error: {exc}"


def notion_create_page(parent_id: str, title: str, content: str = "") -> str:
    """Create a new Notion page under a parent page or database."""
    err = _check_token()
    if err:
        return err
    parent_id = parent_id.strip().replace("-", "")
    # Try to detect parent type; default to page_id if both fail.
    parent_type = "page_id"
    try:
        _get(f"/databases/{parent_id}")
        parent_type = "database_id"
    except Exception:
        pass
    children = []
    for paragraph in content.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": paragraph}}]
                },
            })
    body: dict[str, Any] = {
        "parent": {parent_type: parent_id},
        "properties": {
            "title": {"title": [{"type": "text", "text": {"content": title}}]}
        },
    }
    if children:
        body["children"] = children
    try:
        page = _post("/pages", body)
        url = page.get("url", "")
        return f"Created Notion page '{title}': {url}"
    except Exception as exc:
        return f"Notion create error: {exc}"


def notion_append_to_page(page_id: str, content: str) -> str:
    """Append text content to the end of an existing Notion page."""
    err = _check_token()
    if err:
        return err
    page_id = page_id.strip().replace("-", "")
    children = []
    for paragraph in content.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": paragraph}}]
                },
            })
    if not children:
        return "No content to append."
    try:
        _patch(f"/blocks/{page_id}/children", {"children": children})
        return f"Appended {len(children)} paragraph(s) to Notion page {page_id}."
    except Exception as exc:
        return f"Notion append error: {exc}"
