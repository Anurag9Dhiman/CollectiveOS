"""
Todoist connector — read tasks and add new ones.

Auth: personal API token (no OAuth).
  1. Go to https://app.todoist.com/app/settings/integrations/developer
  2. Copy your API token.
  3. Add to .env:  TODOIST_API_TOKEN=your-token-here

Requires: pip install requests
"""

import os
import requests

_BASE = "https://api.todoist.com/rest/v2"


def _headers() -> dict:
    token = os.environ.get("TODOIST_API_TOKEN", "")
    if not token:
        raise RuntimeError(
            "TODOIST_API_TOKEN not set. Add it to your .env file."
        )
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_tasks(project_name: str = "", filter_str: str = "") -> str:
    """
    List active Todoist tasks.

    - project_name: filter to a specific project (case-insensitive partial match).
    - filter_str:   Todoist filter expression, e.g. 'today', 'overdue',
                    'p1' (priority 1), '7 days' (due in next 7 days).
    """
    params: dict = {}
    if filter_str:
        params["filter"] = filter_str

    resp = requests.get(f"{_BASE}/tasks", headers=_headers(), params=params)
    resp.raise_for_status()
    tasks = resp.json()

    if project_name:
        # Resolve project name → id
        proj_resp = requests.get(f"{_BASE}/projects", headers=_headers())
        proj_resp.raise_for_status()
        matched = [
            p["id"]
            for p in proj_resp.json()
            if project_name.lower() in p["name"].lower()
        ]
        tasks = [t for t in tasks if t.get("project_id") in matched]

    if not tasks:
        return "No tasks found."

    lines = []
    for t in tasks:
        due = t.get("due", {})
        due_str = f"  due {due['date']}" if due else ""
        priority = t.get("priority", 1)
        pstr = f"  [p{priority}]" if priority > 1 else ""
        lines.append(f"- [{t['id']}] {t['content']}{due_str}{pstr}")

    return "\n".join(lines)


def get_projects() -> str:
    """List all Todoist projects."""
    resp = requests.get(f"{_BASE}/projects", headers=_headers())
    resp.raise_for_status()
    projects = resp.json()

    if not projects:
        return "No projects found."

    return "\n".join(f"- [{p['id']}] {p['name']}" for p in projects)


# ---------------------------------------------------------------------------
# Write (requires explicit user confirmation upstream before calling)
# ---------------------------------------------------------------------------

def add_task(content: str, due_string: str = "", project_name: str = "") -> str:
    """
    Add a new task to Todoist.

    - content:      Task title, e.g. 'Buy groceries'.
    - due_string:   Natural-language due date, e.g. 'tomorrow', 'next Monday'.
    - project_name: Destination project (uses Inbox if omitted).
    """
    payload: dict = {"content": content}
    if due_string:
        payload["due_string"] = due_string

    if project_name:
        proj_resp = requests.get(f"{_BASE}/projects", headers=_headers())
        proj_resp.raise_for_status()
        matched = [
            p for p in proj_resp.json()
            if project_name.lower() in p["name"].lower()
        ]
        if matched:
            payload["project_id"] = matched[0]["id"]

    resp = requests.post(f"{_BASE}/tasks", headers=_headers(), json=payload)
    resp.raise_for_status()
    task = resp.json()

    due = task.get("due", {})
    due_str = f", due {due['date']}" if due else ""
    return f"Task added: '{task['content']}' (id {task['id']}{due_str})"
