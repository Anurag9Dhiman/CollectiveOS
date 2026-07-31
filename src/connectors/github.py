"""
GitHub connector — list repos, PRs, issues, CI status; create issues.

Env vars:
  GITHUB_TOKEN — Personal access token (classic or fine-grained)
                 Classic scopes needed: repo (private repos) or public_repo (public only)
                 Create at: https://github.com/settings/tokens
"""

import os

import requests

_BASE = "https://api.github.com"


def _headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN", "")
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _check_token() -> str | None:
    if not os.environ.get("GITHUB_TOKEN"):
        return "GITHUB_TOKEN is not set. Add it to .env and restart."
    return None


def _get(path: str, params: dict | None = None):
    resp = requests.get(f"{_BASE}{path}", headers=_headers(), params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict) -> dict:
    resp = requests.post(f"{_BASE}{path}", headers=_headers(), json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def github_list_repos(limit: int = 20) -> str:
    """List the authenticated user's repos, most recently updated first."""
    err = _check_token()
    if err:
        return err
    try:
        repos = _get("/user/repos", {
            "sort": "updated", "per_page": min(limit, 100),
            "affiliation": "owner,collaborator",
        })
        if not repos:
            return "No repositories found."
        lines = []
        for r in list(repos)[:limit]:
            visibility = "private" if r.get("private") else "public"
            desc = r.get("description") or ""
            lang = r.get("language") or ""
            parts = [r["full_name"]]
            if lang:
                parts.append(f"[{lang}]")
            parts.append(f"({visibility})")
            if desc:
                parts.append(f"— {desc}")
            lines.append(" ".join(parts))
        return "\n".join(lines)
    except Exception as exc:
        return f"GitHub list repos error: {exc}"


def github_list_prs(repo: str, state: str = "open") -> str:
    """List pull requests for owner/repo, sorted by most recently updated."""
    err = _check_token()
    if err:
        return err
    repo = repo.strip()
    try:
        prs = _get(f"/repos/{repo}/pulls", {"state": state, "per_page": 20, "sort": "updated"})
        if not prs:
            return f"No {state} pull requests in {repo}."
        lines = [f"Pull requests ({state}) — {repo}:"]
        for pr in prs:
            labels = ", ".join(l["name"] for l in pr.get("labels", []))
            label_str = f" [{labels}]" if labels else ""
            lines.append(
                f"  #{pr['number']} {pr['title']}{label_str}\n"
                f"    @{pr['user']['login']} · {pr['head']['ref']} → {pr['base']['ref']}\n"
                f"    {pr['html_url']}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"GitHub list PRs error: {exc}"


def github_list_issues(repo: str, state: str = "open", labels: str = "") -> str:
    """List issues for owner/repo. Excludes pull requests."""
    err = _check_token()
    if err:
        return err
    repo = repo.strip()
    params: dict = {"state": state, "per_page": 20, "sort": "updated"}
    if labels:
        params["labels"] = labels
    try:
        items = _get(f"/repos/{repo}/issues", params)
        # GitHub issues API returns PRs too — filter them out
        issues = [i for i in items if "pull_request" not in i]
        if not issues:
            return f"No {state} issues in {repo}."
        lines = [f"Issues ({state}) — {repo}:"]
        for i in issues:
            label_names = ", ".join(l["name"] for l in i.get("labels", []))
            label_str = f" [{label_names}]" if label_names else ""
            assignee = i.get("assignee")
            assignee_str = f" · @{assignee['login']}" if assignee else ""
            lines.append(
                f"  #{i['number']} {i['title']}{label_str}{assignee_str}\n"
                f"    {i['html_url']}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"GitHub list issues error: {exc}"


def github_get_ci_status(repo: str, ref: str) -> str:
    """Get CI check-run results for a branch name, tag, or commit SHA."""
    err = _check_token()
    if err:
        return err
    repo = repo.strip()
    ref = ref.strip()
    try:
        data = _get(f"/repos/{repo}/commits/{ref}/check-runs", {"per_page": 30})
        runs = data.get("check_runs", []) if isinstance(data, dict) else []
        if runs:
            completed = [r for r in runs if r.get("status") == "completed"]
            in_progress = [r for r in runs if r.get("status") != "completed"]
            failures = [r for r in completed if r.get("conclusion") not in ("success", "skipped", "neutral")]
            if in_progress:
                overall = "IN PROGRESS"
            elif failures:
                overall = "FAILING"
            else:
                overall = "PASSING"
            lines = [f"CI checks for {repo}@{ref}: {overall}"]
            for r in runs:
                conclusion = r.get("conclusion") or r.get("status", "pending")
                lines.append(f"  {conclusion.upper():12} {r['name']}")
            return "\n".join(lines)

        # Fall back to legacy Statuses API
        status_data = _get(f"/repos/{repo}/commits/{ref}/status")
        combined = status_data.get("state", "unknown")
        statuses = status_data.get("statuses", [])
        if not statuses:
            return f"No CI status found for {repo}@{ref}."
        lines = [f"CI status for {repo}@{ref}: {combined.upper()}"]
        for s in statuses:
            lines.append(f"  {s['state'].upper():12} {s['context']}: {s.get('description', '')}")
        return "\n".join(lines)
    except Exception as exc:
        return f"GitHub CI status error: {exc}"


def github_create_issue(repo: str, title: str, body: str = "", labels: str = "") -> str:
    """Create a new issue in owner/repo. Always confirm with the user before calling."""
    err = _check_token()
    if err:
        return err
    repo = repo.strip()
    payload: dict = {"title": title.strip()}
    if body:
        payload["body"] = body
    if labels:
        payload["labels"] = [l.strip() for l in labels.split(",") if l.strip()]
    try:
        issue = _post(f"/repos/{repo}/issues", payload)
        return f"Created issue #{issue['number']}: {issue['title']}\n{issue['html_url']}"
    except Exception as exc:
        return f"GitHub create issue error: {exc}"
