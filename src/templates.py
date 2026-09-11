"""
templates.py
------------
Pre-built automation workflow templates.

Each template is a dict that can be installed as a routine with one click.
Installing calls routines.create() with the template's fields.

Fields:
  id          — stable slug, never changes
  name        — display name
  description — what it does (1–2 sentences)
  category    — Morning | Work | Evening | Health | Productivity | Home
  icon        — emoji
  nav_task    — computer-navigation instruction (may be None)
  prompt      — agent chat prompt (may be None)
  schedule    — default cron (user can override)
  notify_via  — default channel
  tags        — list of keyword strings
"""

from __future__ import annotations

_TEMPLATES: list[dict] = [
    # ── Morning ──────────────────────────────────────────────────────────────
    {
        "id":          "morning-briefing",
        "name":        "Morning Briefing",
        "description": "Check today's calendar, summarise overnight emails, and report the weather. Delivered to Slack at 8 am.",
        "category":    "Morning",
        "icon":        "☀️",
        "nav_task":    None,
        "prompt":      "Give me a morning briefing: list today's calendar events, summarise any important emails received overnight, and check the weather for today.",
        "schedule":    "0 8 * * 1-5",
        "notify_via":  "slack",
        "tags":        ["morning", "calendar", "email", "weather"],
    },
    {
        "id":          "morning-news",
        "name":        "Morning News Digest",
        "description": "Search for top headlines in tech, business, and world news and send a Slack summary every weekday morning.",
        "category":    "Morning",
        "icon":        "📰",
        "nav_task":    None,
        "prompt":      "Search the web for today's top headlines in technology, business, and world news. Summarise the 5 most important stories in bullet points.",
        "schedule":    "0 7 * * 1-5",
        "notify_via":  "slack",
        "tags":        ["news", "morning", "search"],
    },
    {
        "id":          "morning-health",
        "name":        "Daily Health Check",
        "description": "Report last night's sleep quality, HRV, and today's readiness score every morning.",
        "category":    "Morning",
        "icon":        "💤",
        "nav_task":    None,
        "prompt":      "Check my health data: report last night's sleep duration, deep sleep percentage, HRV, and today's readiness score. Flag anything that looks off.",
        "schedule":    "0 8 * * *",
        "notify_via":  "slack",
        "tags":        ["health", "sleep", "hrv", "morning"],
    },

    # ── Work ─────────────────────────────────────────────────────────────────
    {
        "id":          "github-pr-summary",
        "name":        "GitHub PR Summary",
        "description": "Open GitHub, check open pull requests, and send a summary of what needs review.",
        "category":    "Work",
        "icon":        "🐙",
        "nav_task":    "Open Safari, go to github.com, navigate to the Pull Requests section, list all open PRs that need review, and summarise their titles and status.",
        "prompt":      "Summarise the GitHub PRs that need attention based on the computer task result above.",
        "schedule":    "0 9 * * 1-5",
        "notify_via":  "slack",
        "tags":        ["github", "prs", "code", "work"],
    },
    {
        "id":          "email-triage",
        "name":        "Email Triage",
        "description": "Open Mail, archive newsletters and promotions, flag anything that needs a reply, and summarise what's left.",
        "category":    "Work",
        "icon":        "📧",
        "nav_task":    "Open the Mail app, archive all emails that are newsletters or promotional, star any email that needs a reply, then summarise how many emails remain and what the most important ones are.",
        "prompt":      "Based on the email triage result above, list the emails that still need attention and suggest brief replies where appropriate.",
        "schedule":    "0 9 * * 1-5",
        "notify_via":  "slack",
        "tags":        ["email", "mail", "work", "morning"],
    },
    {
        "id":          "slack-digest",
        "name":        "Slack Digest",
        "description": "Open Slack, check unread messages and @mentions across all channels, and send a priority summary.",
        "category":    "Work",
        "icon":        "💬",
        "nav_task":    "Open Slack, check all unread messages and @mentions, note any action items or important decisions, then summarise the most important threads.",
        "prompt":      "Based on the Slack digest above, list any action items I need to handle today.",
        "schedule":    "0 9 * * 1-5",
        "notify_via":  "slack",
        "tags":        ["slack", "messages", "work"],
    },
    {
        "id":          "focus-mode",
        "name":        "Focus Mode",
        "description": "Close distracting apps (Slack, Mail, Safari social tabs), open your code editor, and start a Pomodoro timer.",
        "category":    "Productivity",
        "icon":        "🎯",
        "nav_task":    "Close Slack if open. Close Mail if open. Open VS Code or Cursor. Set the system volume to 30%. Open a new terminal window.",
        "prompt":      None,
        "schedule":    "0 10 * * 1-5",
        "notify_via":  "notification",
        "tags":        ["focus", "productivity", "work"],
    },
    {
        "id":          "calendar-review",
        "name":        "Weekly Calendar Review",
        "description": "Every Monday morning, open Calendar, review the week ahead, and send a structured summary to Slack.",
        "category":    "Productivity",
        "icon":        "📅",
        "nav_task":    "Open the Calendar app, switch to week view, and read all events scheduled for this week including times, titles, and any notes.",
        "prompt":      "Based on my calendar for this week, summarise the key meetings, deadlines, and free blocks I have. Suggest the best time for deep work.",
        "schedule":    "0 8 * * 1",
        "notify_via":  "slack",
        "tags":        ["calendar", "planning", "weekly", "monday"],
    },

    # ── Evening ───────────────────────────────────────────────────────────────
    {
        "id":          "end-of-day",
        "name":        "End-of-Day Shutdown",
        "description": "Close all work apps, save any open documents, clear the desktop, and send a done-for-the-day Slack message.",
        "category":    "Evening",
        "icon":        "🌙",
        "nav_task":    "Save any unsaved documents. Close VS Code, Cursor, Slack, Mail, and all Safari windows. Move any files on the Desktop to the Downloads folder.",
        "prompt":      "Send a brief end-of-day message noting that I've wrapped up work for the day.",
        "schedule":    "0 18 * * 1-5",
        "notify_via":  "slack",
        "tags":        ["evening", "shutdown", "productivity"],
    },
    {
        "id":          "activity-summary",
        "name":        "Daily Activity Summary",
        "description": "Report today's step count, active calories, and activity score every evening.",
        "category":    "Evening",
        "icon":        "🏃",
        "nav_task":    None,
        "prompt":      "Report today's activity data: steps, active calories, total calories, and activity score. Compare to my weekly average and flag if I should move more.",
        "schedule":    "0 20 * * *",
        "notify_via":  "slack",
        "tags":        ["health", "activity", "steps", "evening"],
    },

    # ── Home ──────────────────────────────────────────────────────────────────
    {
        "id":          "home-evening-check",
        "name":        "Evening Home Check",
        "description": "Check all smart-home devices at 10 pm and turn off any lights or switches left on.",
        "category":    "Home",
        "icon":        "🏠",
        "nav_task":    None,
        "prompt":      "Check all smart-home devices. List any lights or switches that are still on. Turn off any lights that shouldn't be on at this hour.",
        "schedule":    "0 22 * * *",
        "notify_via":  "notification",
        "tags":        ["home", "lights", "smart-home", "evening"],
    },
    {
        "id":          "weekly-system-cleanup",
        "name":        "Weekly System Cleanup",
        "description": "Every Sunday, empty the Trash, clear Downloads older than 30 days, and report disk usage.",
        "category":    "Productivity",
        "icon":        "🧹",
        "nav_task":    "Open Finder. Empty the Trash. In the Downloads folder, move any file older than 30 days to a folder called 'Old Downloads'. Report how much disk space was freed.",
        "prompt":      "Based on the cleanup result, report current disk usage and suggest anything else I should clean up.",
        "schedule":    "0 10 * * 0",
        "notify_via":  "slack",
        "tags":        ["cleanup", "disk", "weekly", "finder"],
    },
]

_INDEX = {t["id"]: t for t in _TEMPLATES}


def list_templates(category: str | None = None) -> list[dict]:
    if category:
        return [t for t in _TEMPLATES if t["category"].lower() == category.lower()]
    return list(_TEMPLATES)


def get_template(template_id: str) -> dict | None:
    return _INDEX.get(template_id)


def categories() -> list[str]:
    seen: list[str] = []
    for t in _TEMPLATES:
        if t["category"] not in seen:
            seen.append(t["category"])
    return seen


def install(template_id: str, schedule_override: str | None = None,
            notify_override: str | None = None) -> dict:
    """
    Install a template as a saved routine.
    Returns the created routine dict {id, name, schedule, ...}.
    Raises KeyError if template_id is unknown.
    """
    tmpl = _INDEX.get(template_id)
    if not tmpl:
        raise KeyError(f"Unknown template: {template_id!r}")

    from src import routines as _r, scheduler as _s

    routine_id = _r.create(
        name=tmpl["name"],
        prompt=tmpl["prompt"] or "",
        schedule=schedule_override or tmpl["schedule"],
        notify_via=notify_override or tmpl["notify_via"],
        nav_task=tmpl["nav_task"],
    )
    _s.reload_routine(routine_id)

    return {
        "id":       routine_id,
        "name":     tmpl["name"],
        "schedule": schedule_override or tmpl["schedule"],
        "template": template_id,
    }
