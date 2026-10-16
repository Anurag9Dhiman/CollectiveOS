"""
workflow_recorder.py
--------------------
Workflow recorder — converts a plain-English workflow description into a
scheduled nav routine.

The user says something like:
  "Every morning, open Mail, archive newsletters, reply to anything flagged urgent"

This module:
  1. Uses the LLM to break the description into a concise nav_task string
     (what the nav agent should do autonomously).
  2. Infers a cron schedule from any temporal hint in the description
     ("every morning" → "0 8 * * *", "every hour" → "0 * * * *", etc.).
     If no hint is found, defaults to manual-only (no schedule).
  3. Saves the result as a routine via src.routines.create().
  4. Returns a summary dict the API endpoint can return to the client.

Public API:
  record(description, name=None, schedule=None, notify_via="slack") -> dict
"""

from __future__ import annotations

import json
import logging
import os
import re

log = logging.getLogger(__name__)

_PLAN_PROMPT = """You are a workflow planner for an autonomous computer agent.

The user has described a workflow they want to automate. Your job is to:
1. Write a precise nav_task — a single instruction string the computer agent will
   execute to complete the workflow. Be specific: name the apps, the actions, the
   order. The agent navigates macOS like a human (clicks, types, scrolls).
2. Infer the best cron schedule from temporal hints in the description.
   Use standard 5-field cron syntax (minute hour day month weekday).
   Common mappings:
     "every morning" / "every day at 8am" → "0 8 * * *"
     "every evening" / "every night"      → "0 20 * * *"
     "every hour"                         → "0 * * * *"
     "every weekday morning"              → "0 8 * * 1-5"
     "every Monday"                       → "0 9 * * 1"
     "every 30 minutes"                   → "*/30 * * * *"
     No clear schedule mentioned          → null
3. Suggest a short name (3–6 words) for the routine if none is given.

Reply ONLY with this JSON (no markdown, no extra text):
{
  "name": "Short Routine Name",
  "nav_task": "Precise instruction for the nav agent...",
  "schedule": "cron expression or null",
  "schedule_human": "human-readable schedule or 'manual'"
}
"""


def _call_llm(description: str) -> dict:
    from google import genai
    from google.genai import types

    model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[types.Part(text=f"Workflow description:\n{description}")],
        config=types.GenerateContentConfig(
            system_instruction=_PLAN_PROMPT,
            temperature=0.2,
            max_output_tokens=512,
        ),
    )
    text = (response.text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def record(
    description: str,
    name: str | None = None,
    schedule: str | None = None,
    notify_via: str = "slack",
) -> dict:
    """
    Convert a workflow description into a saved routine.

    Parameters
    ----------
    description : str
        Natural-language workflow description from the user.
    name : str | None
        Override routine name; if None the LLM suggests one.
    schedule : str | None
        Override cron schedule; if None the LLM infers one from description.
    notify_via : str
        Notification channel when the routine fires (default: slack).

    Returns
    -------
    dict with keys: id, name, nav_task, schedule, schedule_human, notify_via
    """
    plan = _call_llm(description)

    routine_name = name or plan.get("name", "Recorded Workflow")
    nav_task     = plan.get("nav_task", description)
    cron         = schedule or plan.get("schedule")
    human_sched  = plan.get("schedule_human", "manual")

    if not cron:
        # No schedule inferred — save as a manual-only routine with a placeholder
        # cron that never fires automatically (31st of February = never).
        cron = "0 0 31 2 *"
        human_sched = "manual (no schedule detected)"

    from src import routines as _routines
    routine_id = _routines.create(
        name=routine_name,
        prompt="",           # nav-only routine; no chat prompt needed
        schedule=cron,
        notify_via=notify_via,
        nav_task=nav_task,
    )

    log.info(
        "workflow_recorder: created routine %d (%s) schedule=%s",
        routine_id, routine_name, cron,
    )

    from src import scheduler as _sched
    _sched.reload_routine(routine_id)

    return {
        "id":             routine_id,
        "name":           routine_name,
        "nav_task":       nav_task,
        "schedule":       cron,
        "schedule_human": human_sched,
        "notify_via":     notify_via,
    }
