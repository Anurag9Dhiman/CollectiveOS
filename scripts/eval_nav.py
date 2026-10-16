#!/usr/bin/env python3
"""
OSWorld-style evaluation harness for CollectiveOS nav agent.

Runs a fixed suite of standardised desktop tasks and measures
VTCR (Verified Task Completion Rate = verified_done / total_attempted).

Usage:
    python scripts/eval_nav.py                        # run full suite
    python scripts/eval_nav.py --task "open Calculator"   # single task
    python scripts/eval_nav.py --dry-run              # list tasks, no execution
    python scripts/eval_nav.py --out results.json     # save report
    python scripts/eval_nav.py --tags app_launch      # filter by tag
    python scripts/eval_nav.py --horizon 1_3          # filter by horizon group
    python scripts/eval_nav.py --group A              # run RQ experiment group

Each task records:
  status        — "done" | "max_iter" | "error" | "stopped" | "timeout"
  verified      — bool (did the verifier confirm completion?)
  steps         — number of perceive-act iterations used
  duration      — wall-clock seconds
  model         — "UI-TARS" | "Gemini"
  horizon_group — "1_3" | "4_7" | "8_15" | "16_30" | "30plus"
  tokens_in     — input tokens consumed (if available)
  tokens_out    — output tokens consumed (if available)
  model_calls   — number of model invocations
  error_cat     — failure category (RQ21 taxonomy) if failed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure src/ is on the path regardless of where the script is run from.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Task suite
# ---------------------------------------------------------------------------
# Each task is a dict with:
#   task    — natural-language instruction passed to navigate_computer
#   context — optional extra context string
#   tags    — categories for filtering / reporting
#   desktop_only — True → skip browser routing even if keywords match
# ---------------------------------------------------------------------------

TASKS: list[dict] = [
    # ══════════════════════════════════════════════════════════════════════════
    # HORIZON 1-3  (single action or 2-3 clicks)
    # ══════════════════════════════════════════════════════════════════════════

    # ── App launch ────────────────────────────────────────────────────────────
    {
        "id": "launch_calculator",
        "task": "Open the Calculator app.",
        "tags": ["app_launch", "spotlight"],
        "horizon_group": "1_3",
        "desktop_only": True,
    },
    {
        "id": "launch_finder",
        "task": "Open a new Finder window.",
        "tags": ["app_launch", "finder"],
        "horizon_group": "1_3",
        "desktop_only": True,
    },
    {
        "id": "launch_notes",
        "task": "Open the Notes app.",
        "tags": ["app_launch", "spotlight"],
        "horizon_group": "1_3",
        "desktop_only": True,
    },
    {
        "id": "launch_terminal",
        "task": "Open the Terminal app.",
        "tags": ["app_launch", "spotlight"],
        "horizon_group": "1_3",
        "desktop_only": True,
    },
    {
        "id": "launch_safari",
        "task": "Open Safari.",
        "tags": ["app_launch", "browser"],
        "horizon_group": "1_3",
        "desktop_only": False,
    },
    {
        "id": "launch_activity_monitor",
        "task": "Open Activity Monitor.",
        "tags": ["app_launch", "spotlight"],
        "horizon_group": "1_3",
        "desktop_only": True,
    },
    # ── OS actions ────────────────────────────────────────────────────────────
    {
        "id": "goto_desktop",
        "task": "Show the desktop by hiding all open windows.",
        "tags": ["os_action", "show_desktop"],
        "horizon_group": "1_3",
        "desktop_only": True,
    },
    {
        "id": "read_clipboard",
        "task": "Read whatever text is currently on the clipboard and report it.",
        "tags": ["clipboard", "read"],
        "horizon_group": "1_3",
        "desktop_only": True,
    },
    {
        "id": "spotlight_search",
        "task": "Use Spotlight to search for 'Terminal' and report the top result.",
        "tags": ["spotlight", "search"],
        "horizon_group": "1_3",
        "desktop_only": True,
    },
    {
        "id": "spotlight_math",
        "task": "Use Spotlight to calculate 47 multiplied by 23 and report the result.",
        "tags": ["spotlight", "math"],
        "horizon_group": "1_3",
        "desktop_only": True,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # HORIZON 4-7  (a few coordinated steps)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "launch_textedit",
        "task": "Open TextEdit and create a new blank document.",
        "tags": ["app_launch", "new_doc"],
        "horizon_group": "4_7",
        "desktop_only": True,
    },
    {
        "id": "navigate_downloads",
        "task": "Open the Downloads folder in Finder.",
        "tags": ["finder", "navigation"],
        "horizon_group": "4_7",
        "desktop_only": True,
    },
    {
        "id": "navigate_documents",
        "task": "Open the Documents folder in Finder.",
        "tags": ["finder", "navigation"],
        "horizon_group": "4_7",
        "desktop_only": True,
    },
    {
        "id": "system_info",
        "task": "Open System Settings and navigate to the General page.",
        "tags": ["system", "settings"],
        "horizon_group": "4_7",
        "desktop_only": True,
    },
    {
        "id": "volume_check",
        "task": "Open System Settings, navigate to Sound, and report the current output volume level.",
        "tags": ["system", "settings", "read"],
        "horizon_group": "4_7",
        "desktop_only": True,
    },
    {
        "id": "check_wifi",
        "task": "Open System Settings, navigate to Wi-Fi, and report whether Wi-Fi is on or off.",
        "tags": ["system", "settings", "read"],
        "horizon_group": "4_7",
        "desktop_only": True,
    },
    {
        "id": "check_battery",
        "task": "Open System Settings, navigate to Battery, and report the current battery percentage.",
        "tags": ["system", "settings", "read"],
        "horizon_group": "4_7",
        "desktop_only": True,
    },
    {
        "id": "check_display",
        "task": "Open System Settings, navigate to Displays, and report the current resolution.",
        "tags": ["system", "settings", "read"],
        "horizon_group": "4_7",
        "desktop_only": True,
    },
    {
        "id": "finder_column_view",
        "task": "Open Finder and switch to Column view.",
        "tags": ["finder", "view"],
        "horizon_group": "4_7",
        "desktop_only": True,
    },
    {
        "id": "safari_new_tab",
        "task": "Open Safari and open a new tab.",
        "tags": ["browser", "tab"],
        "horizon_group": "4_7",
        "desktop_only": False,
    },
    {
        "id": "notes_new_note",
        "task": "Open Notes and create a new note.",
        "tags": ["app_launch", "notes"],
        "horizon_group": "4_7",
        "desktop_only": True,
    },
    {
        "id": "screenshot_desktop",
        "task": "Take a screenshot of the desktop using the keyboard shortcut and report the file path.",
        "tags": ["keyboard_shortcut", "screenshot"],
        "horizon_group": "4_7",
        "desktop_only": True,
    },
    {
        "id": "check_accessibility",
        "task": "Open System Settings, navigate to Accessibility, and report what options are visible.",
        "tags": ["system", "settings", "read"],
        "horizon_group": "4_7",
        "desktop_only": True,
    },
    {
        "id": "spotlight_launch_app",
        "task": "Use Spotlight to open the Calculator app.",
        "tags": ["spotlight", "app_launch"],
        "horizon_group": "4_7",
        "desktop_only": True,
    },
    {
        "id": "finder_get_info",
        "task": "Open Finder, navigate to the Downloads folder, and use Get Info to check the folder size.",
        "tags": ["finder", "navigation", "read"],
        "horizon_group": "4_7",
        "desktop_only": True,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # HORIZON 8-15  (multi-step workflows)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "textedit_type",
        "task": (
            "Open TextEdit, create a new plain-text document, "
            "type the text 'Hello CollectiveOS', and leave it open (do not save)."
        ),
        "tags": ["multi_step", "typing"],
        "horizon_group": "8_15",
        "desktop_only": True,
    },
    {
        "id": "textedit_bold",
        "task": (
            "Open TextEdit, create a new Rich Text document, type 'CollectiveOS Test', "
            "select all the text, make it bold, then change the font size to 18."
        ),
        "tags": ["multi_step", "typing", "formatting"],
        "horizon_group": "8_15",
        "desktop_only": True,
    },
    {
        "id": "notes_with_text",
        "task": (
            "Open Notes, create a new note titled 'Experiment Note', "
            "add the body text 'This note was created by CollectiveOS during RQ1 evaluation.'"
        ),
        "tags": ["multi_step", "notes", "typing"],
        "horizon_group": "8_15",
        "desktop_only": True,
    },
    {
        "id": "finder_new_folder",
        "task": (
            "Open Finder, navigate to the Desktop, create a new folder named 'COS_Test_Folder', "
            "then verify the folder exists."
        ),
        "tags": ["multi_step", "finder", "file_system"],
        "horizon_group": "8_15",
        "desktop_only": True,
    },
    {
        "id": "safari_navigate",
        "task": "Open Safari, navigate to apple.com in the address bar, and report the page title.",
        "tags": ["browser", "navigation"],
        "horizon_group": "8_15",
        "desktop_only": False,
    },
    {
        "id": "calculator_compute",
        "task": "Open Calculator, compute 1234 divided by 56, and report the result.",
        "tags": ["app_launch", "multi_step", "read"],
        "horizon_group": "8_15",
        "desktop_only": True,
    },
    {
        "id": "system_notifications",
        "task": "Open System Settings, navigate to Notifications, and report which app is at the top of the list.",
        "tags": ["system", "settings", "read"],
        "horizon_group": "8_15",
        "desktop_only": True,
    },
    {
        "id": "finder_sort_by_date",
        "task": (
            "Open Finder, navigate to the Downloads folder, "
            "and sort the files by Date Modified (most recent first). Report the name of the most recent file."
        ),
        "tags": ["finder", "navigation", "read"],
        "horizon_group": "8_15",
        "desktop_only": True,
    },
    {
        "id": "textedit_find_replace",
        "task": (
            "Open TextEdit, create a new plain-text document, type 'The quick brown fox jumps over the lazy dog', "
            "then use Find & Replace to replace 'fox' with 'cat'. Report the final text."
        ),
        "tags": ["multi_step", "typing", "find_replace"],
        "horizon_group": "8_15",
        "desktop_only": True,
    },
    {
        "id": "app_switch_and_interact",
        "task": (
            "Open Calculator, then open TextEdit. Switch back to Calculator using Cmd+Tab, "
            "compute 99 plus 1, then switch back to TextEdit and type the result."
        ),
        "tags": ["multi_step", "app_switch", "typing"],
        "horizon_group": "8_15",
        "desktop_only": True,
    },
    {
        "id": "safari_bookmark_page",
        "task": (
            "Open Safari, navigate to example.com, "
            "then add it to bookmarks and report whether the bookmark was saved."
        ),
        "tags": ["browser", "multi_step", "bookmark"],
        "horizon_group": "8_15",
        "desktop_only": False,
    },
    {
        "id": "terminal_list_files",
        "task": (
            "Open Terminal and run the command 'ls ~/Documents' to list the files "
            "in the Documents folder. Report the first 5 file names."
        ),
        "tags": ["terminal", "multi_step", "read"],
        "horizon_group": "8_15",
        "desktop_only": True,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # HORIZON 16-30  (longer workflows requiring planning)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "create_and_edit_document",
        "task": (
            "Open TextEdit, create a new Rich Text document, type a short paragraph about AI assistants "
            "(at least 3 sentences), make the first sentence bold, change the font to Georgia size 14, "
            "then select all text and check the word count via the Edit menu."
        ),
        "tags": ["multi_step", "typing", "formatting", "long_horizon"],
        "horizon_group": "16_30",
        "desktop_only": True,
    },
    {
        "id": "finder_organize_files",
        "task": (
            "Open Finder, navigate to the Desktop, create a new folder named 'Organized', "
            "then open a second Finder window showing the Downloads folder. "
            "Switch back to the first window and verify 'Organized' exists on the Desktop."
        ),
        "tags": ["finder", "multi_step", "file_system", "long_horizon"],
        "horizon_group": "16_30",
        "desktop_only": True,
    },
    {
        "id": "system_settings_tour",
        "task": (
            "Open System Settings. Visit the following sections and report one setting from each: "
            "1) General (report macOS version), "
            "2) Sound (report output volume), "
            "3) Displays (report resolution)."
        ),
        "tags": ["system", "settings", "multi_step", "long_horizon"],
        "horizon_group": "16_30",
        "desktop_only": True,
    },
    {
        "id": "safari_research",
        "task": (
            "Open Safari, navigate to wikipedia.org, search for 'Large language model', "
            "and report the first sentence of the article."
        ),
        "tags": ["browser", "multi_step", "long_horizon"],
        "horizon_group": "16_30",
        "desktop_only": False,
    },
    {
        "id": "terminal_system_info",
        "task": (
            "Open Terminal. Run the following commands in sequence and report the output of each: "
            "1) 'uname -m' (architecture), "
            "2) 'sw_vers -productVersion' (macOS version), "
            "3) 'whoami' (current user)."
        ),
        "tags": ["terminal", "multi_step", "long_horizon"],
        "horizon_group": "16_30",
        "desktop_only": True,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # HORIZON 30+  (complex multi-app workflows)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "research_and_document",
        "task": (
            "1. Open Safari and navigate to en.wikipedia.org/wiki/Python_(programming_language). "
            "2. Find and copy the first sentence of the article. "
            "3. Open TextEdit, create a new plain-text document. "
            "4. Paste the copied sentence. "
            "5. Below it, type: 'Source: Wikipedia'. "
            "6. Make the word 'Wikipedia' bold. "
            "7. Report the full content of the document."
        ),
        "tags": ["browser", "multi_step", "long_horizon", "cross_app"],
        "horizon_group": "30plus",
        "desktop_only": False,
    },
    {
        "id": "multi_app_workflow",
        "task": (
            "1. Open Calculator and compute 2 to the power of 10 (use repeated multiplication). "
            "2. Remember the result. "
            "3. Open TextEdit, create a new plain-text document. "
            "4. Type: '2^10 = [result]' replacing [result] with the actual value. "
            "5. Open Notes and create a new note with the same text. "
            "6. Report the value you computed and confirm both apps show the same text."
        ),
        "tags": ["multi_step", "long_horizon", "cross_app"],
        "horizon_group": "30plus",
        "desktop_only": True,
    },
    {
        "id": "terminal_and_document",
        "task": (
            "1. Open Terminal and run 'date' to get the current date and time. "
            "2. Run 'hostname' to get the machine name. "
            "3. Run 'python3 --version' to get the Python version. "
            "4. Open TextEdit, create a new plain-text document. "
            "5. Type a summary with three lines: Date, Hostname, Python version. "
            "6. Report the complete document content."
        ),
        "tags": ["terminal", "multi_step", "long_horizon", "cross_app"],
        "horizon_group": "30plus",
        "desktop_only": True,
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_MODEL_LABEL = "UI-TARS" if os.environ.get("UITARS_BASE_URL") else "Gemini"

# RQ21 error taxonomy: map exception substrings → category
_ERROR_CATEGORIES = [
    ("perception",         ["screenshot", "screen capture", "image"]),
    ("grounding",          ["coordinate", "click", "element not found", "ax"]),
    ("planning",           ["plan", "gemini", "model", "rate limit", "quota"]),
    ("action_selection",   ["action", "tool"]),
    ("execution",          ["pyautogui", "keyboard", "mouse"]),
    ("verification",       ["verif", "verify"]),
    ("connector",          ["connector", "api error", "http"]),
    ("model_api",          ["timeout", "429", "500", "503"]),
]

def _classify_error(exc_str: str) -> str:
    low = exc_str.lower()
    for cat, keywords in _ERROR_CATEGORIES:
        if any(kw in low for kw in keywords):
            return cat
    return "unknown"


async def run_task(
    task_def: dict,
    timeout: int = 120,
    ax_mode: str = "both",
    verify_mode: str = "two_stage",
    stuck_mode: str = "detect",
) -> dict:
    """Run one task and return a result dict."""
    from src.agents.nav_agent import NavAgent

    agent = NavAgent(ax_mode=ax_mode, verify_mode=verify_mode, stuck_mode=stuck_mode)
    task_str = task_def["task"]
    model_label = "UI-TARS" if os.environ.get("UITARS_BASE_URL") else "Gemini"
    start = time.time()

    base = {
        "id":            task_def["id"],
        "task":          task_str,
        "tags":          task_def.get("tags", []),
        "horizon_group": task_def.get("horizon_group", "unknown"),
        "model":         model_label,
        "tokens_in":     0,
        "tokens_out":    0,
        "model_calls":   0,
    }

    try:
        result = await asyncio.wait_for(
            agent.run(task_str, context=""),
            timeout=timeout,
        )
        duration = round(time.time() - start, 1)
        # Pull cost stats if NavAgent exposes them
        tokens_in  = getattr(result, "tokens_in",  0) or 0
        tokens_out = getattr(result, "tokens_out", 0) or 0
        model_calls = getattr(result, "model_calls", len(result.steps)) or len(result.steps)
        return {**base,
            "status":      result.status,
            "verified":    result.verified,
            "steps":       len(result.steps),
            "duration":    duration,
            "result":      result.result[:300],
            "tokens_in":   tokens_in,
            "tokens_out":  tokens_out,
            "model_calls": model_calls,
            "error_cat":   "" if result.verified else "max_iter",
        }
    except asyncio.TimeoutError:
        # Recover any steps that completed before the timeout fired
        partial = getattr(agent, "_partial_result", None)
        steps_done = len(partial.steps) if partial and partial.steps else 0
        return {**base,
            "status":    "timeout",
            "verified":  False,
            "steps":     steps_done,
            "duration":  round(time.time() - start, 1),
            "result":    f"Timed out after {timeout}s ({steps_done} steps done)",
            "error_cat": "model_api",
        }
    except Exception as exc:
        exc_str = str(exc)
        return {**base,
            "status":    "error",
            "verified":  False,
            "steps":     0,
            "duration":  round(time.time() - start, 1),
            "result":    exc_str[:300],
            "error_cat": _classify_error(exc_str),
        }


def _vtcr(results: list[dict]) -> float:
    """VTCR = verified / total."""
    total = len(results)
    if total == 0:
        return 0.0
    return round(sum(1 for r in results if r["verified"]) / total, 3)


def _horizon_breakdown(results: list[dict]) -> dict:
    """VTCR per horizon group (RQ5)."""
    groups: dict[str, list] = {}
    for r in results:
        g = r.get("horizon_group", "unknown")
        groups.setdefault(g, []).append(r)
    order = ["1_3", "4_7", "8_15", "16_30", "30plus", "unknown"]
    return {g: _vtcr(groups[g]) for g in order if g in groups}


def _error_breakdown(results: list[dict]) -> dict:
    """Failure category counts (RQ21)."""
    cats: dict[str, int] = {}
    for r in results:
        if not r.get("verified"):
            cat = r.get("error_cat") or "unknown"
            cats[cat] = cats.get(cat, 0) + 1
    return dict(sorted(cats.items(), key=lambda x: -x[1]))
    verified = sum(1 for r in results if r["verified"])
    """Verified Task Completion Rate = verified_done / total."""
    total = len(results)
    if total == 0:
        return 0.0
    verified = sum(1 for r in results if r["verified"] and r["status"] == "done")
    return round(verified / total, 3)


def _print_report(results: list[dict]) -> None:
    """Print human-readable table + VTCR + horizon breakdown + cost summary."""
    print(f"\n{'─'*80}")
    print(f"{'ID':<28} {'HOR':<8} {'STATUS':<12} {'VER':<5} {'STEPS':<7} {'DUR':>6}")
    print(f"{'─'*80}")
    for r in results:
        ver = "✓" if r["verified"] else "✗"
        hor = r.get("horizon_group", "?")
        print(
            f"{r['id']:<28} {hor:<8} {r['status']:<12} {ver:<5} {r['steps']:<7} {r['duration']:>5}s"
        )
    print(f"{'─'*80}")

    total      = len(results)
    done       = sum(1 for r in results if r["status"] == "done")
    max_iter   = sum(1 for r in results if r["status"] == "max_iter")
    verified   = sum(1 for r in results if r["verified"])
    vtcr       = _vtcr(results)
    avg_steps  = round(sum(r["steps"] for r in results) / total, 1) if total else 0
    avg_dur    = round(sum(r["duration"] for r in results) / total, 1) if total else 0
    total_tok  = sum(r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in results)
    model      = results[0]["model"] if results else "—"
    print(f"{'─'*72}")

    total     = len(results)
    done      = sum(1 for r in results if r["status"] == "done")
    max_iter  = sum(1 for r in results if r["status"] == "max_iter")
    verified  = sum(1 for r in results if r["verified"])
    verified  = sum(1 for r in results if r["verified"] and r["status"] == "done")
    vtcr      = _vtcr(results)
    avg_steps = round(sum(r["steps"] for r in results) / total, 1) if total else 0
    avg_dur   = round(sum(r["duration"] for r in results) / total, 1) if total else 0
    model     = results[0]["model"] if results else "—"

    print(f"\nModel          : {model}")
    print(f"Tasks run      : {total}")
    print(f"Done           : {done}/{total}  (max_iter: {max_iter})")
    print(f"Done           : {done}/{total}")
    print(f"VTCR           : {verified}/{total} = {vtcr:.1%}  ← north-star metric")
    print(f"Avg steps      : {avg_steps}")
    print(f"Avg duration   : {avg_dur}s")
    print(f"Total tokens   : {total_tok}")

    hb = _horizon_breakdown(results)
    if hb:
        print(f"\nVTCR by horizon (RQ5):")
        for g, v in hb.items():
            print(f"  {g:<10} {v:.1%}")

    eb = _error_breakdown(results)
    if eb:
        print(f"\nFailure categories (RQ21):")
        for cat, cnt in eb.items():
            print(f"  {cat:<22} {cnt}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Map RQ experiment groups to horizon/tag filters
_GROUP_FILTERS: dict[str, dict] = {
    "A":  {},                                  # all tasks — baseline
    "A1": {"horizon": "1_3"},
    "A5": {},                                  # long-horizon — all tasks, breakdown by horizon
    "B":  {"tags": ["multi_step"]},            # ablation candidates
}

async def _main(args: argparse.Namespace) -> None:
    # Load .env if present
    env_file = _ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    # Filter tasks
    tasks = list(TASKS)
    if args.task:
        tasks = [t for t in tasks if args.task.lower() in t["task"].lower() or args.task == t["id"]]
        if not tasks:
            print(f"No task matching '{args.task}'. Available IDs:")
            for t in TASKS:
                print(f"  {t['id']}: {t['task'][:70]}")
            return

    if args.tags:
        tag_set = {t.strip() for t in args.tags.split(",")}
        tasks = [t for t in tasks if tag_set & set(t.get("tags", []))]

    if args.horizon:
        tasks = [t for t in tasks if t.get("horizon_group") == args.horizon]

    if args.limit and args.limit > 0:
        tasks = tasks[:args.limit]

    if args.dry_run:
        print(f"\n{'ID':<28} {'HOR':<8} TASK")
        print("─" * 80)
        for t in tasks:
            print(f"{t['id']:<28} {t.get('horizon_group','?'):<8} {t['task'][:44]}")
        print(f"\n{len(tasks)} tasks. Run without --dry-run to execute.")
        return

    model_label = "UI-TARS" if os.environ.get("UITARS_BASE_URL") else "Gemini"
    print(f"\nCollectiveOS nav eval — model: {model_label} — {len(tasks)} tasks")
    print("Each task runs live on this Mac. Keep the screen visible.\n")

    results: list[dict] = []
    for i, task_def in enumerate(tasks, 1):
        hor = task_def.get("horizon_group", "?")
        print(f"[{i}/{len(tasks)}] {task_def['id']} ({hor}) … ", end="", flush=True)
        result = await run_task(
            task_def, timeout=args.timeout,
            ax_mode=args.ax_mode,
            verify_mode=args.verify_mode,
            stuck_mode=args.stuck_mode,
        )
        results.append(result)
        ver = "✓" if result["verified"] else "✗"
        print(f"{result['status']} {ver}  ({result['steps']} steps, {result['duration']}s)")

        # Pause between tasks: screen settles + respects Gemini free-tier rate limits.
        # Use --inter-task-delay 30 on a free-tier key to avoid 429 timeouts.
        if i < len(tasks):
            await asyncio.sleep(args.inter_task_delay)
        # Small pause between tasks so the screen settles
        if i < len(tasks):
            await asyncio.sleep(2)

    _print_report(results)

    if args.out:
        out_path = Path(args.out)
        hb = _horizon_breakdown(results)
        eb = _error_breakdown(results)
        out_path.write_text(json.dumps(
            {
                "model":              model_label,
                "ax_mode":            args.ax_mode,
                "verify_mode":        args.verify_mode,
                "stuck_mode":         args.stuck_mode,
                "vtcr":               _vtcr(results),
                "total":              len(results),
                "vtcr_by_horizon":    hb,
                "error_categories":   eb,
                "total_tokens":       sum(r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in results),
                "results":            results,
            },
            indent=2,
        ))
        print(f"Report saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CollectiveOS nav eval harness (VTCR)")
    parser.add_argument("--task",     help="Run only tasks matching this id or substring")
    parser.add_argument("--tags",     help="Comma-separated tags to filter tasks (any match)")
    parser.add_argument("--horizon",     help="Filter by horizon group: 1_3 | 4_7 | 8_15 | 16_30 | 30plus")
    # Ablation flags (RQ2, RQ3, RQ4)
    parser.add_argument("--ax-mode",     default="both", dest="ax_mode",
                        choices=["both", "screenshot_only", "ax_only"],
                        help="RQ2: grounding modality (default: both)")
    parser.add_argument("--verify-mode", default="two_stage", dest="verify_mode",
                        choices=["two_stage", "gemini_only", "deterministic_only", "none"],
                        help="RQ3: verification strategy (default: two_stage)")
    parser.add_argument("--stuck-mode",  default="detect", dest="stuck_mode",
                        choices=["detect", "recover", "none"],
                        help="RQ4: stuck detection strategy (default: detect)")
    parser.add_argument("--timeout",     type=int, default=120, help="Per-task timeout seconds (default 120)")
    parser.add_argument("--limit",    type=int, default=0, help="Cap number of tasks (0 = all)")
    parser.add_argument("--dry-run",  action="store_true", help="List tasks without running")
    parser.add_argument("--out",      help="Save JSON report to this path")
    parser.add_argument(
        "--inter-task-delay", type=int, default=5, dest="inter_task_delay",
        help="Seconds to wait between tasks (default 5). Use 30 on Gemini free tier to avoid 429s.",
    )
    asyncio.run(_main(parser.parse_args()))
