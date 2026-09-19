#!/usr/bin/env python3
"""
OSWorld-style evaluation harness for CollectiveOS nav agent.

Runs a fixed suite of standardised desktop tasks and measures
VTCR (Verified Task Completion Rate = verified_done / total_attempted).

Usage:
    python scripts/eval_nav.py                 # run full suite
    python scripts/eval_nav.py --task "open Calculator"   # single task
    python scripts/eval_nav.py --dry-run       # list tasks, no execution
    python scripts/eval_nav.py --out results.json         # save report

Each task records:
  status    — "done" | "max_iter" | "error" | "stopped"
  verified  — bool (did the verifier confirm completion?)
  steps     — number of perceive-act iterations used
  duration  — wall-clock seconds
  model     — "UI-TARS" | "Gemini"
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
    # ── App launch (1–2 steps, high confidence baseline) ────────────────────
    {
        "id": "launch_calculator",
        "task": "Open the Calculator app.",
        "tags": ["app_launch", "spotlight"],
        "desktop_only": True,
    },
    {
        "id": "launch_textedit",
        "task": "Open TextEdit and create a new blank document.",
        "tags": ["app_launch", "new_doc"],
        "desktop_only": True,
    },
    {
        "id": "launch_finder",
        "task": "Open a new Finder window.",
        "tags": ["app_launch", "finder"],
        "desktop_only": True,
    },
    # ── File system navigation ───────────────────────────────────────────────
    {
        "id": "navigate_downloads",
        "task": "Open the Downloads folder in Finder.",
        "tags": ["finder", "navigation"],
        "desktop_only": True,
    },
    {
        "id": "goto_desktop",
        "task": "Show the desktop (hide all open windows).",
        "tags": ["os_action", "show_desktop"],
        "desktop_only": True,
    },
    # ── Clipboard & text ────────────────────────────────────────────────────
    {
        "id": "read_clipboard",
        "task": "Read whatever text is currently on the clipboard and report it.",
        "tags": ["clipboard", "read"],
        "desktop_only": True,
    },
    # ── System controls ──────────────────────────────────────────────────────
    {
        "id": "system_info",
        "task": "Open System Settings (or System Preferences) on the General page.",
        "tags": ["system", "settings"],
        "desktop_only": True,
    },
    {
        "id": "volume_check",
        "task": "Open Sound settings and report the current output volume level.",
        "tags": ["system", "settings", "read"],
        "desktop_only": True,
    },
    # ── Spotlight ────────────────────────────────────────────────────────────
    {
        "id": "spotlight_search",
        "task": "Use Spotlight to search for 'Terminal' and report the top result.",
        "tags": ["spotlight", "search"],
        "desktop_only": True,
    },
    # ── Multi-step interaction ───────────────────────────────────────────────
    {
        "id": "textedit_type",
        "task": (
            "Open TextEdit, create a new plain-text document, "
            "type the text 'Hello CollectiveOS', and leave it open (do not save)."
        ),
        "tags": ["multi_step", "typing"],
        "desktop_only": True,
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_task(task_def: dict, timeout: int = 120) -> dict:
    """Run one task and return a result dict."""
    from src.agents.nav_agent import NavAgent

    agent = NavAgent()
    task_str = task_def["task"]
    start = time.time()

    try:
        result = await asyncio.wait_for(
            agent.run(task_str, context=""),
            timeout=timeout,
        )
        duration = round(time.time() - start, 1)
        return {
            "id":       task_def["id"],
            "task":     task_str,
            "tags":     task_def.get("tags", []),
            "status":   result.status,
            "verified": result.verified,
            "steps":    len(result.steps),
            "duration": duration,
            "result":   result.result[:200],
            "model":    os.environ.get("UITARS_BASE_URL") and "UI-TARS" or "Gemini",
        }
    except asyncio.TimeoutError:
        return {
            "id":       task_def["id"],
            "task":     task_str,
            "tags":     task_def.get("tags", []),
            "status":   "timeout",
            "verified": False,
            "steps":    0,
            "duration": round(time.time() - start, 1),
            "result":   f"Timed out after {timeout}s",
            "model":    os.environ.get("UITARS_BASE_URL") and "UI-TARS" or "Gemini",
        }
    except Exception as exc:
        return {
            "id":       task_def["id"],
            "task":     task_str,
            "tags":     task_def.get("tags", []),
            "status":   "error",
            "verified": False,
            "steps":    0,
            "duration": round(time.time() - start, 1),
            "result":   str(exc)[:200],
            "model":    os.environ.get("UITARS_BASE_URL") and "UI-TARS" or "Gemini",
        }


def _vtcr(results: list[dict]) -> float:
    """Verified Task Completion Rate = verified / total.

    A task is counted as successful when verified=True regardless of whether
    status is 'done' or 'max_iter' — reaching the goal is what matters.
    """
    total = len(results)
    if total == 0:
        return 0.0
    verified = sum(1 for r in results if r["verified"])
    """Verified Task Completion Rate = verified_done / total."""
    total = len(results)
    if total == 0:
        return 0.0
    verified = sum(1 for r in results if r["verified"] and r["status"] == "done")
    return round(verified / total, 3)


def _print_report(results: list[dict]) -> None:
    """Print a human-readable table and summary."""
    col_w = 30
    print(f"\n{'─'*72}")
    print(f"{'ID':<25} {'STATUS':<14} {'VER':<5} {'STEPS':<7} {'DUR':>6}")
    print(f"{'─'*72}")
    for r in results:
        ver = "✓" if r["verified"] else "✗"
        print(
            f"{r['id']:<25} {r['status']:<14} {ver:<5} {r['steps']:<7} {r['duration']:>5}s"
        )
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
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    tasks = TASKS
    if args.task:
        tasks = [t for t in tasks if args.task.lower() in t["task"].lower() or args.task == t["id"]]
        if not tasks:
            print(f"No task matching '{args.task}'. Available IDs:")
            for t in TASKS:
                print(f"  {t['id']}: {t['task'][:70]}")
            return

    if args.dry_run:
        print(f"\n{'ID':<25} TASK")
        print("─" * 72)
        for t in tasks:
            print(f"{t['id']:<25} {t['task'][:46]}")
        print(f"\n{len(tasks)} tasks. Run without --dry-run to execute.")
        return

    model_label = "UI-TARS" if os.environ.get("UITARS_BASE_URL") else "Gemini"
    print(f"\nCollectiveOS nav eval — model: {model_label} — {len(tasks)} tasks")
    print("Each task runs live on this Mac. Keep the screen visible.\n")

    results: list[dict] = []
    for i, task_def in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task_def['id']} … ", end="", flush=True)
        result = await run_task(task_def, timeout=args.timeout)
        results.append(result)
        ver = "✓" if result["verified"] else "✗"
        print(f"{result['status']} {ver}  ({result['steps']} steps, {result['duration']}s)")

        # Pause between tasks: screen settles + respects Gemini free-tier 15 RPM.
        # Use --inter-task-delay 30 on a free-tier key to avoid 429 timeouts.
        if i < len(tasks):
            await asyncio.sleep(args.inter_task_delay)
        # Small pause between tasks so the screen settles
        if i < len(tasks):
            await asyncio.sleep(2)

    _print_report(results)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(
            {
                "model":   model_label,
                "vtcr":    _vtcr(results),
                "total":   len(results),
                "results": results,
            },
            indent=2,
        ))
        print(f"Report saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nav agent eval harness (VTCR)")
    parser.add_argument("--task",    help="Run only tasks matching this id or substring")
    parser.add_argument("--timeout", type=int, default=120, help="Per-task timeout seconds (default 120)")
    parser.add_argument("--dry-run", action="store_true", help="List tasks without running")
    parser.add_argument("--out",     help="Save JSON report to this path")
    parser.add_argument(
        "--inter-task-delay", type=int, default=5, dest="inter_task_delay",
        help="Seconds to wait between tasks (default 5). Use 30 on Gemini free tier to avoid 429s.",
    )
    asyncio.run(_main(parser.parse_args()))
