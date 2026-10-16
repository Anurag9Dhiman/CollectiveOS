#!/usr/bin/env python3
"""
RQ19: Cross-Modal Context Sharing

Tests whether the VisualOS visual context (scan session) enriches the task
agent's performance on tasks that have a screen-dependent component.

Design:
  - 6 tasks split into two conditions:
    a) no_context  — task agent receives no VisualOS context
    b) with_context — a synthetic VisualOS context blob is injected as prefix

  - Context blob contains: current app, visible UI elements, last screenshot summary
  - Measured: VTCR, avg steps, first-action accuracy (did it click the right thing?)

Synthetic context is generated once and reused across tasks to be fair.
Real VisualOS context would arrive via multi_agent._fetch_visual_context().

Metrics:
  VTCR_no_context     — baseline VTCR without any visual context
  VTCR_with_context   — VTCR with injected context
  delta_vtcr          — improvement
  avg_steps_no_ctx    — average nav steps without context
  avg_steps_with_ctx  — average nav steps with context
  step_reduction_pct  — % fewer steps with context

Usage:
    python3 scripts/eval_cross_modal.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

for _line in (_ROOT / ".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        k, _, v = _line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

# ── Tasks designed to benefit from screen context ────────────────────────────
TASKS = [
    {
        "id":   "find_open_window",
        "task": "Find the frontmost open application window and report its title.",
        "context": "Currently visible: Finder is in front with window title 'Desktop'.",
    },
    {
        "id":   "click_visible_button",
        "task": "In the currently open app, find and click the Cancel or Close button.",
        "context": "Currently visible: A dialog box is open with a Cancel button at coordinates (410, 320).",
    },
    {
        "id":   "read_status_bar",
        "task": "Report the current battery level shown in the menu bar.",
        "context": "Menu bar shows: Battery 78%, Wi-Fi connected, Bluetooth on, Time 14:23.",
    },
    {
        "id":   "continue_typing",
        "task": "Continue typing from where the cursor is — type 'Context sharing works.'",
        "context": "Currently visible: TextEdit is open with cursor at the end of 'Cross-modal test: '.",
    },
    {
        "id":   "scroll_to_content",
        "task": "Scroll down to see the next section and report its heading.",
        "context": "Currently visible: Safari is open on a Wikipedia article, partially scrolled. Next section heading is 'Architecture' at approx y=800.",
    },
    {
        "id":   "close_dialog",
        "task": "Dismiss any dialog or alert that is currently open.",
        "context": "Currently visible: macOS alert dialog with message 'Do you want to save changes?' and buttons Save, Don't Save, Cancel.",
    },
]

INTER_TASK_DELAY = 30


async def _run_nav(task_str: str, context_prefix: str = "", timeout: int = 150) -> dict:
    from src.agents.nav_agent import NavAgent
    agent = NavAgent()
    full_task = f"[Screen context: {context_prefix}]\n\n{task_str}" if context_prefix else task_str
    start = time.time()
    try:
        result = await asyncio.wait_for(agent.run(full_task), timeout=timeout)
        return {
            "status":   result.status,
            "verified": result.verified,
            "steps":    len(result.steps),
            "duration": round(time.time() - start, 1),
        }
    except asyncio.TimeoutError:
        return {"status": "timeout", "verified": False, "steps": 0,
                "duration": round(time.time() - start, 1)}
    except Exception as exc:
        return {"status": "error", "verified": False, "steps": 0,
                "duration": round(time.time() - start, 1), "error": str(exc)[:80]}


async def _main() -> None:
    print("\n=== RQ19: Cross-Modal Context Sharing ===\n")
    results_no_ctx:   list[dict] = []
    results_with_ctx: list[dict] = []

    for task_def in TASKS:
        print(f"Task: {task_def['id']}")

        # Condition A: no context
        print("  no_context...", end="", flush=True)
        r_no = await _run_nav(task_def["task"])
        r_no["id"] = task_def["id"]
        results_no_ctx.append(r_no)
        print(f" {r_no['status']} {'✓' if r_no['verified'] else '✗'}  steps={r_no['steps']}")
        await asyncio.sleep(INTER_TASK_DELAY)

        # Condition B: with context
        print("  with_context...", end="", flush=True)
        r_ctx = await _run_nav(task_def["task"], context_prefix=task_def["context"])
        r_ctx["id"] = task_def["id"]
        results_with_ctx.append(r_ctx)
        print(f" {r_ctx['status']} {'✓' if r_ctx['verified'] else '✗'}  steps={r_ctx['steps']}")
        await asyncio.sleep(INTER_TASK_DELAY)

    n = len(TASKS)
    vtcr_no  = sum(1 for r in results_no_ctx  if r["verified"]) / n
    vtcr_ctx = sum(1 for r in results_with_ctx if r["verified"]) / n
    steps_no  = sum(r["steps"] for r in results_no_ctx)  / n
    steps_ctx = sum(r["steps"] for r in results_with_ctx) / n
    step_reduction = round((steps_no - steps_ctx) / steps_no * 100, 1) if steps_no > 0 else 0

    print(f"\n{'─'*55}")
    print(f"VTCR (no context)   : {vtcr_no:.0%}")
    print(f"VTCR (with context) : {vtcr_ctx:.0%}  Δ={vtcr_ctx - vtcr_no:+.0%}")
    print(f"Avg steps (no ctx)  : {steps_no:.1f}")
    print(f"Avg steps (with ctx): {steps_ctx:.1f}  reduction={step_reduction:.0f}%")

    out = {
        "rq":                  "RQ19",
        "vtcr_no_context":     vtcr_no,
        "vtcr_with_context":   vtcr_ctx,
        "delta_vtcr":          round(vtcr_ctx - vtcr_no, 3),
        "avg_steps_no_ctx":    round(steps_no, 1),
        "avg_steps_with_ctx":  round(steps_ctx, 1),
        "step_reduction_pct":  step_reduction,
        "results_no_context":  results_no_ctx,
        "results_with_context": results_with_ctx,
    }
    Path("eval_results_rq19_cross_modal.json").write_text(json.dumps(out, indent=2))
    print("\nResults saved → eval_results_rq19_cross_modal.json")


if __name__ == "__main__":
    asyncio.run(_main())
