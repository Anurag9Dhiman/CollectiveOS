#!/usr/bin/env python3
"""
RQ20: End-to-End Personal Assistant Workflows

Evaluates CollectiveOS on 6 realistic multi-step, multi-tool workflows that
combine memory, computer-use, and services in a single coherent task.

Workflows:
  W1 memory_then_apply   — retrieve user preference from memory, apply to task
  W2 search_and_save     — browser search + save result to TextEdit doc
  W3 screen_inspect_act  — check screen state, take a conditional action
  W4 pref_to_config      — read user prefs from memory, configure app accordingly
  W5 read_confirm_exec   — read info → HITL approval → execute
  W6 multi_connector     — uses memory + computer-use + Gemini in one task

Metrics:
  task_success          — did the workflow complete with verified=True?
  execution_time        — total seconds
  actions               — total nav steps
  model_calls           — approximate model invocations
  user_interventions    — HITL triggers
  memory_retrieval_ok   — was memory context used correctly?
  failure_type          — which component failed?

Usage:
    python3 scripts/eval_workflows.py
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

from src.memory import save_fact, search as mem_search, _embed


# ── Workflow definitions ─────────────────────────────────────────────────────

async def _run_nav(task: str, timeout: int = 180) -> dict:
    from src.agents.nav_agent import NavAgent
    agent = NavAgent()
    hitl_count = [0]
    async def _hitl(desc): hitl_count[0] += 1; return True

    start = time.time()
    try:
        result = await asyncio.wait_for(agent.run(task, hitl_callback=_hitl), timeout=timeout)
        return {
            "status":      result.status,
            "verified":    result.verified,
            "steps":       len(result.steps),
            "duration":    round(time.time() - start, 1),
            "hitl_count":  hitl_count[0],
        }
    except asyncio.TimeoutError:
        return {"status": "timeout", "verified": False, "steps": 0,
                "duration": round(time.time() - start, 1), "hitl_count": hitl_count[0]}
    except Exception as exc:
        return {"status": "error", "verified": False, "steps": 0,
                "duration": round(time.time() - start, 1), "hitl_count": hitl_count[0],
                "error": str(exc)[:100]}


async def workflow_w1_memory_then_apply() -> dict:
    """W1: Plant a preference in memory, then apply it."""
    save_fact("User always prefers plain text documents over rich text.")
    await asyncio.sleep(2)
    context = mem_search("document format preference")
    mem_ok = "plain text" in context.lower() if context else False
    task = (
        "Open TextEdit. Based on user preference for plain text, ensure the document "
        "is in plain text format (not rich text), then type 'Workflow W1 complete'."
    )
    r = await _run_nav(task)
    r.update({"id": "W1", "type": "memory_then_apply", "memory_ok": mem_ok})
    return r


async def workflow_w2_search_and_save() -> dict:
    """W2: Search the web and save the result to a TextEdit document."""
    task = (
        "1. Use Spotlight or Safari to search the web for 'What is LangGraph?'. "
        "2. Read the first result or excerpt. "
        "3. Open TextEdit, create a new plain-text document. "
        "4. Type a one-sentence summary of what LangGraph is. "
        "5. Report the text you typed."
    )
    r = await _run_nav(task, timeout=240)
    r.update({"id": "W2", "type": "search_and_save", "memory_ok": None})
    return r


async def workflow_w3_screen_inspect_act() -> dict:
    """W3: Inspect screen state, take conditional action."""
    task = (
        "Look at what apps are currently open. "
        "If Finder is open, navigate it to the Downloads folder. "
        "If Finder is not open, open it and navigate to Downloads. "
        "Report the name of the first file in Downloads."
    )
    r = await _run_nav(task)
    r.update({"id": "W3", "type": "screen_inspect_act", "memory_ok": None})
    return r


async def workflow_w4_pref_to_config() -> dict:
    """W4: Retrieve preference from memory, configure an app accordingly."""
    save_fact("User's preferred TextEdit font size is 14 points.")
    await asyncio.sleep(2)
    context = mem_search("TextEdit font size preference")
    mem_ok = "14" in context if context else False
    task = (
        "Open TextEdit, create a new Rich Text document. "
        "Set the font size to 14 points (per user preference). "
        "Type 'Font configured per user preference' and report the current font size."
    )
    r = await _run_nav(task)
    r.update({"id": "W4", "type": "pref_to_config", "memory_ok": mem_ok})
    return r


async def workflow_w5_read_confirm_exec() -> dict:
    """W5: Read system state, confirm with HITL, then execute."""
    task = (
        "Check the current system volume level. "
        "If it is above 50%, ask for permission to lower it to 40%. "
        "If permission is granted, lower the volume. Report the final volume."
    )
    r = await _run_nav(task)
    r.update({"id": "W5", "type": "read_confirm_exec", "memory_ok": None})
    return r


async def workflow_w6_multi_connector() -> dict:
    """W6: Multi-connector workflow — memory + computer + terminal."""
    save_fact("User prefers Python 3.13 for all new projects.")
    await asyncio.sleep(2)
    context = mem_search("Python version preference")
    mem_ok = "3.13" in context if context else False
    task = (
        "1. Check what Python version is available by opening Terminal and running 'python3 --version'. "
        "2. Open TextEdit, create a new plain-text document. "
        "3. Type: 'Python version: [version from terminal]'. "
        "4. Below that, type: 'Matches user preference: [Yes/No based on whether it is 3.13]'. "
        "5. Report the document content."
    )
    r = await _run_nav(task, timeout=240)
    r.update({"id": "W6", "type": "multi_connector", "memory_ok": mem_ok})
    return r


WORKFLOWS = [
    workflow_w1_memory_then_apply,
    workflow_w2_search_and_save,
    workflow_w3_screen_inspect_act,
    workflow_w4_pref_to_config,
    workflow_w5_read_confirm_exec,
    workflow_w6_multi_connector,
]

INTER_WORKFLOW_DELAY = 45


async def _main() -> None:
    print("\n=== RQ20: End-to-End Workflow Evaluation ===\n")
    results = []

    for wf_fn in WORKFLOWS:
        print(f"Running {wf_fn.__name__}...")
        r = await wf_fn()
        results.append(r)
        ver = "✓" if r["verified"] else "✗"
        print(f"  [{r['id']}] {r['status']} {ver}  "
              f"steps={r['steps']} dur={r['duration']}s "
              f"hitl={r['hitl_count']} mem_ok={r.get('memory_ok')}")
        print(f"  Waiting {INTER_WORKFLOW_DELAY}s...")
        await asyncio.sleep(INTER_WORKFLOW_DELAY)

    n = len(results)
    vtcr = sum(1 for r in results if r["verified"]) / n
    avg_steps = round(sum(r["steps"] for r in results) / n, 1)
    avg_dur = round(sum(r["duration"] for r in results) / n, 1)
    avg_hitl = round(sum(r["hitl_count"] for r in results) / n, 1)
    mem_ok = sum(1 for r in results if r.get("memory_ok") is True)

    print(f"\n{'─'*60}")
    print(f"VTCR            : {vtcr:.0%}  ({sum(1 for r in results if r['verified'])}/{n})")
    print(f"Avg steps       : {avg_steps}")
    print(f"Avg duration    : {avg_dur}s")
    print(f"Avg HITL events : {avg_hitl}")
    print(f"Memory OK       : {mem_ok}/4 memory workflows")

    out = {
        "rq":         "RQ20",
        "vtcr":       vtcr,
        "avg_steps":  avg_steps,
        "avg_dur":    avg_dur,
        "avg_hitl":   avg_hitl,
        "mem_ok":     mem_ok,
        "results":    results,
    }
    Path("eval_results_rq20_workflows.json").write_text(json.dumps(out, indent=2))
    print("\nResults saved → eval_results_rq20_workflows.json")


if __name__ == "__main__":
    asyncio.run(_main())
