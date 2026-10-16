#!/usr/bin/env python3
"""
RQ16: API Rate Limits and Planning Robustness

Injects artificial delays into Gemini API calls to simulate:
  1. unlimited_retry    — original behavior, no cap
  2. original_schedule  — (8, 20, 45)s waits
  3. reduced_schedule   — (4, 10, 20)s waits
  4. fixed_timeout_30   — 30s planning timeout (current behavior)
  5. fixed_timeout_15   — 15s planning timeout (more aggressive cap)

Measures per config:
  - task success (VTCR)
  - avg planning time
  - avg execution time
  - timeout rate (planning aborted)
  - retry count
  - % of task budget consumed by planning

Uses a small representative subset (10 tasks) to keep run time manageable.

Usage:
    python3 scripts/eval_rate_limits.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

for _line in (_ROOT / ".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        k, _, v = _line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

# ── Test tasks (horizon 1_3 + 4_7 for speed) ─────────────────────────────────
RQ16_TASKS = [
    {"id": "launch_calculator",   "task": "Open the Calculator app."},
    {"id": "launch_finder",       "task": "Open a new Finder window."},
    {"id": "navigate_downloads",  "task": "Open the Downloads folder in Finder."},
    {"id": "check_wifi",          "task": "Open System Settings, navigate to Wi-Fi, and report status."},
    {"id": "spotlight_math",      "task": "Use Spotlight to calculate 47 multiplied by 23."},
]

INTER_TASK_DELAY = 30

# ── Configs ──────────────────────────────────────────────────────────────────
CONFIGS = {
    "fixed_timeout_30":  {"retry_waits": (8, 20, 45), "plan_timeout": 30},
    "fixed_timeout_15":  {"retry_waits": (8, 20, 45), "plan_timeout": 15},
    "original_schedule": {"retry_waits": (8, 20, 45), "plan_timeout": None},
    "reduced_schedule":  {"retry_waits": (4, 10, 20), "plan_timeout": None},
}


async def run_task_with_config(task_str: str, config: dict, timeout: int = 150) -> dict:
    """Patch nav_agent constants and run a task."""
    import src.agents.nav_agent as nav_mod
    from src.agents.nav_agent import NavAgent

    retry_waits = config["retry_waits"]
    plan_timeout = config.get("plan_timeout")

    planning_start = None
    planning_end = None
    plan_timed_out = False

    original_plan = NavAgent._plan_task
    original_waits = nav_mod._RETRY_WAITS

    async def _timed_plan(self, task, context):
        nonlocal planning_start, planning_end, plan_timed_out
        planning_start = time.time()
        if plan_timeout is not None:
            try:
                result = await asyncio.wait_for(
                    original_plan(self, task, context),
                    timeout=plan_timeout,
                )
                planning_end = time.time()
                return result
            except asyncio.TimeoutError:
                planning_end = time.time()
                plan_timed_out = True
                return []
        else:
            result = await original_plan(self, task, context)
            planning_end = time.time()
            return result

    nav_mod._RETRY_WAITS = retry_waits
    NavAgent._plan_task = _timed_plan

    agent = NavAgent()
    start = time.time()
    try:
        result = await asyncio.wait_for(agent.run(task_str), timeout=timeout)
        duration = round(time.time() - start, 1)
        plan_dur = round((planning_end or start) - (planning_start or start), 1) if planning_start else 0
        return {
            "status":           result.status,
            "verified":         result.verified,
            "steps":            len(result.steps),
            "duration":         duration,
            "planning_time_s":  plan_dur,
            "plan_timed_out":   plan_timed_out,
            "plan_pct_budget":  round(plan_dur / duration * 100, 1) if duration > 0 else 0,
        }
    except asyncio.TimeoutError:
        return {
            "status": "timeout", "verified": False, "steps": 0,
            "duration": round(time.time() - start, 1),
            "planning_time_s": 0, "plan_timed_out": plan_timed_out,
            "plan_pct_budget": 0,
        }
    except Exception as exc:
        return {
            "status": "error", "verified": False, "steps": 0,
            "duration": round(time.time() - start, 1),
            "planning_time_s": 0, "plan_timed_out": plan_timed_out,
            "plan_pct_budget": 0, "error": str(exc)[:100],
        }
    finally:
        nav_mod._RETRY_WAITS = original_waits
        NavAgent._plan_task = original_plan


async def _main() -> None:
    print("\n=== RQ16: Rate Limit and Planning Robustness ===\n")
    config_results: dict[str, dict] = {}

    for config_name, config in CONFIGS.items():
        print(f"\n[Config: {config_name}]")
        results = []
        for task_def in RQ16_TASKS:
            print(f"  {task_def['id']}...", end="", flush=True)
            r = await run_task_with_config(task_def["task"], config)
            r["id"] = task_def["id"]
            results.append(r)
            print(f" {r['status']} {'✓' if r['verified'] else '✗'}  "
                  f"plan={r['planning_time_s']:.1f}s  total={r['duration']}s"
                  + (" [PLAN TIMEOUT]" if r.get("plan_timed_out") else ""))
            await asyncio.sleep(INTER_TASK_DELAY)

        n = len(results)
        vtcr = sum(1 for r in results if r["verified"]) / n
        avg_plan = round(sum(r["planning_time_s"] for r in results) / n, 1)
        avg_dur  = round(sum(r["duration"] for r in results) / n, 1)
        timeout_rate = sum(1 for r in results if r.get("plan_timed_out")) / n
        avg_budget_pct = round(sum(r["plan_pct_budget"] for r in results) / n, 1)

        config_results[config_name] = {
            "vtcr":            vtcr,
            "avg_planning_s":  avg_plan,
            "avg_duration_s":  avg_dur,
            "timeout_rate":    timeout_rate,
            "avg_budget_pct":  avg_budget_pct,
            "results":         results,
        }
        print(f"  → VTCR={vtcr:.0%} plan={avg_plan}s total={avg_dur}s "
              f"timeouts={timeout_rate:.0%} plan_budget={avg_budget_pct:.0f}%")

    print(f"\n{'─'*72}")
    print(f"{'Config':<22} {'VTCR':>6} {'Plan(s)':>8} {'Total(s)':>9} {'Timeouts':>9} {'Budget%':>8}")
    print("─" * 72)
    for name, m in config_results.items():
        print(f"{name:<22} {m['vtcr']:>6.0%} {m['avg_planning_s']:>8.1f} "
              f"{m['avg_duration_s']:>9.1f} {m['timeout_rate']:>9.0%} {m['avg_budget_pct']:>8.0f}%")

    out = {"rq": "RQ16", "configs": config_results}
    Path("eval_results_rq16_rate_limits.json").write_text(json.dumps(out, indent=2))
    print("\nResults saved → eval_results_rq16_rate_limits.json")


if __name__ == "__main__":
    asyncio.run(_main())
