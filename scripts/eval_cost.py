#!/usr/bin/env python3
"""
RQ24: Cost per Successful Task Tracking

Measures C_success = C_total / N_successful across task horizons and conditions.

Tracks:
  - Input tokens, output tokens, total tokens per task
  - Gemini API cost estimate (using public pricing)
  - Wall-clock time as an operational cost proxy
  - C_success = total cost / # verified tasks

Pricing reference (as of 2024-12):
  gemini-2.0-flash:
    Input:  $0.075 / 1M tokens  (> 200k context: $0.15)
    Output: $0.30  / 1M tokens
  gemini-2.0-flash-lite:
    Input:  $0.0375 / 1M tokens
    Output: $0.15  / 1M tokens
  gemini-1.5-flash:
    Input:  $0.075 / 1M tokens
    Output: $0.30  / 1M tokens

Usage:
    python3 scripts/eval_cost.py
    python3 scripts/eval_cost.py --model gemini-2.0-flash --tasks 20
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

for _line in (_ROOT / ".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        k, _, v = _line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

# ── Pricing table (USD per 1M tokens) ─────────────────────────────────────────
PRICING = {
    "gemini-2.5-flash":      {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-lite": {"input": 0.0375, "output": 0.15},
    "gemini-2.0-flash":      {"input": 0.075, "output": 0.30},
    "models/gemini-3.6-flash": {"input": 0.0375, "output": 0.15},
    "default":               {"input": 0.075, "output": 0.30},
}

INTER_TASK_DELAY = 30

TASKS = [
    {"id": "launch_calculator",  "task": "Open the Calculator app.",                        "horizon": "1_3"},
    {"id": "open_finder",        "task": "Open a new Finder window.",                        "horizon": "1_3"},
    {"id": "spotlight_search",   "task": "Use Spotlight to search for TextEdit.",            "horizon": "1_3"},
    {"id": "navigate_downloads", "task": "Open the Downloads folder in Finder.",             "horizon": "4_7"},
    {"id": "check_wifi",         "task": "Open System Settings, navigate to Wi-Fi, and report its status.", "horizon": "4_7"},
    {"id": "type_hello",         "task": "Open TextEdit, create a new plain text doc, and type 'Cost eval'.", "horizon": "4_7"},
    {"id": "open_safari_google", "task": "Open Safari and navigate to google.com.",          "horizon": "8_15"},
    {"id": "terminal_python",    "task": "Open Terminal and run 'python3 --version'.",       "horizon": "8_15"},
    {"id": "finder_screenshots", "task": "Open Finder, navigate to Pictures, report the file count.", "horizon": "8_15"},
    {"id": "settings_display",   "task": "Open System Settings, navigate to Displays, report the resolution.", "horizon": "16_30"},
]


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = PRICING.get(model, PRICING["default"])
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


async def run_task_with_cost(task_str: str, model_label: str, timeout: int = 150) -> dict:
    from src.agents.nav_agent import NavAgent
    agent = NavAgent()
    start = time.time()
    token_log: list[dict] = []

    original_generate = None
    try:
        import src.agents.nav_agent as nav_mod
        original_generate = getattr(nav_mod, "_gemini_generate", None)
    except Exception:
        pass

    try:
        result = await asyncio.wait_for(agent.run(task_str), timeout=timeout)
        duration = round(time.time() - start, 1)

        # Extract token usage from result if available
        total_input  = getattr(result, "input_tokens",  0) or 0
        total_output = getattr(result, "output_tokens", 0) or 0
        total_tokens = getattr(result, "total_tokens",  0) or (total_input + total_output)

        # Fallback: estimate from step count
        if total_tokens == 0:
            n_steps = len(getattr(result, "steps", []))
            total_input  = n_steps * 4000
            total_output = n_steps * 300

        cost_usd = estimate_cost(total_input, total_output, model_label)

        return {
            "status":        result.status,
            "verified":      result.verified,
            "steps":         len(result.steps),
            "duration":      duration,
            "input_tokens":  total_input,
            "output_tokens": total_output,
            "total_tokens":  total_input + total_output,
            "cost_usd":      round(cost_usd, 6),
        }
    except asyncio.TimeoutError:
        return {"status": "timeout", "verified": False, "steps": 0, "duration": round(time.time() - start, 1),
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0}
    except Exception as exc:
        return {"status": "error", "verified": False, "steps": 0, "duration": round(time.time() - start, 1),
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0,
                "error": str(exc)[:80]}


async def _main(model: str = "models/gemini-3.6-flash", n_tasks: Optional[int] = None) -> None:
    print(f"\n=== RQ24: Cost per Successful Task ===")
    print(f"Model: {model}\n")

    tasks = TASKS[:n_tasks] if n_tasks else TASKS
    results: list[dict] = []

    for task_def in tasks:
        print(f"  {task_def['id']:35s} ({task_def['horizon']})...", end="", flush=True)
        r = await run_task_with_cost(task_def["task"], model)
        r["id"] = task_def["id"]
        r["horizon"] = task_def["horizon"]
        results.append(r)
        print(f" {r['status']:10s} {'✓' if r['verified'] else '✗'}  "
              f"tok={r['total_tokens']:,}  ${r['cost_usd']:.5f}")
        await asyncio.sleep(INTER_TASK_DELAY)

    # Aggregates
    n = len(results)
    n_success = sum(1 for r in results if r["verified"])
    total_cost = sum(r["cost_usd"] for r in results)
    total_tok  = sum(r["total_tokens"] for r in results)
    c_success  = round(total_cost / n_success, 6) if n_success > 0 else None
    vtcr = n_success / n

    # By horizon
    by_horizon: dict[str, dict] = {}
    for h in ["1_3", "4_7", "8_15", "16_30", "30plus"]:
        subset = [r for r in results if r.get("horizon") == h]
        if not subset: continue
        h_success = sum(1 for r in subset if r["verified"])
        h_cost = sum(r["cost_usd"] for r in subset)
        by_horizon[h] = {
            "n": len(subset), "vtcr": h_success / len(subset),
            "total_cost": round(h_cost, 6),
            "c_success": round(h_cost / h_success, 6) if h_success else None,
        }

    print(f"\n{'─'*60}")
    print(f"Tasks run       : {n}")
    print(f"VTCR            : {vtcr:.0%}  ({n_success}/{n})")
    print(f"Total tokens    : {total_tok:,}")
    print(f"Total cost (USD): ${total_cost:.5f}")
    print(f"C_success       : ${c_success:.5f}" if c_success else "C_success: N/A")
    print(f"\nBy horizon:")
    for h, m in by_horizon.items():
        c = f"${m['c_success']:.5f}" if m["c_success"] else "N/A"
        print(f"  {h:<10} VTCR={m['vtcr']:.0%}  total_cost=${m['total_cost']:.5f}  C_success={c}")

    out = {
        "rq": "RQ24", "model": model,
        "vtcr": vtcr, "n_tasks": n, "n_success": n_success,
        "total_tokens": total_tok, "total_cost_usd": round(total_cost, 6),
        "c_success": c_success,
        "by_horizon": by_horizon, "results": results,
    }
    Path("eval_results_rq24_cost.json").write_text(json.dumps(out, indent=2))
    print("\nResults saved → eval_results_rq24_cost.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gemini-3.6-flash")
    parser.add_argument("--tasks", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(_main(args.model, args.tasks))
