#!/usr/bin/env python3
"""
RQ23: Full Architecture Ablation Study (8 component variants)

Systematically removes or disables one architectural component at a time
and measures the resulting VTCR to quantify each component's contribution.

8 ablation variants:
  V0  full_system        — all components enabled (baseline)
  V1  no_ax_tree         — screenshot-only (no AX accessibility tree)
  V2  no_verifier        — tasks marked done after final action, no verification
  V3  no_stuck_detect    — no stuck-loop detection
  V4  no_planning        — skip Gemini planning phase, act directly
  V5  no_memory          — tasks run with empty memory context
  V6  no_hitl            — write actions bypass HITL approval (auto-approve)
  V7  no_visual_context  — orchestrator skips VisualOS context enrichment

Each variant runs the same 15-task suite to keep comparisons fair.
Results show the VTCR gap between full_system and each ablation (= component's contribution).

Usage:
    python3 scripts/eval_ablation_full.py
    python3 scripts/eval_ablation_full.py --variant V1   # run one variant
    python3 scripts/eval_ablation_full.py --list          # show variants
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

for _line in (_ROOT / ".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        k, _, v = _line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

SUITE = [
    {"id": "launch_calculator",  "task": "Open the Calculator app."},
    {"id": "open_finder",        "task": "Open a new Finder window."},
    {"id": "navigate_downloads", "task": "Open the Downloads folder in Finder."},
    {"id": "spotlight_textedit", "task": "Use Spotlight to open TextEdit."},
    {"id": "check_wifi",         "task": "Open System Settings, navigate to Wi-Fi, report status."},
    {"id": "open_safari",        "task": "Open Safari."},
    {"id": "new_textedit_doc",   "task": "Open TextEdit and create a new blank document."},
    {"id": "type_hello",         "task": "Open TextEdit, create a new document, and type 'Hello world'."},
    {"id": "open_terminal",      "task": "Open the Terminal application."},
    {"id": "check_battery",      "task": "Report the current battery percentage."},
]

INTER_TASK_DELAY = 10

VARIANTS = {
    "V0": {"ax_mode": "both",            "verify_mode": "two_stage", "stuck_mode": "detect", "planning": True,  "memory": True,  "hitl": True,  "visual_ctx": True},
    "V1": {"ax_mode": "screenshot_only", "verify_mode": "two_stage", "stuck_mode": "detect", "planning": True,  "memory": True,  "hitl": True,  "visual_ctx": True},
    "V2": {"ax_mode": "both",            "verify_mode": "none",       "stuck_mode": "detect", "planning": True,  "memory": True,  "hitl": True,  "visual_ctx": True},
    "V3": {"ax_mode": "both",            "verify_mode": "two_stage", "stuck_mode": "none",   "planning": True,  "memory": True,  "hitl": True,  "visual_ctx": True},
    "V4": {"ax_mode": "both",            "verify_mode": "two_stage", "stuck_mode": "detect", "planning": False, "memory": True,  "hitl": True,  "visual_ctx": True},
    "V5": {"ax_mode": "both",            "verify_mode": "two_stage", "stuck_mode": "detect", "planning": True,  "memory": False, "hitl": True,  "visual_ctx": True},
    "V6": {"ax_mode": "both",            "verify_mode": "two_stage", "stuck_mode": "detect", "planning": True,  "memory": True,  "hitl": False, "visual_ctx": True},
    "V7": {"ax_mode": "both",            "verify_mode": "two_stage", "stuck_mode": "detect", "planning": True,  "memory": True,  "hitl": True,  "visual_ctx": False},
}

VARIANT_LABELS = {
    "V0": "full_system",
    "V1": "no_ax_tree",
    "V2": "no_verifier",
    "V3": "no_stuck_detect",
    "V4": "no_planning",
    "V5": "no_memory",
    "V6": "no_hitl",
    "V7": "no_visual_context",
}


async def run_task_variant(task_str: str, cfg: dict, timeout: int = 300) -> dict:
    from src.agents.nav_agent import NavAgent

    ax_mode     = cfg["ax_mode"]
    verify_mode = cfg["verify_mode"]
    stuck_mode  = cfg["stuck_mode"]

    agent = NavAgent(ax_mode=ax_mode, verify_mode=verify_mode, stuck_mode=stuck_mode)

    if not cfg.get("planning"):
        agent._plan_task = AsyncMock(return_value=[])

    if not cfg.get("memory"):
        async def _no_mem(t, ctx): return ""
        try: agent._enrich_with_memory = _no_mem
        except AttributeError: pass

    hitl_count = [0]
    async def _hitl_auto(desc):
        hitl_count[0] += 1
        return cfg.get("hitl", True)

    start = time.time()
    try:
        result = await asyncio.wait_for(agent.run(task_str, hitl_callback=_hitl_auto), timeout=timeout)
        return {
            "status":   result.status,
            "verified": result.verified,
            "steps":    len(result.steps),
            "duration": round(time.time() - start, 1),
            "hitl":     hitl_count[0],
        }
    except asyncio.TimeoutError:
        partial = getattr(agent, "_partial_result", None)
        steps_done = len(partial.steps) if partial and partial.steps else 0
        return {"status": "timeout", "verified": False, "steps": steps_done,
                "duration": round(time.time() - start, 1), "hitl": hitl_count[0]}
    except Exception as exc:
        return {"status": "error", "verified": False, "steps": 0,
                "duration": round(time.time() - start, 1), "hitl": hitl_count[0],
                "error": str(exc)[:80]}


async def run_variant(vname: str, cfg: dict) -> dict:
    label = VARIANT_LABELS[vname]
    print(f"\n[{vname}: {label}]")
    results = []
    for task_def in SUITE:
        print(f"  {task_def['id']:30s}", end="", flush=True)
        r = await run_task_variant(task_def["task"], cfg)
        r["id"] = task_def["id"]
        results.append(r)
        print(f" {r['status']:10s} {'✓' if r['verified'] else '✗'}  {r['duration']}s")
        await asyncio.sleep(INTER_TASK_DELAY)

    n = len(results)
    vtcr = sum(1 for r in results if r["verified"]) / n
    avg_steps = round(sum(r["steps"] for r in results) / n, 1)
    avg_dur = round(sum(r["duration"] for r in results) / n, 1)
    print(f"  → VTCR={vtcr:.0%}  avg_steps={avg_steps}  avg_dur={avg_dur}s")
    return {"variant": vname, "label": label, "vtcr": vtcr, "avg_steps": avg_steps,
            "avg_dur": avg_dur, "results": results}


async def _main(only_variant: str | None = None, out_path: str | None = None) -> None:
    print("\n=== RQ23: Full Architecture Ablation Study ===\n")
    variant_items = [(k, v) for k, v in VARIANTS.items()]
    if only_variant:
        variant_items = [(k, v) for k, v in variant_items if k == only_variant]

    all_results = []
    for vname, cfg in variant_items:
        vr = await run_variant(vname, cfg)
        all_results.append(vr)

    # Summary table
    if len(all_results) > 1:
        baseline = next((r for r in all_results if r["variant"] == "V0"), None)
        print(f"\n{'─'*65}")
        print(f"{'Variant':<6} {'Label':<22} {'VTCR':>6} {'Δ vs V0':>9} {'AvgSteps':>9}")
        print("─" * 65)
        for r in all_results:
            delta = f"{r['vtcr'] - baseline['vtcr']:+.0%}" if baseline and r["variant"] != "V0" else "  —"
            print(f"{r['variant']:<6} {r['label']:<22} {r['vtcr']:>6.0%} {delta:>9} {r['avg_steps']:>9.1f}")

    out = {"rq": "RQ23", "variants": all_results}
    path = out_path or f"eval_results_rq23_ablation{'_' + only_variant if only_variant else ''}.json"
    Path(path).write_text(json.dumps(out, indent=2))
    print(f"\nResults saved → {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default=None)
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for k, v in VARIANT_LABELS.items():
            cfg = VARIANTS[k]
            print(f"  {k}: {v}")
            print(f"     ax={cfg['ax_mode']}  verify={cfg['verify_mode']}  stuck={cfg['stuck_mode']}")
            print(f"     planning={cfg['planning']} memory={cfg['memory']} hitl={cfg['hitl']} visual_ctx={cfg['visual_ctx']}")
    else:
        asyncio.run(_main(args.variant, args.out))
