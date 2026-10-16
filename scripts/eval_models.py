#!/usr/bin/env python3
"""
RQ14 + RQ15: Model Comparison and Quantization Study

RQ14: Gemini vs UI-TARS model comparison
  Runs the same 10-task suite with:
    a) gemini_flash   — default: models/gemini-2.0-flash via Gemini API
    b) uitars_q4      — UI-TARS-7B Q4_K_M via local llama-server (if available)
    c) uitars_q8      — UI-TARS-7B Q8_0 via local llama-server (if available)

RQ15: UI-TARS quantization accuracy study
  Compares coordinate grounding accuracy across quantizations:
    IQ2_XXS, Q4_K_M, Q8_0 — measured by E_coord (normalized coordinate error)

Prerequisites for UI-TARS conditions:
  - llama-server must be running with the appropriate model loaded
  - Set UITARS_SERVER_URL=http://localhost:8080 in .env
  - Download model weights:
      UI-TARS-7B-Q4_K_M.gguf
      UI-TARS-7B-Q8_0.gguf
      UI-TARS-7B-IQ2_XXS.gguf

If llama-server is not running, UI-TARS conditions are skipped and a
WARNING is recorded in the output.

Metrics:
  VTCR            — verified task completion rate
  avg_steps       — average navigation steps
  avg_dur_s       — average wall-clock time per task
  avg_tokens      — average total tokens consumed
  E_coord_mean    — mean normalized coordinate error (lower = better)
  E_coord_p95     — 95th-percentile coordinate error

Usage:
    python3 scripts/eval_models.py
    python3 scripts/eval_models.py --rq 14
    python3 scripts/eval_models.py --rq 15
    python3 scripts/eval_models.py --uitars-url http://localhost:8080
"""

from __future__ import annotations

import asyncio
import json
import math
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

INTER_TASK_DELAY = 30

TASKS = [
    {"id": "launch_calculator",  "task": "Open the Calculator app."},
    {"id": "open_finder",        "task": "Open a new Finder window."},
    {"id": "navigate_downloads", "task": "Open the Downloads folder in Finder."},
    {"id": "spotlight_textedit", "task": "Use Spotlight to open TextEdit."},
    {"id": "check_wifi",         "task": "Open System Settings and report Wi-Fi status."},
    {"id": "open_safari",        "task": "Open Safari."},
    {"id": "type_hello",         "task": "Open TextEdit and type 'Hello model comparison'."},
    {"id": "open_terminal",      "task": "Open the Terminal application."},
    {"id": "spotlight_calc",     "task": "Use Spotlight to calculate 55 multiplied by 9."},
    {"id": "volume_check",       "task": "Open System Settings and report the output volume."},
]


def _check_llama_server(url: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


async def _run_task(task_str: str, model_label: str, uitars_url: Optional[str], timeout: int = 150) -> dict:
    from src.agents.nav_agent import NavAgent

    env_patch = {}
    if model_label.startswith("uitars") and uitars_url:
        env_patch["UITARS_SERVER_URL"] = uitars_url
        env_patch["UITARS_ENABLED"] = "1"
    else:
        env_patch["UITARS_ENABLED"] = "0"

    orig = {k: os.environ.get(k) for k in env_patch}
    for k, v in env_patch.items():
        os.environ[k] = v

    agent = NavAgent()
    start = time.time()
    try:
        result = await asyncio.wait_for(agent.run(task_str), timeout=timeout)
        r = {
            "status":   result.status,
            "verified": result.verified,
            "steps":    len(result.steps),
            "duration": round(time.time() - start, 1),
            "tokens":   getattr(result, "total_tokens", 0),
        }
        # Collect coordinate errors if steps expose them
        errors = []
        for step in getattr(result, "steps", []):
            if hasattr(step, "coord_error"):
                errors.append(step.coord_error)
        if errors:
            r["e_coord_mean"] = round(sum(errors) / len(errors), 4)
            r["e_coord_p95"]  = round(sorted(errors)[int(len(errors) * 0.95)], 4)
        return r
    except asyncio.TimeoutError:
        return {"status": "timeout", "verified": False, "steps": 0,
                "duration": round(time.time() - start, 1), "tokens": 0}
    except Exception as exc:
        return {"status": "error", "verified": False, "steps": 0,
                "duration": round(time.time() - start, 1), "tokens": 0,
                "error": str(exc)[:80]}
    finally:
        for k, v in orig.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


async def run_rq14(uitars_url: Optional[str] = None) -> dict:
    print("\n── RQ14: Gemini vs UI-TARS Comparison ──")
    conditions: list[tuple[str, bool]] = [("gemini_flash", True)]

    if uitars_url and _check_llama_server(uitars_url):
        conditions.append(("uitars_q4", True))
        conditions.append(("uitars_q8", True))
    else:
        print(f"  [WARN] llama-server not reachable at {uitars_url or 'N/A'} — skipping UI-TARS conditions")

    all_cond: dict[str, dict] = {}
    for label, _ in conditions:
        print(f"\n  Condition: {label}")
        results = []
        for task_def in TASKS:
            print(f"    {task_def['id']:30s}", end="", flush=True)
            r = await _run_task(task_def["task"], label, uitars_url)
            r["id"] = task_def["id"]
            results.append(r)
            print(f" {r['status']:10s} {'✓' if r['verified'] else '✗'}  {r['duration']}s")
            await asyncio.sleep(INTER_TASK_DELAY)

        n = len(results)
        vtcr = sum(1 for r in results if r["verified"]) / n
        avg_steps = round(sum(r["steps"] for r in results) / n, 1)
        avg_dur   = round(sum(r["duration"] for r in results) / n, 1)
        avg_tok   = round(sum(r.get("tokens", 0) for r in results) / n, 0)
        all_cond[label] = {
            "vtcr": vtcr, "avg_steps": avg_steps, "avg_dur": avg_dur,
            "avg_tokens": avg_tok, "results": results,
        }
        print(f"  → VTCR={vtcr:.0%}  avg_steps={avg_steps}  avg_dur={avg_dur}s  tokens={avg_tok:.0f}")

    print(f"\n{'─'*65}")
    print(f"{'Condition':<20} {'VTCR':>6} {'Steps':>7} {'Dur(s)':>7} {'Tokens':>9}")
    print("─" * 65)
    for label, m in all_cond.items():
        print(f"{label:<20} {m['vtcr']:>6.0%} {m['avg_steps']:>7.1f} {m['avg_dur']:>7.1f} {m['avg_tokens']:>9.0f}")
    return all_cond


async def run_rq15(uitars_url: Optional[str] = None) -> dict:
    print("\n── RQ15: UI-TARS Quantization Comparison ──")
    quants = ["IQ2_XXS", "Q4_K_M", "Q8_0"]

    if not uitars_url:
        print("  [WARN] --uitars-url not set — skipping RQ15")
        return {"skipped": True, "reason": "UITARS_SERVER_URL not provided"}

    if not _check_llama_server(uitars_url):
        print(f"  [WARN] llama-server not reachable at {uitars_url} — skipping RQ15")
        return {"skipped": True, "reason": f"llama-server not reachable at {uitars_url}"}

    print("  NOTE: Requires llama-server to be restarted with each quantization.")
    print("  This script assumes the active llama-server model is switched externally.")
    print("  Record results manually or automate with a separate shell launcher.\n")

    # For each quant we just run the tasks and collect E_coord if available
    all_quants: dict[str, dict] = {}
    for quant in quants:
        input(f"\n  [ACTION REQUIRED] Load {quant} model into llama-server, then press Enter...")
        print(f"\n  Quant: {quant}")
        results = []
        for task_def in TASKS[:5]:  # shorter suite for quantization comparison
            print(f"    {task_def['id']:30s}", end="", flush=True)
            r = await _run_task(task_def["task"], f"uitars_{quant.lower()}", uitars_url)
            r["id"] = task_def["id"]
            results.append(r)
            print(f" {r['status']:10s} {'✓' if r['verified'] else '✗'}  e_coord={r.get('e_coord_mean','N/A')}")
            await asyncio.sleep(INTER_TASK_DELAY)

        n = len(results)
        vtcr = sum(1 for r in results if r["verified"]) / n
        errors = [r["e_coord_mean"] for r in results if "e_coord_mean" in r]
        e_mean = round(sum(errors) / len(errors), 4) if errors else None
        all_quants[quant] = {"vtcr": vtcr, "e_coord_mean": e_mean, "results": results}
        print(f"  → VTCR={vtcr:.0%}  E_coord_mean={e_mean}")

    return all_quants


async def _main(rq: str = "all", uitars_url: Optional[str] = None) -> None:
    print("\n=== RQ14 + RQ15: Model Comparison and Quantization ===\n")
    if not uitars_url:
        uitars_url = os.environ.get("UITARS_SERVER_URL")
    out: dict = {"rq": "RQ14+RQ15"}

    if rq in ("all", "14"):
        out["rq14"] = await run_rq14(uitars_url)
    if rq in ("all", "15"):
        out["rq15"] = await run_rq15(uitars_url)

    Path("eval_results_rq14_rq15_models.json").write_text(json.dumps(out, indent=2))
    print("\nResults saved → eval_results_rq14_rq15_models.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq", default="all", choices=["all", "14", "15"])
    parser.add_argument("--uitars-url", default=None)
    args = parser.parse_args()
    asyncio.run(_main(args.rq, args.uitars_url))
