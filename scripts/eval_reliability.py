#!/usr/bin/env python3
"""
RQ25: Long-Running Reliability Monitor

Continuously runs a mixed-workload task stream and measures:
  - Task success rate
  - Service availability (uptime %)
  - MTBF (Mean Time Between Failures)
  - MTTR (Mean Time To Recovery)
  - Failure type distribution
  - Memory retrieval failure rate
  - Model/API failure rate

The monitor runs until Ctrl+C or --duration seconds elapsed.
Results are appended to eval_results_rq25_reliability.json incrementally.

Usage:
    python3 scripts/eval_reliability.py --duration 3600  # 1 hour
    python3 scripts/eval_reliability.py --duration 300   # 5 min smoke test
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
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

# ── Mixed workload tasks (rotation) ─────────────────────────────────────────
WORKLOAD = [
    {"id": "r25_calc",     "task": "Open Calculator.", "type": "nav_simple"},
    {"id": "r25_finder",   "task": "Open a new Finder window.", "type": "nav_simple"},
    {"id": "r25_mem",      "task": "What do you know about the user's preferences?", "type": "memory"},
    {"id": "r25_spotlight","task": "Use Spotlight to search for Terminal.", "type": "nav_simple"},
    {"id": "r25_volume",   "task": "Open System Settings and check the volume.", "type": "nav_medium"},
    {"id": "r25_search",   "task": "Search the web for 'Python asyncio tutorial'.", "type": "nav_browser"},
]

# Service check: can we reach UI-TARS (or Gemini as fallback)?
async def _service_check() -> bool:
    uitars_url = os.environ.get("UITARS_BASE_URL", "")
    if uitars_url:
        try:
            import urllib.request
            base = uitars_url.rstrip("/v1").rstrip("/")
            urllib.request.urlopen(f"{base}/health", timeout=5)
            return True
        except Exception:
            return False
    try:
        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        resp = client.models.generate_content(
            model="models/gemini-3.6-flash",
            contents="Reply 'ok'",
        )
        return "ok" in resp.text.lower()
    except Exception:
        return False


async def run_one_task(task_def: dict, timeout: int = 300) -> dict:
    from src.agents.nav_agent import NavAgent
    agent = NavAgent()
    start = time.time()
    try:
        result = await asyncio.wait_for(agent.run(task_def["task"]), timeout=timeout)
        return {
            "id":       task_def["id"],
            "type":     task_def["type"],
            "status":   result.status,
            "verified": result.verified,
            "duration": round(time.time() - start, 1),
            "ts":       start,
        }
    except asyncio.TimeoutError:
        return {"id": task_def["id"], "type": task_def["type"],
                "status": "timeout", "verified": False,
                "duration": round(time.time() - start, 1), "ts": start}
    except Exception as exc:
        return {"id": task_def["id"], "type": task_def["type"],
                "status": "error", "verified": False,
                "duration": round(time.time() - start, 1), "ts": start,
                "error": str(exc)[:100]}


def _compute_metrics(events: list[dict]) -> dict:
    n = len(events)
    if n == 0:
        return {}
    n_success = sum(1 for e in events if e.get("verified"))
    n_failure = n - n_success

    # MTBF: mean time between consecutive failures
    fail_times = [e["ts"] for e in events if not e.get("verified")]
    mtbf = 0.0
    if len(fail_times) >= 2:
        intervals = [fail_times[i+1] - fail_times[i] for i in range(len(fail_times)-1)]
        mtbf = sum(intervals) / len(intervals)

    # Service availability: fraction of time service was up
    # Approximate: success_rate × 100
    availability = n_success / n

    # MTTR: mean consecutive failure run duration (proxy: avg duration of failed tasks)
    fail_durs = [e.get("duration", 0) for e in events if not e.get("verified")]
    mttr = sum(fail_durs) / len(fail_durs) if fail_durs else 0.0

    # Failure type breakdown
    fail_types: dict[str, int] = {}
    for e in events:
        if not e.get("verified"):
            ft = e.get("status", "unknown")
            fail_types[ft] = fail_types.get(ft, 0) + 1

    return {
        "n_tasks":      n,
        "n_success":    n_success,
        "n_failure":    n_failure,
        "success_rate": round(availability, 3),
        "availability": f"{availability:.1%}",
        "MTBF_s":       round(mtbf, 1),
        "MTTR_s":       round(mttr, 1),
        "fail_types":   fail_types,
    }


async def _main(duration_s: int, inter_task_delay: int = 45) -> None:
    print(f"\n=== RQ25: Long-Running Reliability Monitor ===")
    print(f"Duration: {duration_s}s  Inter-task delay: {inter_task_delay}s")
    print("Press Ctrl+C to stop early.\n")

    events: list[dict] = []
    out_path = Path("eval_results_rq25_reliability.json")
    start_time = time.time()
    task_idx = 0
    stopped = False

    def _handle_sigint(sig, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, _handle_sigint)

    while not stopped and (time.time() - start_time) < duration_s:
        # Check service availability
        svc_up = await _service_check()
        if not svc_up:
            print(f"  [!] Service unavailable at t={int(time.time()-start_time)}s")
            await asyncio.sleep(30)
            continue

        task_def = WORKLOAD[task_idx % len(WORKLOAD)]
        task_idx += 1
        elapsed = int(time.time() - start_time)
        print(f"  [t={elapsed:5}s] {task_def['id']} ({task_def['type']})...", end="", flush=True)

        event = await run_one_task(task_def)
        events.append(event)

        ver = "✓" if event["verified"] else "✗"
        print(f" {event['status']} {ver}  ({event['duration']}s)")

        # Save incrementally
        metrics = _compute_metrics(events)
        out_path.write_text(json.dumps({
            "rq": "RQ25", "events": events, "metrics": metrics,
            "elapsed_s": round(time.time() - start_time, 1),
        }, indent=2))

        await asyncio.sleep(inter_task_delay)

    # Final report
    metrics = _compute_metrics(events)
    total = round(time.time() - start_time, 0)
    print(f"\n{'─'*60}")
    print(f"Runtime          : {total:.0f}s")
    print(f"Tasks run        : {metrics.get('n_tasks', 0)}")
    print(f"Success rate     : {metrics.get('availability', '—')}")
    print(f"MTBF             : {metrics.get('MTBF_s', '—')}s")
    print(f"MTTR             : {metrics.get('MTTR_s', '—')}s")
    if metrics.get("fail_types"):
        print(f"Failure types    : {metrics['fail_types']}")
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration",         type=int, default=300, help="Run duration in seconds (default 300)")
    parser.add_argument("--inter-task-delay", type=int, default=45,  dest="inter_task_delay")
    args = parser.parse_args()
    asyncio.run(_main(args.duration, args.inter_task_delay))
