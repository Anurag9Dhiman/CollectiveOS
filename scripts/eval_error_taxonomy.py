#!/usr/bin/env python3
"""
RQ21: Error Taxonomy and Root-Cause Analysis

Reads an existing eval JSON result file and classifies each failure into
one of 12 root-cause categories. Also produces a prioritized fix roadmap.

12 failure categories:
  1.  perception         — screenshot capture failed / blank screen
  2.  grounding          — wrong coordinates, element not found in AX tree
  3.  planning           — bad plan, wrong sequence of actions
  4.  action_selection   — chose wrong action type (scroll vs click, etc.)
  5.  execution          — pyautogui raised an error
  6.  verification       — verifier said failed when task was actually done (FN)
  7.  memory_retrieval   — memory search returned wrong facts
  8.  memory_update      — stale fact served after update
  9.  routing            — request sent to wrong agent
  10. connector          — external API call failed or timed out
  11. model_api          — Gemini/UI-TARS rate-limit, quota, or 5xx
  12. hitl               — user rejected the action at HITL gate

Usage:
    python3 scripts/eval_error_taxonomy.py eval_results_rq1_v5.json
    python3 scripts/eval_error_taxonomy.py eval_results_rq1_v5.json eval_results_rq2*.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


CATEGORIES = [
    "perception", "grounding", "planning", "action_selection",
    "execution", "verification", "memory_retrieval", "memory_update",
    "routing", "connector", "model_api", "hitl", "unknown",
]

# Keyword hints for auto-classification when error_cat is missing
_KEYWORD_RULES = [
    ("perception",       ["screenshot", "screen capture", "image capture", "blank"]),
    ("grounding",        ["coordinate", "element not found", "ax tree", "misclick", "wrong click"]),
    ("planning",         ["plan", "wrong sequence", "gemini planning"]),
    ("action_selection", ["wrong action", "should have scrolled", "wrong type"]),
    ("execution",        ["pyautogui", "keyboard error", "mouse error", "typewrite"]),
    ("verification",     ["verif", "false negative", "verify_failed"]),
    ("model_api",        ["timeout", "429", "resource_exhausted", "quota", "rate limit", "500", "503"]),
    ("connector",        ["connector", "api error", "http error", "connection refused"]),
    ("hitl",             ["hitl", "cancelled", "human rejected"]),
    ("routing",          ["wrong agent", "routing error"]),
]


def classify(result: dict) -> str:
    """Return the error category for a failed result."""
    cat = result.get("error_cat", "")
    if cat and cat not in ("", "max_iter", "unknown"):
        return cat
    # Classify by result text
    text = (result.get("result", "") + " " + result.get("status", "")).lower()
    for cat_name, kws in _KEYWORD_RULES:
        if any(kw in text for kw in kws):
            return cat_name
    # max_iter without other clues → likely planning or grounding
    if result.get("status") == "max_iter":
        return "planning"
    return "unknown"


def analyze_file(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[MISSING] {path}")
        return {}
    data = json.loads(p.read_text())
    results = data.get("results", [])
    failures = [r for r in results if not r.get("verified")]

    counts: dict[str, int] = defaultdict(int)
    per_task_type: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    per_horizon: dict[str, dict] = defaultdict(lambda: defaultdict(int))

    for r in failures:
        cat = classify(r)
        counts[cat] += 1
        tags = r.get("tags", [])
        hor = r.get("horizon_group", "unknown")
        per_horizon[hor][cat] += 1
        for tag in tags:
            per_task_type[tag][cat] += 1

    return {
        "file":             p.stem,
        "total":            len(results),
        "failures":         len(failures),
        "vtcr":             data.get("vtcr", 0),
        "categories":       dict(counts),
        "per_horizon":      {k: dict(v) for k, v in per_horizon.items()},
        "per_task_type":    {k: dict(v) for k, v in per_task_type.items()},
    }


def print_report(analyses: list[dict]) -> None:
    print(f"\n{'═'*70}")
    print("  RQ21: Error Taxonomy and Root-Cause Analysis")
    print(f"{'═'*70}")

    # Aggregate
    total_fail = sum(a["failures"] for a in analyses)
    agg: dict[str, int] = defaultdict(int)
    for a in analyses:
        for cat, cnt in a["categories"].items():
            agg[cat] += cnt

    if total_fail == 0:
        print("  No failures found — all tasks verified!")
        return

    print(f"\n  Total failures analyzed: {total_fail}")
    print(f"\n  {'Category':<22} {'Count':>6}  {'%':>6}  {'Fix Priority'}")
    print("  " + "─" * 55)

    # Priority mapping (higher = fix first)
    priority = {
        "grounding": "HIGH — AX element centers, coordinate scaling",
        "planning":  "HIGH — planning prompt, max_iter increase",
        "model_api": "MED  — retry backoff, rate-limit handling",
        "verification": "MED  — FN rate reduction, deterministic rules",
        "action_selection": "MED  — action schema clarity",
        "execution": "LOW  — pyautogui error handling",
        "perception": "LOW  — screencapture stability",
        "connector": "LOW  — timeout + fallback",
        "hitl":      "LOW  — HITL UX improvement",
        "routing":   "LOW  — routing classifier",
        "memory_retrieval": "MED  — retrieval recall improvement",
        "memory_update":    "MED  — update-on-conflict logic",
        "unknown":   "—",
    }

    sorted_cats = sorted(agg.items(), key=lambda x: -x[1])
    for cat, cnt in sorted_cats:
        pct = cnt / total_fail * 100
        pri = priority.get(cat, "—")
        print(f"  {cat:<22} {cnt:>6}  {pct:>5.1f}%  {pri}")

    # Per-horizon breakdown
    if any(a.get("per_horizon") for a in analyses):
        print(f"\n  Failures by horizon group:")
        agg_hor: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for a in analyses:
            for hor, cats in a.get("per_horizon", {}).items():
                for cat, cnt in cats.items():
                    agg_hor[hor][cat] += cnt
        for hor in ["1_3", "4_7", "8_15", "16_30", "30plus", "unknown"]:
            if hor in agg_hor:
                top = sorted(agg_hor[hor].items(), key=lambda x: -x[1])[:3]
                top_str = ", ".join(f"{c}:{n}" for c, n in top)
                print(f"    {hor:<10} {sum(agg_hor[hor].values())} failures  [{top_str}]")

    print(f"\n  Fix roadmap (top 3 categories → highest VTCR gain):")
    for rank, (cat, cnt) in enumerate(sorted_cats[:3], 1):
        pct = cnt / total_fail * 100
        pri = priority.get(cat, "—")
        print(f"    {rank}. {cat} ({pct:.0f}% of failures): {pri}")

    print()


def main(files: list[str]) -> None:
    analyses = [analyze_file(f) for f in files]
    analyses = [a for a in analyses if a]
    print_report(analyses)

    out = {"rq": "RQ21", "analyses": analyses}
    Path("eval_results_rq21_taxonomy.json").write_text(json.dumps(out, indent=2))
    print("Results saved → eval_results_rq21_taxonomy.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Try default RQ1 output
        candidates = sorted(Path(".").glob("eval_results_rq1*.json"))
        if candidates:
            main([str(candidates[-1])])
        else:
            print("Usage: eval_error_taxonomy.py result1.json [result2.json ...]")
            sys.exit(1)
    else:
        main(sys.argv[1:])
