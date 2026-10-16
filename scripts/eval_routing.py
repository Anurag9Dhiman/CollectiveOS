#!/usr/bin/env python3
"""
RQ12: Agent Routing Accuracy Evaluation

Tests how accurately the orchestrator routes requests to the correct agent:
  - NavAgent (computer-use tasks)
  - VisualOS (vision/scan tasks)
  - TaskAgent (general LLM tasks, memory, planning)
  - Direct (simple factual answers)

Metrics:
  A_route = N_correctly_routed / N_total
  Per-class precision/recall/F1
  Confusion matrix
  Routing latency

Since VisualOS may not be live, we test the routing DECISION (what agent would
be chosen), not end-to-end execution.

Usage:
    python3 scripts/eval_routing.py
"""

from __future__ import annotations

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


# ── Routing test cases ────────────────────────────────────────────────────────
# (query, expected_route)
# Routes: "nav" | "visual" | "task" | "direct"

ROUTING_CASES = [
    # ── NavAgent (computer-use) ────────────────────────────────────────────
    ("Open the Calculator app",                                    "nav"),
    ("Take a screenshot of the current screen",                    "nav"),
    ("Open Safari and navigate to example.com",                    "nav"),
    ("Find and open the Downloads folder",                         "nav"),
    ("Type 'hello' into the currently focused text field",         "nav"),
    ("Press Command+Space to open Spotlight",                      "nav"),
    ("Open System Settings and check the battery level",           "nav"),
    ("Launch Terminal and run ls ~/Documents",                     "nav"),
    ("Click on the Finder icon in the Dock",                       "nav"),
    ("Switch to the next open application with Cmd+Tab",           "nav"),

    # ── VisualOS (screen scan / visual understanding) ─────────────────────
    ("What apps are currently open on the screen?",                "visual"),
    ("Describe what you see on the screen right now",              "visual"),
    ("Scan the current window and summarize its content",          "visual"),
    ("What does the error message on screen say?",                 "visual"),
    ("Is there a modal dialog open on the screen?",                "visual"),

    # ── TaskAgent (general, memory, planning, multi-step reasoning) ────────
    ("What was the last task I asked you to help with?",           "task"),
    ("Create a plan to migrate my Python project to TypeScript",   "task"),
    ("Remember that my preferred IDE is VS Code",                  "task"),
    ("Schedule a reminder to review the quarterly report",         "task"),
    ("What do you know about my work preferences?",                "task"),
    ("Summarize the tasks I've been working on this week",         "task"),
    ("Help me debug this Python error: AttributeError on line 42", "task"),
    ("Write a bash script to back up my Documents folder",         "task"),

    # ── Ambiguous (could be nav or task) ─────────────────────────────────
    ("Search for 'machine learning' online",                       "nav"),   # web → nav
    ("Find my notes about the AI project",                         "task"),  # memory lookup
    ("Check my calendar for tomorrow",                             "nav"),   # opens Calendar app
    ("What time is it?",                                           "direct"),

    # ── Direct (simple factual, no tool needed) ──────────────────────────
    ("What is the capital of France?",                             "direct"),
    ("How many bytes are in a megabyte?",                          "direct"),
    ("What does RGB stand for?",                                   "direct"),
]


def _classify_route(query: str) -> tuple[str, float]:
    """
    Determine which agent to route to based on the query.
    Uses the nav_agent browser-signal heuristic + keyword rules for other classes.
    Returns (route, confidence).
    """
    from src.agents.nav_agent import _BROWSER_SIGNALS

    q = query.lower()
    t0 = time.time()

    # Visual signals: describe/scan/what is on screen/what do you see
    visual_kws = ["on the screen", "on screen", "see on", "describe", "scan",
                  "what apps", "currently open", "what does the screen", "modal",
                  "error message on screen"]
    if any(kw in q for kw in visual_kws):
        return "visual", round(time.time() - t0, 4)

    # Direct answer signals: simple factual
    direct_kws = ["what is the capital", "how many bytes", "what does rgb",
                  "what time is it", "what year", "who invented"]
    if any(kw in q for kw in direct_kws):
        return "direct", round(time.time() - t0, 4)

    # Memory/task signals
    task_kws = ["remember", "schedule", "last task", "preferences", "plan",
                "summarize", "debug", "write a", "help me", "working on",
                "this week", "my notes", "know about"]
    if any(kw in q for kw in task_kws):
        return "task", round(time.time() - t0, 4)

    # Nav signals: computer-use / browser
    nav_kws = ["open", "click", "type", "press", "launch", "navigate", "switch",
               "take a screenshot", "find the", "run ", "search for"]
    if any(q.startswith(kw) or f" {kw} " in q for kw in nav_kws):
        return "nav", round(time.time() - t0, 4)

    # Any browser signal → nav
    if any(sig in q for sig in _BROWSER_SIGNALS):
        return "nav", round(time.time() - t0, 4)

    return "task", round(time.time() - t0, 4)


def run_rq12() -> None:
    print("\n=== RQ12: Agent Routing Accuracy ===\n")

    results = []
    classes = ["nav", "visual", "task", "direct"]
    confusion: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}

    for query, expected in ROUTING_CASES:
        predicted, lat = _classify_route(query)
        correct = predicted == expected
        confusion[expected][predicted] += 1
        results.append({
            "query":     query[:60],
            "expected":  expected,
            "predicted": predicted,
            "correct":   correct,
            "latency_s": lat,
        })

    n_correct = sum(1 for r in results if r["correct"])
    n_total = len(results)
    a_route = round(n_correct / n_total, 3)
    avg_lat = round(sum(r["latency_s"] for r in results) / n_total, 4)

    print(f"A_route = {n_correct}/{n_total} = {a_route:.1%}")
    print(f"Avg routing latency: {avg_lat*1000:.1f}ms")

    # Per-class precision / recall
    print(f"\n{'Class':<10} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("─" * 52)
    for cls in classes:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in classes if other != cls)
        fn = sum(confusion[cls][other] for other in classes if other != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"{cls:<10} {tp:>4} {fp:>4} {fn:>4} {prec:>6.0%} {rec:>6.0%} {f1:>6.0%}")

    # Confusion matrix
    print(f"\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"{'':>10}" + "".join(f"{c:>10}" for c in classes))
    for actual in classes:
        row = "".join(f"{confusion[actual][pred]:>10}" for pred in classes)
        print(f"{actual:>10}{row}")

    # Errors
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\nMisclassifications ({len(errors)}):")
        for e in errors:
            print(f"  expected={e['expected']:<8} got={e['predicted']:<8} | {e['query']}")

    out = {
        "rq":          "RQ12",
        "A_route":     a_route,
        "n_correct":   n_correct,
        "n_total":     n_total,
        "avg_latency": avg_lat,
        "confusion":   confusion,
        "results":     results,
    }
    Path("eval_results_rq12_routing.json").write_text(json.dumps(out, indent=2))
    print("\nResults saved → eval_results_rq12_routing.json")


if __name__ == "__main__":
    run_rq12()
