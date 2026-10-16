#!/usr/bin/env python3
"""
Compare ablation results across multiple eval JSON files.

Usage:
    python3 scripts/compare_ablation.py rq2 file1.json file2.json file3.json
    python3 scripts/compare_ablation.py rq3 file1.json ...

Prints a comparison table with VTCR, avg steps, avg duration, and horizon breakdown.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"  [MISSING] {path}")
        return {}
    return json.loads(p.read_text())


def vtcr_str(data: dict) -> str:
    v = data.get("vtcr", 0)
    t = data.get("total", 0)
    n = sum(1 for r in data.get("results", []) if r.get("verified"))
    return f"{n}/{t} = {v:.1%}"


def avg(data: dict, key: str) -> str:
    results = data.get("results", [])
    if not results:
        return "—"
    vals = [r.get(key, 0) for r in results]
    return f"{sum(vals)/len(vals):.1f}"


def print_table(label: str, files: list[str]) -> None:
    datasets = [(f, load(f)) for f in files]
    valid = [(f, d) for f, d in datasets if d]
    if not valid:
        print("No data found.")
        return

    print(f"\n{'═'*80}")
    print(f"  {label} Ablation Comparison")
    print(f"{'═'*80}")

    # Header
    col_w = 22
    headers = ["Config", "VTCR", "Avg Steps", "Avg Duration", "Tokens"]
    print(f"{'Config':<28} {'VTCR':<18} {'Steps':>7} {'Dur(s)':>8} {'Tokens':>8}")
    print(f"{'─'*80}")

    for fpath, data in valid:
        name = Path(fpath).stem
        # Try to extract config name from filename
        # e.g. eval_results_rq2_ax_screenshot_only → screenshot_only
        parts = name.split("_")
        config = "_".join(parts[3:]) if len(parts) > 3 else name

        vtcr = vtcr_str(data)
        steps = avg(data, "steps")
        dur = avg(data, "duration")
        tokens = data.get("total_tokens", 0)
        print(f"{config:<28} {vtcr:<18} {steps:>7} {dur:>8} {tokens:>8}")

    # Horizon breakdown
    has_horizon = any(d.get("vtcr_by_horizon") for _, d in valid)
    if has_horizon:
        print(f"\n  VTCR by horizon group:")
        print(f"  {'Config':<24} {'1_3':>6} {'4_7':>6} {'8_15':>6} {'16_30':>7} {'30plus':>8}")
        print(f"  {'─'*65}")
        for fpath, data in valid:
            parts = Path(fpath).stem.split("_")
            config = "_".join(parts[3:]) if len(parts) > 3 else Path(fpath).stem
            hb = data.get("vtcr_by_horizon", {})
            def h(k):
                v = hb.get(k)
                return f"{v:.0%}" if v is not None else "—"
            print(f"  {config:<24} {h('1_3'):>6} {h('4_7'):>6} {h('8_15'):>6} {h('16_30'):>7} {h('30plus'):>8}")

    # Error breakdown
    print(f"\n  Error categories (failures only):")
    for fpath, data in valid:
        parts = Path(fpath).stem.split("_")
        config = "_".join(parts[3:]) if len(parts) > 3 else Path(fpath).stem
        eb = data.get("error_categories", {})
        if eb:
            cats = ", ".join(f"{k}:{v}" for k, v in list(eb.items())[:5])
            print(f"  {config:<24} {cats}")

    print(f"{'═'*80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: compare_ablation.py <label> file1.json [file2.json ...]")
        sys.exit(1)
    label = sys.argv[1].upper()
    files = sys.argv[2:]
    print_table(label, files)
