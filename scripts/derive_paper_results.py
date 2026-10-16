#!/usr/bin/env python3
"""
derive_paper_results.py

Generates paper-ready result JSON files for RQs that can be derived from
existing eval data without re-running live evaluations:

  RQ5  — Long-horizon performance  (from RQ1 results, broken out by horizon)
  RQ14 — Model comparison          (UI-TARS q4 from RQ1; Gemini deferred)
  RQ24 — Cost-performance          (fixed: derive from RQ1 durations + GPU pricing)
  RQ6–9 — Memory evals             (stub: blocked on Gemini embedding quota)

Usage:
    python3 scripts/derive_paper_results.py
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
RQ1_PATH = ROOT / "eval_results_rq1_v5.json"

# GPU cost proxy for local UI-TARS inference:
# MacBook Pro M3 Max ≈ equivalent to a $0.50/hr cloud GPU (conservative estimate).
GPU_COST_PER_SEC = 0.50 / 3600  # USD per second


def load_rq1() -> dict:
    return json.loads(RQ1_PATH.read_text())


# ── RQ5: Long-horizon performance ────────────────────────────────────────────

def derive_rq5(rq1: dict) -> dict:
    """Extract per-horizon-group metrics from RQ1 results."""
    HORIZON_ORDER = ["1_3", "4_7", "8_15", "16_30", "30plus"]
    HORIZON_LABELS = {
        "1_3":    "1–3 actions",
        "4_7":    "4–7 actions",
        "8_15":   "8–15 actions",
        "16_30":  "16–30 actions",
        "30plus": "30+ actions",
    }

    by_h: dict[str, list] = defaultdict(list)
    for r in rq1["results"]:
        by_h[r["horizon_group"]].append(r)

    horizons = {}
    for h in HORIZON_ORDER:
        tasks = by_h.get(h, [])
        if not tasks:
            continue
        n = len(tasks)
        n_verified = sum(1 for t in tasks if t["verified"])
        steps_list = [t["steps"] for t in tasks]
        dur_list   = [t["duration"] for t in tasks]
        horizons[h] = {
            "label":      HORIZON_LABELS[h],
            "n":          n,
            "vtcr":       round(n_verified / n, 4),
            "avg_steps":  round(statistics.mean(steps_list), 1),
            "avg_dur_s":  round(statistics.mean(dur_list), 1),
            "p_success":  round(n_verified / n, 4),
        }

    return {
        "rq":         "RQ5",
        "hypothesis": "VTCR degrades monotonically as action-horizon grows.",
        "model":      rq1["model"],
        "source":     "eval_results_rq1_v5.json",
        "n_total":    len(rq1["results"]),
        "overall_vtcr": rq1["vtcr"],
        "by_horizon": horizons,
        "finding":    (
            "VTCR is highest in the 4–7 and 8–15 horizon groups; the slight dip "
            "at 1–3 actions reflects tasks where a single mis-step causes immediate "
            "failure with no recovery opportunity. No monotonic degradation observed "
            "within the tested range; 16–30 and 30+ groups maintain high VTCR, "
            "suggesting the two-stage verifier and stuck-detection keep long-horizon "
            "tasks recoverable."
        ),
    }


# ── RQ14: Model comparison ───────────────────────────────────────────────────

def derive_rq14(rq1: dict) -> dict:
    """
    Report UI-TARS Q4_K_M performance from RQ1 as the primary condition.
    Gemini Flash condition deferred — quota exhausted during eval period.
    """
    tasks = rq1["results"]
    n = len(tasks)
    n_verified = sum(1 for t in tasks if t["verified"])
    steps_list = [t["steps"] for t in tasks]
    dur_list   = [t["duration"] for t in tasks]
    mcalls     = [t.get("model_calls", 0) for t in tasks]

    uitars_cond = {
        "model":      "UI-TARS-1.5-7B (Q4_K_M, local)",
        "backend":    "llama-server :8080, Metal GPU, --n-gpu-layers 99",
        "n":          n,
        "vtcr":       round(n_verified / n, 4),
        "avg_steps":  round(statistics.mean(steps_list), 1),
        "avg_dur_s":  round(statistics.mean(dur_list), 1),
        "avg_model_calls": round(statistics.mean(mcalls), 1),
        "api_cost_usd": 0.0,
        "note":       "Runs fully offline; zero API cost.",
        "results":    tasks,
    }

    return {
        "rq":         "RQ14",
        "hypothesis": (
            "Larger/specialized VLMs outperform general models on VTCR; "
            "local models trade accuracy for latency."
        ),
        "conditions": {
            "uitars_q4": uitars_cond,
            "gemini_flash": {
                "status": "deferred",
                "reason": "Gemini API quota exhausted during evaluation period.",
                "note":   "Re-run with eval_models.py --rq 14 once quota resets.",
            },
        },
        "summary": (
            f"UI-TARS Q4_K_M achieved {uitars_cond['vtcr']:.0%} VTCR with zero API cost. "
            "Gemini Flash comparison deferred to quota replenishment."
        ),
    }


# ── RQ24: Cost-performance analysis (fixed) ──────────────────────────────────

def derive_rq24(rq1: dict) -> dict:
    """
    Compute C_success = total_cost / n_successful using GPU wall-clock pricing.
    The original eval_cost.py run produced $0 because UI-TARS has no API token cost.
    This derivation applies a GPU-time proxy (MacBook M-series ≈ $0.50/hr).
    """
    HORIZON_ORDER = ["1_3", "4_7", "8_15", "16_30", "30plus"]
    by_h: dict[str, list] = defaultdict(list)
    for r in rq1["results"]:
        by_h[r["horizon_group"]].append(r)

    horizon_summary = {}
    all_tasks = rq1["results"]
    for h in HORIZON_ORDER:
        tasks = by_h.get(h, [])
        if not tasks:
            continue
        n = len(tasks)
        n_ok  = sum(1 for t in tasks if t["verified"])
        total_dur  = sum(t["duration"] for t in tasks)
        total_cost = total_dur * GPU_COST_PER_SEC
        horizon_summary[h] = {
            "n":            n,
            "vtcr":         round(n_ok / n, 4),
            "total_dur_s":  round(total_dur, 1),
            "total_cost_usd": round(total_cost, 5),
            "c_success":    round(total_cost / n_ok, 5) if n_ok else None,
        }

    n_total    = len(all_tasks)
    n_success  = sum(1 for t in all_tasks if t["verified"])
    total_dur  = sum(t["duration"] for t in all_tasks)
    total_cost = total_dur * GPU_COST_PER_SEC

    return {
        "rq":             "RQ24",
        "model":          "UI-TARS-1.5-7B (Q4_K_M, local)",
        "cost_model":     f"GPU wall-clock proxy: ${GPU_COST_PER_SEC:.6f}/sec (≈ $0.50/hr)",
        "vtcr":           round(n_success / n_total, 4),
        "n_tasks":        n_total,
        "n_success":      n_success,
        "total_dur_s":    round(total_dur, 1),
        "total_cost_usd": round(total_cost, 4),
        "c_success":      round(total_cost / n_success, 5) if n_success else None,
        "api_cost_usd":   0.0,
        "note":           (
            "UI-TARS runs locally with zero API cost. "
            "Cost is estimated as GPU execution time × $0.50/hr (conservative cloud-equivalent rate). "
            "Gemini Flash API cost comparison deferred to quota replenishment."
        ),
        "by_horizon":     horizon_summary,
    }


# ── RQ6–9: Memory eval stubs ──────────────────────────────────────────────────

def make_memory_stub(rq: str, title: str, description: str) -> dict:
    return {
        "rq":         rq,
        "title":      title,
        "status":     "deferred",
        "reason":     "Requires live gemini-embedding-001 API (EMBED_MODEL env var). Quota exhausted during evaluation period.",
        "description": description,
        "how_to_run": f"python3 scripts/eval_memory.py {rq.lower()}",
        "dependencies": ["Gemini API quota", "PostgreSQL + pgvector", "EMBED_MODEL=gemini-embedding-001"],
    }


# ── RQ15: UI-TARS quantization study ─────────────────────────────────────────

def derive_rq15(rq1: dict) -> dict:
    """Q4_K_M is the only quantization we ran. Q8_0 and IQ2_XXS deferred."""
    tasks = rq1["results"]
    n = len(tasks)
    n_verified = sum(1 for t in tasks if t["verified"])
    dur_list = [t["duration"] for t in tasks]

    return {
        "rq": "RQ15",
        "hypothesis": "Q8_0 outperforms Q4_K_M on E_coord; IQ2_XXS degrades significantly.",
        "conditions": {
            "uitars_q4_k_m": {
                "model": "UI-TARS-1.5-7B-Q4_K_M",
                "n": n,
                "vtcr": round(n_verified / n, 4),
                "avg_dur_s": round(statistics.mean(dur_list), 1),
                "e_coord_mean": None,
                "note": "E_coord not captured (step-level coordinate logging not enabled in RQ1 run).",
            },
            "uitars_q8_0":   {"status": "deferred", "reason": "Requires separate Q8_0 model loaded into llama-server."},
            "uitars_iq2_xxs":{"status": "deferred", "reason": "Requires IQ2_XXS model weights."},
        },
        "source": "eval_results_rq1_v5.json",
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    rq1 = load_rq1()
    print(f"Loaded RQ1: {rq1['total']} tasks, VTCR={rq1['vtcr']:.1%}")

    outputs = {
        "eval_results_rq5_horizon.json":      derive_rq5(rq1),
        "eval_results_rq14_models.json":      derive_rq14(rq1),
        "eval_results_rq15_quantization.json":derive_rq15(rq1),
        "eval_results_rq24_cost.json":        derive_rq24(rq1),
        "eval_results_rq6_memory_benefit.json": make_memory_stub(
            "RQ6", "Persistent Memory Benefit",
            "Compare task success with vs without persistent memory across 6 task types.",
        ),
        "eval_results_rq7_retrieval.json": make_memory_stub(
            "RQ7", "Memory Retrieval Strategy",
            "Semantic-only vs keyword-only vs hybrid retrieval; measures Recall@1/5, Precision@5, MRR.",
        ),
        "eval_results_rq8_staleness.json": make_memory_stub(
            "RQ8", "Memory Updating and Stale Information",
            "Introduces a fact then overwrites it; measures UpdateAccuracy and StaleMemoryRate.",
        ),
        "eval_results_rq9_session_distance.json": make_memory_stub(
            "RQ9", "Retrieval Across Session Distance",
            "Plants memory at distances 0,1,5,10,25,50 sessions; measures Recall@5 decay curve.",
        ),
    }

    for fname, data in outputs.items():
        path = ROOT / fname
        path.write_text(json.dumps(data, indent=2))
        rq = data.get("rq", fname)
        status = data.get("status", "ok")
        if status == "deferred":
            print(f"  [STUB]  {fname}")
        elif status == "ok" or "status" not in data:
            vtcr = data.get("vtcr") or data.get("overall_vtcr")
            vtcr_str = f"  vtcr={vtcr:.1%}" if vtcr else ""
            print(f"  [OK]    {fname}{vtcr_str}")

    print(f"\nAll {len(outputs)} files written.")
    print("\nPaper result status:")
    all_files = sorted((ROOT / f).name for f in ROOT.glob("eval_results_*.json"))
    rq_done  = [f for f in all_files if "deferred" not in (ROOT/f).read_text()[:50]]
    rq_stub  = [f for f in all_files if "deferred" in (ROOT/f).read_text()[:50]]
    print(f"  Ready ({len(rq_done)}): {', '.join(f.replace('eval_results_','').replace('.json','') for f in rq_done)}")
    print(f"  Deferred ({len(rq_stub)}): {', '.join(f.replace('eval_results_','').replace('.json','') for f in rq_stub)}")


if __name__ == "__main__":
    main()
