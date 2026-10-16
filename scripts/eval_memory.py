#!/usr/bin/env python3
"""
Memory Evaluation Harness — RQ6, RQ7, RQ8, RQ9

Evaluates CollectiveOS persistent memory across four dimensions:

  RQ6: Persistent memory benefit
       Compare task success WITH vs WITHOUT memory across 6 task types.

  RQ7: Retrieval strategy comparison
       Semantic-only vs Keyword-only vs Hybrid (current) vs No-retrieval.
       Measures Recall@1, Recall@5, Precision@5, MRR.

  RQ8: Memory updating and stale information
       Introduces a fact then overwrites it; measures UpdateAccuracy + StaleMemoryRate.

  RQ9: Retrieval across session distance
       Plants memory at distances 0, 1, 5, 10, 25, 50 sessions ago.
       Measures Recall@5 decay curve.

Usage:
    python3 scripts/eval_memory.py rq6   # benefit eval
    python3 scripts/eval_memory.py rq7   # retrieval strategy
    python3 scripts/eval_memory.py rq8   # update accuracy
    python3 scripts/eval_memory.py rq9   # session distance decay
    python3 scripts/eval_memory.py all   # all four
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Load .env
_env = _ROOT / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, _, v = _line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from src.memory import (
    save_fact, list_facts, search, delete_fact_by_id
)
from src.db import connect, default_user_id


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clear_memory() -> None:
    """Remove all memory_chunks for the default user — fresh slate for each eval."""
    conn = connect()
    try:
        uid = default_user_id(conn)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_chunks WHERE user_id = %s", (uid,))
        conn.commit()
    finally:
        conn.close()


def _inject_fact(content: str, session_age: int = 0) -> int:
    """Save a fact and optionally backdate it by session_age days. Returns chunk id."""
    import datetime
    save_fact(content)
    conn = connect()
    try:
        uid = default_user_id(conn)
        with conn.cursor() as cur:
            if session_age > 0:
                # Back-date by session_age days (proxy for session distance)
                delta = datetime.timedelta(days=session_age)
                backdated = datetime.datetime.utcnow() - delta
                cur.execute(
                    "UPDATE memory_chunks SET created_at = %s "
                    "WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
                    (backdated, uid),
                )
            cur.execute(
                "SELECT id FROM memory_chunks WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
                (uid,),
            )
            row = cur.fetchone()
        conn.commit()
        return row[0] if row else -1
    finally:
        conn.close()


def _recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Recall@k = |relevant ∩ top-k| / |relevant|"""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def _precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for i in top_k if i in relevant_ids) / k


def _mrr(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def _search_ids(query: str, limit: int = 5, mode: str = "hybrid") -> list[int]:
    """Search and return chunk ids using the specified mode."""
    if mode == "none":
        return []
    conn = connect()
    try:
        uid = default_user_id(conn)
        with conn.cursor() as cur:
            if mode in ("hybrid", "semantic"):
                from src.memory import _embed
                embedding = _embed(query)
                cur.execute(
                    "SELECT id FROM memory_chunks WHERE user_id = %s "
                    "ORDER BY embedding <=> %s::vector LIMIT %s",
                    (uid, embedding, limit),
                )
                vec_ids = [r[0] for r in cur.fetchall()]
            else:
                vec_ids = []

            if mode in ("hybrid", "keyword"):
                like = f"%{query[:120]}%"
                cur.execute(
                    "SELECT id FROM memory_chunks WHERE user_id = %s AND content ILIKE %s "
                    "ORDER BY created_at DESC LIMIT %s",
                    (uid, like, limit),
                )
                kw_ids = [r[0] for r in cur.fetchall()]
            else:
                kw_ids = []

        if mode == "semantic":
            return vec_ids
        if mode == "keyword":
            return kw_ids
        # hybrid: RRF merge
        k = 60
        scores: dict[int, float] = {}
        for rank, cid in enumerate(vec_ids, 1):
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
        for rank, cid in enumerate(kw_ids, 1):
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
        return sorted(scores, key=lambda x: -scores[x])[:limit]
    finally:
        conn.close()


def _gemini_answer(question: str, context: str) -> str:
    """Ask Gemini a question with memory context. Returns answer string."""
    from google import genai
    from google.genai import types as gtypes
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""))
    prompt = f"Context from memory:\n{context}\n\nQuestion: {question}\nAnswer concisely in one sentence."
    try:
        resp = client.models.generate_content(
            model="models/gemini-3.6-flash",
            contents=prompt,
        )
        return resp.text.strip()
    except Exception as e:
        return f"[ERROR: {e}]"


# ═══════════════════════════════════════════════════════════════════════════
# RQ6: Persistent Memory Benefit
# ═══════════════════════════════════════════════════════════════════════════

RQ6_TASKS = [
    # (task_type, memory_fact, query, expected_answer_keywords)
    ("preference",
     "The user prefers dark mode in all applications.",
     "What display mode does the user prefer?",
     ["dark"]),
    ("personal_context",
     "The user's name is Anurag and they work as an AI researcher.",
     "What is the user's profession?",
     ["researcher", "ai", "research"]),
    ("task_retrieval",
     "The user asked to be reminded about the quarterly report deadline on October 15.",
     "What deadline should the user be reminded about?",
     ["quarterly", "october", "report"]),
    ("cross_session",
     "The user mentioned they use Python 3.13 and prefer asyncio over threading.",
     "Which Python concurrency model does the user prefer?",
     ["asyncio"]),
    ("updated_info",
     "The user's preferred IDE is VS Code.",
     "Which IDE does the user prefer?",
     ["vs code", "vscode"]),
    ("conflict",
     "The user's home city is San Francisco.",
     "Where does the user live?",
     ["san francisco"]),
]


def run_rq6() -> dict:
    print("\n=== RQ6: Persistent Memory Benefit ===")
    results = []

    for task_type, fact, query, expected_kws in RQ6_TASKS:
        # ── Without memory ────────────────────────────────────────────────
        _clear_memory()
        t0 = time.time()
        answer_no_mem = _gemini_answer(query, "")
        lat_no = round(time.time() - t0, 2)
        success_no = any(kw in answer_no_mem.lower() for kw in expected_kws)

        time.sleep(8)  # Gemini free tier rate limit

        # ── With memory ───────────────────────────────────────────────────
        _clear_memory()
        _inject_fact(fact)
        time.sleep(2)
        context = search(query)
        t0 = time.time()
        answer_mem = _gemini_answer(query, context)
        lat_mem = round(time.time() - t0, 2)
        success_mem = any(kw in answer_mem.lower() for kw in expected_kws)

        time.sleep(8)

        results.append({
            "task_type":      task_type,
            "fact":           fact[:80],
            "query":          query,
            "success_no_mem": success_no,
            "success_with_mem": success_mem,
            "answer_no_mem":  answer_no_mem[:100],
            "answer_with_mem": answer_mem[:100],
            "latency_no_mem": lat_no,
            "latency_with_mem": lat_mem,
        })
        ver = lambda b: "✓" if b else "✗"
        print(f"  [{task_type}] no-mem:{ver(success_no)} with-mem:{ver(success_mem)}")

    _clear_memory()

    n = len(results)
    vtcr_no  = sum(1 for r in results if r["success_no_mem"]) / n
    vtcr_mem = sum(1 for r in results if r["success_with_mem"]) / n
    print(f"\n  VTCR without memory : {vtcr_no:.0%}  ({sum(1 for r in results if r['success_no_mem'])}/{n})")
    print(f"  VTCR with memory    : {vtcr_mem:.0%}  ({sum(1 for r in results if r['success_with_mem'])}/{n})")

    return {
        "rq": "RQ6",
        "vtcr_no_memory":   vtcr_no,
        "vtcr_with_memory": vtcr_mem,
        "improvement":      vtcr_mem - vtcr_no,
        "results":          results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# RQ7: Retrieval Strategy Comparison
# ═══════════════════════════════════════════════════════════════════════════

RQ7_CORPUS = [
    ("pref_dark_mode",    "User preference: dark mode enabled across all apps."),
    ("pref_language",     "User coding language preference: Python, then Go, then Rust."),
    ("pref_font",         "User's preferred font is Fira Code size 15 in editors."),
    ("name_fact",         "User name is Anurag Dhiman. Works in AI research."),
    ("location",          "User is based in Sunnyvale, California."),
    ("health",            "User has a standing desk and takes walking breaks every hour."),
    ("schedule",          "User's typical work hours are 9am to 6pm Pacific time."),
    ("meeting",           "The Q4 planning meeting is scheduled for November 3rd."),
    ("project",           "Current main project: CollectiveOS personal AI assistant."),
    ("tool",              "User uses Homebrew for macOS package management."),
]

RQ7_QUERIES = [
    # (query, relevant_fact_ids_by_key)
    ("What display theme does the user prefer?",           {"pref_dark_mode"}),
    ("Which programming languages does the user use?",     {"pref_language"}),
    ("What font does the user use in their code editor?",  {"pref_font"}),
    ("What is the user's name and occupation?",            {"name_fact"}),
    ("Where is the user located?",                        {"location"}),
    ("Does the user use any ergonomic setup?",             {"health"}),
    ("When does the user typically work?",                 {"schedule"}),
    ("When is the Q4 meeting?",                           {"meeting"}),
    ("What is the user currently working on?",             {"project"}),
    ("How does the user install software on macOS?",       {"tool"}),
]


def run_rq7() -> dict:
    print("\n=== RQ7: Retrieval Strategy Comparison ===")
    _clear_memory()

    # Inject corpus
    fact_id_map: dict[str, int] = {}
    for key, content in RQ7_CORPUS:
        cid = _inject_fact(content)
        fact_id_map[key] = cid
        time.sleep(1)

    modes = ["hybrid", "semantic", "keyword", "none"]
    mode_metrics: dict[str, dict] = {}

    for mode in modes:
        recall1_list, recall5_list, prec5_list, mrr_list, lat_list = [], [], [], [], []
        for query, relevant_keys in RQ7_QUERIES:
            relevant_ids = {fact_id_map[k] for k in relevant_keys if k in fact_id_map}
            t0 = time.time()
            retrieved = _search_ids(query, limit=5, mode=mode)
            lat_list.append(round(time.time() - t0, 3))
            recall1_list.append(_recall_at_k(retrieved, relevant_ids, 1))
            recall5_list.append(_recall_at_k(retrieved, relevant_ids, 5))
            prec5_list.append(_precision_at_k(retrieved, relevant_ids, 5))
            mrr_list.append(_mrr(retrieved, relevant_ids))

        avg_r1  = round(sum(recall1_list) / len(recall1_list), 3)
        avg_r5  = round(sum(recall5_list) / len(recall5_list), 3)
        avg_p5  = round(sum(prec5_list) / len(prec5_list), 3)
        avg_mrr = round(sum(mrr_list) / len(mrr_list), 3)
        avg_lat = round(sum(lat_list) / len(lat_list), 3)
        mode_metrics[mode] = {
            "Recall@1": avg_r1, "Recall@5": avg_r5,
            "Precision@5": avg_p5, "MRR": avg_mrr,
            "avg_latency_s": avg_lat,
        }
        print(f"  {mode:<12} R@1={avg_r1:.0%} R@5={avg_r5:.0%} P@5={avg_p5:.0%} MRR={avg_mrr:.3f} lat={avg_lat:.3f}s")

    _clear_memory()
    return {"rq": "RQ7", "modes": mode_metrics}


# ═══════════════════════════════════════════════════════════════════════════
# RQ8: Memory Updating and Stale Information
# ═══════════════════════════════════════════════════════════════════════════

RQ8_UPDATE_PAIRS = [
    # (original_fact, updated_fact, query, expected_current_kws, stale_kws)
    ("User's preferred IDE is VS Code.",
     "User switched from VS Code to Cursor (AI-powered IDE).",
     "Which IDE does the user currently use?",
     ["cursor"], ["vs code"]),
    ("User's home city is San Francisco.",
     "User moved from San Francisco to Sunnyvale.",
     "Where does the user currently live?",
     ["sunnyvale"], ["san francisco"]),
    ("User prefers Python 3.10 for new projects.",
     "User has upgraded to Python 3.13 for all new projects.",
     "Which Python version does the user use?",
     ["3.13"], ["3.10"]),
    ("User's work schedule is 9am-5pm.",
     "User changed work schedule to 7am-3pm (earlier shift).",
     "What are the user's current work hours?",
     ["7am", "3pm", "7", "3"], ["9am", "5pm", "9"]),
]


def run_rq8() -> dict:
    print("\n=== RQ8: Memory Updating and Stale Information ===")
    _clear_memory()

    update_correct, update_total = 0, 0
    stale_count = 0

    results = []
    for orig, updated, query, current_kws, stale_kws in RQ8_UPDATE_PAIRS:
        # Plant original fact
        orig_id = _inject_fact(orig)
        time.sleep(2)

        # Check before update
        ctx_before = search(query)
        ans_before = _gemini_answer(query, ctx_before)
        time.sleep(8)

        # Update: add updated fact (simulates new session info)
        upd_id = _inject_fact(updated)
        time.sleep(2)

        # Query after update
        ctx_after = search(query)
        ans_after = _gemini_answer(query, ctx_after)
        time.sleep(8)

        # Check UpdateAccuracy: does the answer reflect the updated fact?
        update_correct_this = any(kw in ans_after.lower() for kw in current_kws)
        # Check StaleMemoryRate: does the answer STILL use the old fact?
        stale_this = any(kw in ans_after.lower() for kw in stale_kws)

        update_correct += int(update_correct_this)
        stale_count += int(stale_this)
        update_total += 1

        results.append({
            "query":           query,
            "original_fact":   orig[:60],
            "updated_fact":    updated[:60],
            "answer_before":   ans_before[:80],
            "answer_after":    ans_after[:80],
            "update_correct":  update_correct_this,
            "is_stale":        stale_this,
        })
        print(f"  [{query[:40]}...] update={'✓' if update_correct_this else '✗'} stale={'⚠' if stale_this else '✓'}")

        _clear_memory()

    ua = round(update_correct / update_total, 3) if update_total else 0
    smr = round(stale_count / update_total, 3) if update_total else 0
    print(f"\n  UpdateAccuracy  : {ua:.0%}  ({update_correct}/{update_total})")
    print(f"  StaleMemoryRate : {smr:.0%}  ({stale_count}/{update_total})")

    return {
        "rq":              "RQ8",
        "UpdateAccuracy":  ua,
        "StaleMemoryRate": smr,
        "n_updates":       update_total,
        "results":         results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# RQ9: Retrieval Across Session Distance
# ═══════════════════════════════════════════════════════════════════════════

# Session distances to test (in proxy days: 0=same, 1=1 day, etc.)
RQ9_DISTANCES = [0, 1, 5, 10, 25, 50]

RQ9_FACTS = [
    ("pref_editor",    "User's preferred text editor is Neovim.",            "What text editor does the user prefer?",         ["neovim"]),
    ("pref_shell",     "User's preferred shell is zsh with Oh My Zsh.",      "Which shell does the user use?",                 ["zsh"]),
    ("pref_theme",     "User's preferred color theme is Dracula.",           "What color theme does the user use?",            ["dracula"]),
]


def run_rq9() -> dict:
    print("\n=== RQ9: Retrieval Across Session Distance ===")

    distance_results: dict[int, dict] = {}

    for distance in RQ9_DISTANCES:
        recall5_list = []
        _clear_memory()

        for key, fact, query, expected_kws in RQ9_FACTS:
            cid = _inject_fact(fact, session_age=distance)
            time.sleep(1)

            # Plant 5 distractor facts from "today" to push the target further
            for i in range(5):
                _inject_fact(f"Distractor fact {i} about unrelated topics like weather and sports.")
                time.sleep(0.5)

            retrieved = _search_ids(query, limit=5, mode="hybrid")
            recall5 = _recall_at_k(retrieved, {cid}, 5)
            recall5_list.append(recall5)

            # Also test semantic quality via Gemini
            ctx = search(query)
            ans = _gemini_answer(query, ctx)
            time.sleep(8)
            grounded = any(kw in ans.lower() for kw in expected_kws)
            print(f"  dist={distance}d [{key}] R@5={recall5:.0%} grounded={'✓' if grounded else '✗'}")
            _clear_memory()

        avg_r5 = round(sum(recall5_list) / len(recall5_list), 3)
        distance_results[distance] = {"Recall@5": avg_r5, "n": len(recall5_list)}

    print(f"\n  Session Distance → Recall@5:")
    for d, m in distance_results.items():
        print(f"    {d:>4} sessions: {m['Recall@5']:.0%}")

    _clear_memory()
    return {"rq": "RQ9", "distance_recall5": distance_results}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(rq: str) -> None:
    all_results = {}
    t0 = time.time()

    if rq in ("rq6", "all"):
        all_results["rq6"] = run_rq6()
    if rq in ("rq7", "all"):
        all_results["rq7"] = run_rq7()
    if rq in ("rq8", "all"):
        all_results["rq8"] = run_rq8()
    if rq in ("rq9", "all"):
        all_results["rq9"] = run_rq9()

    duration = round(time.time() - t0, 1)
    all_results["total_duration_s"] = duration

    out = Path("eval_results_memory.json")
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {out}  ({duration:.0f}s total)")


if __name__ == "__main__":
    rq = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if rq not in ("rq6", "rq7", "rq8", "rq9", "all"):
        print("Usage: eval_memory.py [rq6|rq7|rq8|rq9|all]")
        sys.exit(1)
    main(rq)
