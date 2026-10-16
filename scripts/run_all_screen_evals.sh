#!/usr/bin/env bash
# run_all_screen_evals.sh
#
# Runs all screen-based experiments in sequence after RQ1 is done.
# Each experiment uses the screen; they MUST NOT run in parallel.
#
# Order:
#   1. RQ2/3/4 ablations      (AX grounding, verifier, stuck detection)
#   2. RQ10   HITL safety
#   3. RQ16   Rate-limit robustness
#   4. RQ11/13 Architecture + failure isolation
#   5. RQ19   Cross-modal context sharing
#   6. RQ20   End-to-end workflows
#   7. RQ23   Full architecture ablation (V0–V7, longest)
#   8. RQ25   Reliability monitor (30-min smoke run)
#   9. RQ21   Error taxonomy (offline, reads result files)
#  10. RQ24   Cost tracking
#
# Usage:
#   bash scripts/run_all_screen_evals.sh
#   bash scripts/run_all_screen_evals.sh --dry-run    # print plan only

set -euo pipefail
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

RQ1_RESULT="eval_results_rq1_v5.json"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local label="$1"; shift
    # Check if --out <file> already exists → skip (resume support)
    local out_file=""
    local args=("$@")
    for i in "${!args[@]}"; do
        if [[ "${args[$i]}" == "--out" ]]; then
            out_file="${args[$((i+1))]}"
        fi
    done
    if [[ -n "$out_file" && -f "$out_file" ]]; then
        log "=== $label — SKIP (already done: $out_file) ==="
        return
    fi
    log "=== $label ==="
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  (dry-run) would run: $*"
        return
    fi
    "$@"
    log "$label done"
    sleep 30
}

# ── Wait for RQ1 if still running ────────────────────────────────────────────
if ! ls $RQ1_RESULT 2>/dev/null; then
    log "Waiting for $RQ1_RESULT to appear (RQ1 still running)..."
    until ls $RQ1_RESULT 2>/dev/null; do sleep 60; done
    log "RQ1 done — proceeding with screen evals."
fi

# ── 1. Ablations (RQ2 / RQ3 / RQ4) ──────────────────────────────────────────
NAV_FLAGS="--timeout 300 --inter-task-delay 60 --limit 10"

run "RQ2: AX Grounding Ablation" \
    python3 scripts/eval_nav.py --ax-mode both --out eval_results_rq2_ax_both.json \
    $NAV_FLAGS

sleep 180

run "RQ2b: Screenshot Only" \
    python3 scripts/eval_nav.py --ax-mode screenshot_only --out eval_results_rq2_screenshot_only.json \
    $NAV_FLAGS

sleep 180

run "RQ2c: AX Only" \
    python3 scripts/eval_nav.py --ax-mode ax_only --out eval_results_rq2_ax_only.json \
    $NAV_FLAGS

sleep 180

run "RQ3: Verifier — Gemini Only" \
    python3 scripts/eval_nav.py --verify-mode gemini_only --out eval_results_rq3_gemini_only.json \
    $NAV_FLAGS

sleep 180

run "RQ3b: Verifier — Deterministic Only" \
    python3 scripts/eval_nav.py --verify-mode deterministic_only --out eval_results_rq3_det_only.json \
    $NAV_FLAGS

sleep 180

run "RQ3c: Verifier — None" \
    python3 scripts/eval_nav.py --verify-mode none --out eval_results_rq3_none.json \
    $NAV_FLAGS

sleep 180

run "RQ4: Stuck — Recover" \
    python3 scripts/eval_nav.py --stuck-mode recover --out eval_results_rq4_recover.json \
    $NAV_FLAGS

sleep 180

run "RQ4b: Stuck — None" \
    python3 scripts/eval_nav.py --stuck-mode none --out eval_results_rq4_none.json \
    $NAV_FLAGS

sleep 180

# ── 2. HITL Safety (RQ10) ────────────────────────────────────────────────────
run "RQ10: HITL Safety" python3 scripts/eval_hitl.py

sleep 120

# ── 3. Rate-Limit Robustness (RQ16) ─────────────────────────────────────────
run "RQ16: Rate Limit Robustness" python3 scripts/eval_rate_limits.py

sleep 120

# ── 4. Architecture + Failure Isolation (RQ11/13) ────────────────────────────
run "RQ11/13: Architecture" python3 scripts/eval_architecture.py

sleep 120

# ── 5. Cross-Modal Context (RQ19) ────────────────────────────────────────────
run "RQ19: Cross-Modal Context" python3 scripts/eval_cross_modal.py

sleep 120

# ── 6. End-to-End Workflows (RQ20) ───────────────────────────────────────────
run "RQ20: End-to-End Workflows" python3 scripts/eval_workflows.py

sleep 120

# ── 7. Full Ablation (RQ23) ──────────────────────────────────────────────────
run "RQ23: Full Architecture Ablation" python3 scripts/eval_ablation_full.py --out eval_results_rq23_ablation.json

sleep 120

# ── 8. Reliability Monitor (RQ25) — 30-min smoke ────────────────────────────
run "RQ25: Reliability Monitor (30min)" python3 scripts/eval_reliability.py --duration 1800

sleep 60

# ── 9. Error Taxonomy (RQ21) — offline ──────────────────────────────────────
run "RQ21: Error Taxonomy" \
    python3 scripts/eval_error_taxonomy.py \
    eval_results_rq1_v5.json \
    eval_results_rq2_ax_both.json \
    eval_results_rq2_screenshot_only.json \
    eval_results_rq2_ax_only.json \
    eval_results_rq3_*.json \
    eval_results_rq4_*.json

# ── 10. Cost Tracking (RQ24) ─────────────────────────────────────────────────
run "RQ24: Cost Tracking" python3 scripts/eval_cost.py

log "=== All screen evals complete ==="
log "Results: eval_results_*.json"
