#!/usr/bin/env bash
# =============================================================================
# Ablation runner — RQ2, RQ3, RQ4, RQ5 (horizon), RQ16 (rate limits), RQ17
# Runs each config sequentially; each uses horizon 1_3+4_7 tasks (25 tasks)
# for speed. Results saved to eval_results_rq*.json.
#
# Usage:
#   bash scripts/run_ablations.sh          # run all ablations
#   bash scripts/run_ablations.sh rq2      # run only RQ2
#   bash scripts/run_ablations.sh rq3      # run only RQ3
#   bash scripts/run_ablations.sh rq4      # run only RQ4
#
# Prerequisites: RQ1 must have finished (eval running live on Mac display).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

DELAY=30      # inter-task delay (Gemini free tier)
TIMEOUT=150   # per-task timeout

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── RQ2: Accessibility-Tree Grounding Ablation ─────────────────────────────
run_rq2() {
    log "=== RQ2: AX Grounding Ablation ==="
    for config in both screenshot_only ax_only; do
        log "RQ2 config: ax_mode=$config"
        python3 scripts/eval_nav.py \
            --ax-mode "$config" \
            --inter-task-delay "$DELAY" \
            --timeout "$TIMEOUT" \
            --out "eval_results_rq2_ax_${config}.json" \
            2>&1 | tee "logs/rq2_ax_${config}.log"
        log "RQ2 $config done. Sleeping 60s before next config..."
        sleep 60
    done
    log "=== RQ2 COMPLETE ==="
    python3 scripts/compare_ablation.py rq2 eval_results_rq2_ax_both.json \
        eval_results_rq2_ax_screenshot_only.json eval_results_rq2_ax_ax_only.json \
        2>/dev/null || true
}

# ── RQ3: Verification Strategy Ablation ────────────────────────────────────
run_rq3() {
    log "=== RQ3: Verification Strategy Ablation ==="
    for config in two_stage gemini_only deterministic_only none; do
        log "RQ3 config: verify_mode=$config"
        python3 scripts/eval_nav.py \
            --verify-mode "$config" \
            --inter-task-delay "$DELAY" \
            --timeout "$TIMEOUT" \
            --out "eval_results_rq3_verify_${config}.json" \
            2>&1 | tee "logs/rq3_verify_${config}.log"
        log "RQ3 $config done. Sleeping 60s..."
        sleep 60
    done
    log "=== RQ3 COMPLETE ==="
}

# ── RQ4: Stuck Detection Ablation ──────────────────────────────────────────
run_rq4() {
    log "=== RQ4: Stuck Detection Ablation ==="
    for config in detect recover none; do
        log "RQ4 config: stuck_mode=$config"
        python3 scripts/eval_nav.py \
            --stuck-mode "$config" \
            --inter-task-delay "$DELAY" \
            --timeout "$TIMEOUT" \
            --out "eval_results_rq4_stuck_${config}.json" \
            2>&1 | tee "logs/rq4_stuck_${config}.log"
        log "RQ4 $config done. Sleeping 60s..."
        sleep 60
    done
    log "=== RQ4 COMPLETE ==="
}

# ── RQ5: already covered by RQ1 (horizon breakdown in report) ──────────────
# The RQ1 baseline already records vtcr_by_horizon. RQ5 is the analysis.

# ── Main ────────────────────────────────────────────────────────────────────
mkdir -p logs

FILTER="${1:-all}"

case "$FILTER" in
    rq2)  run_rq2 ;;
    rq3)  run_rq3 ;;
    rq4)  run_rq4 ;;
    all)
        run_rq2
        log "Sleeping 2 min between RQ groups..."
        sleep 120
        run_rq3
        sleep 120
        run_rq4
        log "=== ALL ABLATIONS COMPLETE ==="
        ;;
    *)
        echo "Usage: $0 [rq2|rq3|rq4|all]"
        exit 1
        ;;
esac
