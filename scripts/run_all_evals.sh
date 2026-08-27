#!/usr/bin/env bash
# run_all_evals.sh — evaluate every run in runs/ on the test split.
#
# Adapted from the reference irc-classification-project's
# scripts/hpc/run_all_evals.sh. The reference version hardcoded that
# project's own timestamped run directory names (accumulated over months of
# their own training) — meaningless here. This version just walks every
# run in runs/ automatically, which works for any project at any stage: no
# manual bookkeeping of run names to keep in sync as more runs complete.
#
# Usage:
#   bash scripts/run_all_evals.sh
#   bash scripts/run_all_evals.sh --images_dir /path/to/data/raw
#   bash scripts/run_all_evals.sh --split train    # diagnostic, not test

set -euo pipefail

IMAGES_DIR="data/raw"
SPLIT="test"
RUNS_DIR="runs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --images_dir) IMAGES_DIR="$2"; shift 2 ;;
        --split)      SPLIT="$2";      shift 2 ;;
        --runs_dir)   RUNS_DIR="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

EVAL="python scripts/evaluate.py"

echo "=========================================================="
echo "  run_all_evals.sh — SICM $SPLIT-set evaluation"
echo "  images_dir = $IMAGES_DIR"
echo "  runs_dir   = $RUNS_DIR"
echo "=========================================================="

shopt -s nullglob
n_done=0
n_skipped=0
for run_dir in "$RUNS_DIR"/*/; do
    run_dir="${run_dir%/}"
    if [[ ! -f "$run_dir/config.json" || ! -f "$run_dir/best.pt" ]]; then
        continue  # not a training run directory (e.g. runs/search, runs/ensemble_*)
    fi
    echo ""
    echo ">>> $(basename "$run_dir")"
    if $EVAL --run_dir "$run_dir" --split "$SPLIT" --images_dir "$IMAGES_DIR"; then
        n_done=$((n_done + 1))
    else
        echo "    (failed — skipping)"
        n_skipped=$((n_skipped + 1))
    fi
done

echo ""
echo "=========================================================="
echo "  Evaluated $n_done run(s), $n_skipped failed/skipped."
echo "  Next: python scripts/collect_results.py  ->  results/summary.csv"
echo "=========================================================="
