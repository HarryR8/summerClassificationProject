#!/usr/bin/env bash
# Train the full comparison matrix: 9 models x 2 conditions (A/B).
#
# Condition A — freeze_backbone, 30 epochs, patience 8
# Condition B — full fine-tune,   50 epochs, patience 12
#
# Adapted from the reference irc-classification-project's scripts/train_all.sh.
# No condition C (SICM scans have no lesion-mask equivalent — see
# src/sicm/data/dataset.py), and epoch/patience budgets are scaled down for
# this dataset's ~86-image train split (see train_cnn.sh/train_dino.sh for
# the full rationale on both).
#
# All runs include --eval_test_every_epoch to produce epoch_test_preds.npz
# for scripts/plot_epoch_roc.py — remember that's diagnostic only, never
# a basis for picking "the best epoch" (see train.py's module docstring).
#
# Usage:
#   bash scripts/train_all.sh
#
# On HPC this script is called by scripts/hpc/train_all_hpc.pbs. For a
# faster turnaround than one 9-model job, scripts/hpc/train_cnn_hpc.pbs and
# scripts/hpc/train_dino_hpc.pbs submit the two halves (this script's two
# `bash` calls below) as separate, shorter, parallel PBS jobs instead.

set -euo pipefail

echo "=========================================================="
echo "  train_all.sh — full comparison matrix (SICM)"
echo "=========================================================="

bash scripts/train_cnn.sh
bash scripts/train_dino.sh

echo ""
echo "=== train_all.sh complete ==="
echo "Next: scripts/collect_results.py to aggregate everything in runs/ into results/summary.csv"
