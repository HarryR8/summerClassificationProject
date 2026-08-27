#!/usr/bin/env bash
# train_dino.sh — DINO / ViT foundation-model subset of the comparison matrix.
# Covers: dinov2_base, dinov2_large, dinov3_base, dinov3_large — each
# condition A (frozen backbone) and B (full fine-tune). No condition C
# (no lesion-mask equivalent for SICM scans).
#
# Adapted from the reference irc-classification-project's scripts/train_dino.sh.
# Same two changes as train_cnn.sh: epoch/patience scaled down for this
# dataset's tiny train split, and lr/weight_decay for condition A carried
# over from the reference project's own grid search as a starting point
# (not validated on this data) — see that script's header for the full
# rationale. dinov2_base/dinov3_base fit comfortably on a single GPU;
# dinov2_large/dinov3_large need more VRAM (request gpu_type=A100 or L40S
# if running via scripts/hpc/train_dino_hpc.pbs and you hit OOM on the
# default GPU — see README's HPC section).
#
# Requires an HF_TOKEN with access to the gated DINOv3 checkpoints — see
# scripts/sanity_dataloader.py's --model_key dinov3 error message for setup
# instructions, or the README's HPC section for the HPC-specific version.
#
# Given only ~86 training images, frozen-backbone (condition A) is the
# safer default for these — full fine-tuning a 300M+ parameter ViT on 86
# images is a serious overfitting risk regardless of learning rate. B is
# included for completeness/comparison, but don't be surprised if A
# generalises better on the held-out test sessions.
#
# Usage:
#   bash scripts/train_dino.sh

set -euo pipefail

TRAIN="python scripts/train.py"

echo "=========================================================="
echo "  train_dino.sh — DINO/ViT training (SICM, no masks)"
echo "=========================================================="

# ── dinov2_base ──────────────────────────────────────────────────────────────
echo ""
echo ">>> dinov2_base — A: freeze=True"
$TRAIN --model dinov2_base --freeze_backbone --head_type mlp --dropout 0.3 \
       --batch_size 16 --epochs 30 --patience 8 \
       --eval_test_every_epoch

echo ""
echo ">>> dinov2_base — B: freeze=False  [backbone_lr_scale=0.1: backbone trains 10x slower than head]"
$TRAIN --model dinov2_base --head_type mlp --dropout 0.5 \
       --batch_size 16 --epochs 50 --patience 12 --backbone_lr_scale 0.1 \
       --eval_test_every_epoch

# ── dinov2_large ─────────────────────────────────────────────────────────────
echo ""
echo ">>> dinov2_large — A: freeze=True"
$TRAIN --model dinov2_large --freeze_backbone --head_type mlp --dropout 0.3 \
       --batch_size 8 --epochs 30 --patience 8 \
       --eval_test_every_epoch

echo ""
echo ">>> dinov2_large — B: freeze=False  [backbone_lr_scale=0.1]"
$TRAIN --model dinov2_large --head_type mlp --dropout 0.5 \
       --batch_size 8 --epochs 50 --patience 12 --backbone_lr_scale 0.1 \
       --eval_test_every_epoch

# ── dinov3_base ───────────────────────────────────────────────────────────────
echo ""
echo ">>> dinov3_base — A: freeze=True"
$TRAIN --model dinov3_base --freeze_backbone --head_type mlp --dropout 0.3 \
       --batch_size 16 --epochs 30 --patience 8 \
       --eval_test_every_epoch

echo ""
echo ">>> dinov3_base — B: freeze=False  [backbone_lr_scale=0.1]"
$TRAIN --model dinov3_base --head_type mlp --dropout 0.5 \
       --batch_size 16 --epochs 50 --patience 12 --backbone_lr_scale 0.1 \
       --eval_test_every_epoch

# ── dinov3_large ──────────────────────────────────────────────────────────────
echo ""
echo ">>> dinov3_large — A: freeze=True"
$TRAIN --model dinov3_large --freeze_backbone --head_type mlp --dropout 0.3 \
       --batch_size 8 --epochs 30 --patience 8 \
       --eval_test_every_epoch

echo ""
echo ">>> dinov3_large — B: freeze=False  [backbone_lr_scale=0.1]"
$TRAIN --model dinov3_large --head_type mlp --dropout 0.5 \
       --batch_size 8 --epochs 50 --patience 12 --backbone_lr_scale 0.1 \
       --eval_test_every_epoch

echo ""
echo "=========================================================="
echo "  train_dino.sh complete."
echo "=========================================================="
