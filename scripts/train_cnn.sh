#!/usr/bin/env bash
# train_cnn.sh — CNN + CLIP subset of the comparison matrix.
# Covers: resnet18, resnet50, densenet121, clip_vit_base — each condition A
# (frozen backbone) and B (full fine-tune). No condition C: SICM scans have
# no lesion-mask equivalent (see src/sicm/data/dataset.py).
#
# Adapted from the reference irc-classification-project's scripts/train_cnn.sh.
# Two real changes beyond dropping masks:
#   1. Epoch/patience budgets are scaled down for this dataset's ~86-image
#      train split, not the reference's ~1300+. 50/100-epoch budgets tuned
#      for a dataset 10-15x this size will mostly just overfit longer here.
#   2. Condition A's --lr/--weight_decay below are the reference project's
#      own grid-search-tuned values for the SAME architectures — a
#      reasonable starting point since the optimizer/augmentation pipeline
#      is identical, but NOT validated on this dataset. Condition B
#      deliberately omits --lr/--weight_decay so train.py's own per-model
#      defaults apply (see MODEL_TRAINING_CONFIGS in scripts/train.py) —
#      one source of truth rather than duplicating numbers in two places.
# Run scripts/search.py on your own data if you want genuinely tuned values.
#
# Usage:
#   bash scripts/train_cnn.sh

set -euo pipefail

TRAIN="python scripts/train.py"

echo "=========================================================="
echo "  train_cnn.sh — CNN + CLIP training (SICM, no masks)"
echo "=========================================================="

# ── resnet18 ────────────────────────────────────────────────────────────────
echo ""
echo ">>> resnet18 — A: freeze=True  [reference grid-search optimum: lr=1e-3, wd=1e-5, bs=32]"
$TRAIN --model resnet18 --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-3 --weight_decay 1e-5 --batch_size 16 --epochs 30 --patience 8 \
       --eval_test_every_epoch

echo ""
echo ">>> resnet18 — B: freeze=False  [lr/wd from train.py's per-model defaults]"
$TRAIN --model resnet18 --head_type mlp --dropout 0.5 \
       --batch_size 16 --epochs 50 --patience 12 \
       --eval_test_every_epoch

# ── resnet50 ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> resnet50 — A: freeze=True  [reference grid-search optimum: lr=1e-3, wd=1e-4, bs=16]"
$TRAIN --model resnet50 --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-3 --weight_decay 1e-4 --batch_size 16 --epochs 30 --patience 8 \
       --eval_test_every_epoch

echo ""
echo ">>> resnet50 — B: freeze=False  [lr/wd from train.py's per-model defaults]"
$TRAIN --model resnet50 --head_type mlp --dropout 0.5 \
       --batch_size 16 --epochs 50 --patience 12 \
       --eval_test_every_epoch

# ── densenet121 ──────────────────────────────────────────────────────────────
echo ""
echo ">>> densenet121 — A: freeze=True  [reference grid-search optimum: lr=5e-4, wd=1e-4, bs=32]"
$TRAIN --model densenet121 --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 5e-4 --weight_decay 1e-4 --batch_size 16 --epochs 30 --patience 8 \
       --eval_test_every_epoch

echo ""
echo ">>> densenet121 — B: freeze=False  [lr/wd from train.py's per-model defaults]"
$TRAIN --model densenet121 --head_type mlp --dropout 0.5 \
       --batch_size 16 --epochs 50 --patience 12 \
       --eval_test_every_epoch

# ── clip_vit_base ─────────────────────────────────────────────────────────────
echo ""
echo ">>> clip_vit_base — A: freeze=True  [reference grid-search optimum: lr=1e-3, wd=1e-4, bs=16]"
$TRAIN --model clip_vit_base --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-3 --weight_decay 1e-4 --batch_size 16 --epochs 30 --patience 8 \
       --eval_test_every_epoch

echo ""
echo ">>> clip_vit_base — B: freeze=False  [CLIP needs a much lower fine-tuning lr]"
$TRAIN --model clip_vit_base --head_type mlp --dropout 0.5 \
       --lr 1e-5 --batch_size 16 --epochs 50 --patience 12 \
       --eval_test_every_epoch

echo ""
echo "=========================================================="
echo "  train_cnn.sh complete."
echo "=========================================================="
