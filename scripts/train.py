"""
CLI entrypoint for training the SICM cancerous/noncancerous classifier.

Adapted from the reference irc-classification-project's scripts/train.py.
Structural differences, all driven by the two-way (train/test) split — see
src/sicm/data/prepare_data.py's module docstring for why there's no val
split for this dataset:

  1. No val_loader / val_metrics. `create_dataloaders` returns
     (train_loader, test_loader) only.
  2. No "best.pt chosen by best val_auc". Without a held-out val set,
     picking a checkpoint by watching *any* metric computed on data the
     model will later be scored on (i.e. test) is a form of test-set
     leakage — you'd be tuning epoch count to the test set. So this
     script just trains for --epochs and saves the FINAL epoch's weights
     as both last.pt and best.pt (identical) — "best" here means "final",
     kept only so evaluate.py / ensemble_eval.py can keep loading
     `best.pt` unmodified.
  3. --patience now means "stop early if TRAIN loss hasn't improved for
     N epochs" (was: val_auc). This is a compute-saving convergence
     check, not model selection — it never looks at test data.
  4. --eval_test_every_epoch still works exactly as before (writes
     epoch_test_preds.npz for scripts/plot_epoch_roc.py) but is
     explicitly a DIAGNOSTIC — use it to visualise how test AUC evolves
     across training, never to cherry-pick "the best epoch" for a
     reported number. The reported number should come from the final
     epoch's checkpoint.
  5. Loss class weights are computed dynamically from the actual train
     split's class balance (`--class_weight_mode balanced`, the default)
     rather than the reference repo's hardcoded [0.32, 0.68] (BUS-BRA's
     own benign/malignant ratio, meaningless for this dataset). Pass
     `--class_weight_mode none` for unweighted CrossEntropyLoss.

Example usage:
    python scripts/train.py --model resnet18 --epochs 30 --batch_size 16
    python scripts/train.py --model dinov2_base --epochs 30
    python scripts/train.py --model resnet18 --lr 5e-5 --epochs 30  # override default lr

Given the training set here is ~86-105 images (depending on QC-flag
exclusion), --batch_size 32 (the reference default) with drop_last=True
throws away a large fraction of each epoch (e.g. 22/86 images). Strongly
consider --batch_size 8 or 16 instead.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from sicm.data.loaders import create_dataloaders
from sicm.models import create_model, get_preprocess_key, count_parameters
from sicm.training import train_one_epoch, evaluate


# ── Per-model recommended training settings ────────────────────────────────────
# CLI arguments always override these when explicitly provided.
MODEL_TRAINING_CONFIGS = {
    "resnet18":        {"lr": 1e-4, "weight_decay": 1e-5, "warmup_epochs": 0, "freeze_backbone": False},
    "resnet50":        {"lr": 1e-4, "weight_decay": 1e-5, "warmup_epochs": 0, "freeze_backbone": False},
    "efficientnet_b0": {"lr": 1e-4, "weight_decay": 1e-5, "warmup_epochs": 0, "freeze_backbone": False},
    "densenet121":     {"lr": 1e-4, "weight_decay": 1e-5, "warmup_epochs": 0, "freeze_backbone": False},
    "dinov2_base":     {"lr": 1e-5, "weight_decay": 1e-2, "warmup_epochs": 5, "freeze_backbone": True},
    "dinov2_large":    {"lr": 1e-5, "weight_decay": 1e-2, "warmup_epochs": 5, "freeze_backbone": True},
    "dinov3_base":     {"lr": 1e-5, "weight_decay": 1e-2, "warmup_epochs": 5, "freeze_backbone": True},
    "dinov3_large":    {"lr": 1e-5, "weight_decay": 1e-2, "warmup_epochs": 5, "freeze_backbone": True},
    "clip_vit_base":   {"lr": 1e-5, "weight_decay": 1e-2, "warmup_epochs": 5, "freeze_backbone": True},
}
_DEFAULT_CONFIG = {"lr": 1e-4, "weight_decay": 1e-5, "warmup_epochs": 0, "freeze_backbone": False}


def resolve_config(args):
    """Fill None CLI args with per-model defaults. CLI always wins."""
    defaults = MODEL_TRAINING_CONFIGS.get(args.model, _DEFAULT_CONFIG)
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.weight_decay is None:
        args.weight_decay = defaults["weight_decay"]
    if args.freeze_backbone is None:
        args.freeze_backbone = defaults["freeze_backbone"]
    args.warmup_epochs = defaults.get("warmup_epochs", 0)
    return args


def build_scheduler(optimizer, epochs, warmup_epochs):
    """Cosine annealing with optional linear warmup for ViT models."""
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(train_labels: np.ndarray, mode: str, device) -> torch.Tensor | None:
    """Compute CrossEntropyLoss class weights from the actual train split.

    mode="balanced": sklearn-style balanced weighting,
        weight_c = n_samples / (n_classes * count_c)
    mode="none": return None (unweighted loss).
    """
    if mode == "none":
        return None
    counts = np.bincount(train_labels, minlength=2)
    n_samples = len(train_labels)
    weights = n_samples / (2.0 * np.maximum(counts, 1))
    print(f"Class weights (mode={mode}): noncancerous={weights[0]:.4f}  "
          f"cancerous={weights[1]:.4f}  (train counts: {counts.tolist()})")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SICM cancerous/noncancerous classifier")

    parser.add_argument("--model", type=str, default="resnet18",
                        help="Model name from registry")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience on TRAIN loss (0 = disabled). "
                             "No val set exists, so this cannot be val-based — see module docstring.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for all dataloaders (default lowered from the "
                             "reference repo's 32 — this dataset's train split is far smaller)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for AdamW (default: per-model config)")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Weight decay for AdamW (default: per-model config)")
    parser.add_argument("--freeze_backbone", action="store_true", default=False,
                        help="Freeze backbone weights (only train head)")
    parser.add_argument("--head_type", type=str, default="linear",
                        choices=["linear", "mlp", "mlp_deep"],
                        help="Head architecture on top of backbone")
    parser.add_argument("--class_weight_mode", type=str, default="balanced",
                        choices=["balanced", "none"],
                        help="How to weight CrossEntropyLoss classes (default: balanced, "
                             "computed from the train split — NOT the reference repo's "
                             "hardcoded BUS-BRA weights)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs. Defaults to runs/<model>_<timestamp>")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader worker processes")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate for MLP head layers")
    parser.add_argument("--split_file", type=str, default="data/splits/splits.csv",
                        help="Path to splits CSV file")
    parser.add_argument("--images_dir", type=str, default="data/raw",
                        help="Directory containing image files")
    parser.add_argument("--backbone_lr_scale", type=float, default=1.0,
                        help="Scale factor for backbone LR relative to head LR. "
                             "E.g. 0.02 -> backbone gets lr*0.02, head gets lr. "
                             "Only applies to BackboneWithHead models (DINO/CLIP). "
                             "Default 1.0 = uniform LR for all parameters.")
    parser.add_argument("--eval_test_every_epoch", action="store_true", default=False,
                        help="Evaluate on test split after each epoch and save probs to "
                             "epoch_test_preds.npz. DIAGNOSTIC ONLY (e.g. for "
                             "plot_epoch_roc.py) — do not use this to pick 'the best epoch'; "
                             "that would be tuning to the test set. The reported result "
                             "should come from the final-epoch checkpoint.")

    return parser.parse_args()


def main():
    args = parse_args()
    args = resolve_config(args)

    # ── Reproducibility ────────────────────────────────────────────────────────
    set_seed(args.seed)

    # ── Output directory ───────────────────────────────────────────────────────
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("runs", f"{args.model}_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device ─────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Data ───────────────────────────────────────────────────────────────────
    preprocess_key = get_preprocess_key(args.model)
    train_loader, test_loader = create_dataloaders(
        split_file=args.split_file,
        images_dir=args.images_dir,
        model_key=preprocess_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = create_model(
        args.model,
        num_classes=2,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
        head_type=args.head_type,
        head_dropout=args.dropout,
    )
    model = model.to(device)

    param_info = count_parameters(model)
    print(f"Parameters — total: {param_info['total']:,}  |  trainable: {param_info['trainable']:,}")
    print(f"Config     — lr={args.lr}  weight_decay={args.weight_decay}  "
          f"freeze_backbone={args.freeze_backbone}  warmup_epochs={args.warmup_epochs}  "
          f"backbone_lr_scale={args.backbone_lr_scale}")

    # ── Loss, optimiser, scheduler ─────────────────────────────────────────────
    train_labels = train_loader.dataset.df["label"].values
    class_weights = compute_class_weights(train_labels, args.class_weight_mode, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    use_diff_lr = (
        args.backbone_lr_scale != 1.0
        and hasattr(model, "backbone")
        and hasattr(model, "head")
    )
    if use_diff_lr:
        backbone_lr = args.lr * args.backbone_lr_scale
        param_groups = [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad],
             "lr": backbone_lr},
            {"params": [p for p in model.head.parameters() if p.requires_grad],
             "lr": args.lr},
        ]
        print(f"Differential LR  — backbone: {backbone_lr:.2e}  head: {args.lr:.2e}")
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)

    # ── Save config ────────────────────────────────────────────────────────────
    config = vars(args).copy()
    config["device"] = str(device)
    config["param_total"] = param_info["total"]
    config["param_trainable"] = param_info["trainable"]

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Training loop ──────────────────────────────────────────────────────────
    history = []
    best_train_loss = float("inf")
    patience_counter = 0

    # Per-epoch test evaluation accumulators (diagnostic only — see module docstring)
    all_test_probs = []
    all_test_aucs = []
    all_epoch_nums = []
    test_labels_once = None
    test_image_ids_once = None

    final_epoch = 0
    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": round(train_metrics["loss"], 6),
            "train_auc": round(train_metrics["auc"], 6),
        }

        if args.eval_test_every_epoch:
            test_metrics = evaluate(model, test_loader, criterion, device)
            all_test_probs.append(test_metrics["probs"])
            all_test_aucs.append(test_metrics["auc"])
            all_epoch_nums.append(epoch)
            if test_labels_once is None:
                test_labels_once = test_metrics["labels"]
                test_image_ids_once = test_metrics["image_ids"]
            epoch_record["test_auc"] = round(test_metrics["auc"], 6)

        history.append(epoch_record)

        log_line = (
            f"Epoch {epoch:02d}/{args.epochs:02d} | lr={current_lr:.2e} | "
            f"Train loss={train_metrics['loss']:.3f} auc={train_metrics['auc']:.3f}"
        )
        if args.eval_test_every_epoch:
            log_line += f" | [diagnostic] Test auc={test_metrics['auc']:.3f}"
        print(log_line)

        # Early stopping on TRAIN loss only — never looks at test data.
        if train_metrics["loss"] < best_train_loss:
            best_train_loss = train_metrics["loss"]
            patience_counter = 0
        elif args.patience > 0:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (train loss hasn't improved for {args.patience} epochs)")
                break

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # Save the FINAL epoch's weights as both last.pt and best.pt (identical).
    # "best.pt" is not chosen by any val/test metric here — see module docstring.
    checkpoint = {
        "epoch": final_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": history[-1]["train_loss"] if history else None,
        "train_auc": history[-1]["train_auc"] if history else None,
    }
    torch.save(checkpoint, os.path.join(args.output_dir, "last.pt"))
    torch.save(checkpoint, os.path.join(args.output_dir, "best.pt"))

    if args.eval_test_every_epoch and all_test_probs:
        npz_path = os.path.join(args.output_dir, "epoch_test_preds.npz")
        np.savez_compressed(
            npz_path,
            probs=np.array(all_test_probs, dtype=np.float32),
            labels=np.array(test_labels_once, dtype=np.int64),
            aucs=np.array(all_test_aucs, dtype=np.float64),
            epochs=np.array(all_epoch_nums, dtype=np.int32),
            image_ids=np.array(test_image_ids_once, dtype=object),
        )
        print(f"Per-epoch test predictions (diagnostic) saved to: {npz_path}")

    print(f"\nTraining complete. Final epoch = {final_epoch}, "
          f"final train loss = {history[-1]['train_loss'] if history else float('nan'):.4f}, "
          f"final train AUC = {history[-1]['train_auc'] if history else float('nan'):.4f}.")
    print("Run scripts/evaluate.py against this run_dir to get the (only) test-set numbers to report.")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
