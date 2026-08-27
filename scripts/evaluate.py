"""Evaluate a trained SICM classifier on the train or test split.

Adapted from the reference irc-classification-project's scripts/evaluate.py.
Differences, driven by the two-way (train/test) split (see
src/sicm/data/prepare_data.py's module docstring):

  - `--split` chooses "test" (default, the number to report) or "train"
    (diagnostic — how well did it fit the data it saw, useful for
    eyeballing overfitting; never report this as generalisation
    performance).
  - `--threshold_split` chooses which split's predictions are used to pick
    the four "optimal" operating-point thresholds. The reference repo
    defaults this to "val" specifically so thresholds aren't tuned on the
    same data being scored. There is no val here, so the default is
    "train" instead — still a different split from whatever `--split` is
    reporting on (unless you explicitly pass --threshold_split same). Note
    this is a weaker guard than a true held-out val set: thresholds picked
    from train-split predictions may be optimistic, since those are
    predictions on data the model was fit to. Report threshold=0.5 metrics
    as primary; treat the "optimal threshold" numbers as exploratory.
  - No class-weighted loss reconstruction — the CrossEntropyLoss weight
    only affects the `loss` value evaluate() returns internally, which is
    never surfaced in eval_<split>.json (only AUC/accuracy/sensitivity/
    etc., all threshold-based and weight-independent, are). Unweighted
    loss is used here purely so evaluate() has a criterion to call.

Typical usage
-------------
Evaluate the final checkpoint on the test split (default):
    uv run python scripts/evaluate.py --run_dir runs/resnet18_20260824_143021

Evaluate on the train split instead (diagnostic):
    uv run python scripts/evaluate.py --run_dir runs/resnet18_20260824_143021 --split train

Use the test split both for evaluation and for threshold selection
(NOT recommended — tunes thresholds to the numbers you're reporting):
    uv run python scripts/evaluate.py --run_dir runs/resnet18_20260824_143021 \
        --split test --threshold_split same

Override data paths:
    uv run python scripts/evaluate.py --run_dir runs/resnet18_20260824_143021 \
        --images_dir /path/to/data/raw

CLI arguments
-------------
--run_dir           Path to the run directory (must contain config.json and best.pt).
--split             Split to evaluate: "train" or "test" (default: "test").
--images_dir        Directory containing the raw image files (default: data/raw).
--split_file        Path to the splits CSV (default: data/splits/splits.csv).
--threshold_split   Which split is used to select optimal thresholds:
                      "train" — use train predictions (default; still not the
                                report split, but not a clean held-out set either)
                      "test"  — use the same split being evaluated
                      "same"  — alias for the split passed to --split
--num_thresholds    Number of evenly-spaced thresholds to sweep in [0, 1] (default: 201).
--thresholds_csv    Override the output path for the threshold-sweep CSV.

Outputs written to --run_dir
----------------------------
eval_<split>.json                Structured results (AUC, metrics at 0.5, optimal thresholds).
eval_<split>_roc_curve.png       Publication-ready ROC curve at 300 dpi.
eval_<split>_threshold_sweep.csv Full per-threshold metrics table (num_thresholds rows).
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Allow running this script directly while importing from the local src package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Use a writable local Matplotlib config/cache directory in all environments.
MPL_CONFIG_DIR = Path(__file__).resolve().parent.parent / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from sicm.data.loaders import create_dataloaders
from sicm.training.metrics import find_optimal_thresholds, metrics_at_threshold
from sicm.models import create_model, get_preprocess_key
from sicm.training import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a SICM classifier checkpoint")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--images_dir", type=str, default="data/raw")
    parser.add_argument("--split_file", type=str, default="data/splits/splits.csv")
    parser.add_argument(
        "--threshold_split",
        type=str,
        default="train",
        choices=["train", "test", "same"],
        help="Split used to choose optimal thresholds. 'same' uses --split.",
    )
    parser.add_argument("--num_thresholds", type=int, default=201)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers (overrides value stored in config.json)")
    parser.add_argument("--head_type", type=str, default=None,
                        choices=["linear", "mlp", "mlp_deep"],
                        help="Override head architecture from config.json")
    parser.add_argument(
        "--thresholds_csv",
        type=str,
        default=None,
        help="Optional path for threshold sweep CSV. Defaults to <run_dir>/eval_<split>_threshold_sweep.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific .pt checkpoint file (default: best.pt in run_dir)",
    )
    return parser.parse_args()


def resolve_run_artifacts(run_dir_arg: str, checkpoint_arg: str | None = None):
    run_dir = Path(run_dir_arg).expanduser()
    run_dir = (Path.cwd() / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
    config_path = run_dir / "config.json"
    if checkpoint_arg is not None:
        ckpt_path = Path(checkpoint_arg).expanduser().resolve()
    else:
        ckpt_path = run_dir / "best.pt"
    missing = [str(p) for p in (config_path, ckpt_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)}")
    return run_dir, config_path, ckpt_path


def save_roc_curve(path: Path, split: str, fpr: np.ndarray, tpr: np.ndarray, auc: float):
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=300)
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.0, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"ROC Curve ({split} split)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _as_rounded_float(value: float) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, 6)


def _serialize_metrics(metrics: dict[str, float | int]) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {}
    int_fields = {"tp", "fp", "tn", "fn"}
    for key, value in metrics.items():
        if key in int_fields:
            out[key] = int(value)
        else:
            out[key] = _as_rounded_float(float(value))
    return out


def _serialize_threshold_result(
    threshold: float | None, metrics: dict[str, float | int] | None
) -> dict[str, Any]:
    if threshold is None or metrics is None:
        return {"threshold": None, "metrics": None}
    metrics_without_threshold = dict(metrics)
    metrics_without_threshold.pop("threshold", None)
    return {
        "threshold": _as_rounded_float(threshold),
        "metrics": _serialize_metrics(metrics_without_threshold),
    }


def main():
    args = parse_args()
    run_dir, config_path, ckpt_path = resolve_run_artifacts(args.run_dir, args.checkpoint)

    with open(config_path) as f:
        config = json.load(f)

    print(f"Run dir   : {run_dir}")
    print(f"Model     : {config['model']}")
    print(f"Split     : {args.split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")

    preprocess_key = get_preprocess_key(config["model"])
    train_loader, test_loader = create_dataloaders(
        split_file=args.split_file,
        images_dir=args.images_dir,
        model_key=preprocess_key,
        batch_size=config.get("batch_size", 16),
        num_workers=args.num_workers if args.num_workers is not None else config.get("num_workers", 4),
    )
    loader = train_loader if args.split == "train" else test_loader
    threshold_source_split = args.split if args.threshold_split == "same" else args.threshold_split
    threshold_loader = train_loader if threshold_source_split == "train" else test_loader

    model = create_model(
        config["model"],
        num_classes=2,
        pretrained=False,
        freeze_backbone=config.get("freeze_backbone", False),
        head_type=args.head_type if args.head_type is not None else config.get("head_type", "linear"),
        head_dropout=config.get("dropout", 0.3),
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(final training epoch — no val-based checkpoint selection; see train.py)")

    # Unweighted — the `loss` evaluate() returns is not used in the report below.
    criterion = nn.CrossEntropyLoss()

    results = evaluate(model, loader, criterion, device)
    labels = results["labels"]
    probs  = results["probs"]

    preds = (probs >= 0.5).astype(int)
    auc   = roc_auc_score(labels, probs)
    cm    = confusion_matrix(labels, preds, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()
    accuracy    = (tp + tn) / len(labels)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # recall cancerous
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # recall noncancerous

    print("\n" + "=" * 52)
    print(f"  Evaluation - {args.split} split")
    print("=" * 52)
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"  Accuracy    : {accuracy:.4f}  ({int(tp + tn)}/{len(labels)})")
    print(f"  Sensitivity : {sensitivity:.4f}  (cancerous recall,    TP={tp})")
    print(f"  Specificity : {specificity:.4f}  (noncancerous recall, TN={tn})")
    print()
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print("                Pred noncancerous  Pred cancerous")
    print(f"  Act noncancerous   {tn:5d}              {fp:5d}")
    print(f"  Act cancerous      {fn:5d}              {tp:5d}")
    print("=" * 52)

    baseline_metrics = metrics_at_threshold(labels, probs, threshold=0.5)

    if threshold_loader is loader:
        thr_labels, thr_probs = labels, probs
    else:
        thr_results = evaluate(model, threshold_loader, criterion, device)
        thr_labels, thr_probs = thr_results["labels"], thr_results["probs"]

    optimal = find_optimal_thresholds(thr_labels, thr_probs, num_thresholds=args.num_thresholds)
    metrics_df = optimal["metrics_df"]
    threshold_candidates = {k: v for k, v in optimal.items() if k != "metrics_df"}
    threshold_candidate_metrics = {
        k: (metrics_at_threshold(thr_labels, thr_probs, threshold=optimal[k]) if optimal[k] is not None else None)
        for k in threshold_candidates
    }

    thresholds_csv = (
        Path(args.thresholds_csv) if args.thresholds_csv
        else run_dir / f"eval_{args.split}_threshold_sweep.csv"
    )
    metrics_df.to_csv(thresholds_csv, index=False)

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_png = run_dir / f"eval_{args.split}_roc_curve.png"
    save_roc_curve(roc_png, args.split, fpr, tpr, auc)

    best_threshold = threshold_candidates["by_roc_youden"]
    best_threshold_metrics = threshold_candidate_metrics["by_roc_youden"]
    optimal_thresholds_output = {
        key: _serialize_threshold_result(threshold_candidates[key], threshold_candidate_metrics[key])
        for key in threshold_candidates
    }

    output = {
        "split": args.split,
        "threshold_source_split": threshold_source_split,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "auc_roc": _as_rounded_float(auc),
        "accuracy": _as_rounded_float(accuracy),
        "sensitivity": _as_rounded_float(sensitivity),
        "specificity": _as_rounded_float(specificity),
        "precision": _as_rounded_float(float(baseline_metrics["precision"])),
        "npv": _as_rounded_float(float(baseline_metrics["npv"])),
        "f1": _as_rounded_float(float(baseline_metrics["f1"])),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "optimal_thresholds": optimal_thresholds_output,
        "best_threshold": _as_rounded_float(best_threshold) if best_threshold is not None else None,
        "best_threshold_sensitivity": (
            _as_rounded_float(float(best_threshold_metrics["sensitivity"]))
            if best_threshold_metrics is not None
            else None
        ),
        "best_threshold_specificity": (
            _as_rounded_float(float(best_threshold_metrics["specificity"]))
            if best_threshold_metrics is not None
            else None
        ),
        "roc_curve_png": str(roc_png),
        "threshold_sweep_csv": str(thresholds_csv),
        "threshold_grid_points": int(metrics_df.shape[0]),
        "roc_points": int(fpr.shape[0]),
        "n_samples": int(len(labels)),
    }
    out_path = run_dir / f"eval_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
