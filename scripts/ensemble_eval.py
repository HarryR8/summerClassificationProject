"""Ensemble evaluation: average softmax probabilities from multiple trained runs.

Adapted from the reference irc-classification-project's scripts/ensemble_eval.py.
See scripts/evaluate.py's docstring for the split/threshold rationale — the
same two-way (train/test, no val) reasoning applies here. `masks_dir` is
dropped entirely (unsupported for SICM data — no lesion-mask equivalent).

Loads each checkpoint, runs inference on the requested split, aligns predictions
by image_id, averages P(cancerous) across models, then reports AUC and threshold
sweep metrics. No re-training required.

Usage
-----
# Two-model ensemble:
uv run python scripts/ensemble_eval.py \\
    --run_dirs runs/resnet18_20260824_143021 runs/densenet121_20260824_150210

# Custom output directory:
uv run python scripts/ensemble_eval.py \\
    --run_dirs runs/A runs/B --output_dir results/ensemble_AB

# Evaluate on train split instead (diagnostic):
uv run python scripts/ensemble_eval.py \\
    --run_dirs runs/A runs/B --split train

CLI arguments
-------------
--run_dirs          One or more run directories (each must contain config.json + best.pt).
--split             Split to evaluate: "train" or "test" (default: "test").
--images_dir        Directory containing raw images (default: data/raw).
--split_file        Path to the splits CSV (default: data/splits/splits.csv).
--output_dir        Where to write ensemble results (default: runs/ensemble_<timestamp>/).
--num_thresholds    Threshold grid points for sweep (default: 201).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent.parent / ".matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from sicm.data.loaders import create_dataloaders
from sicm.models import create_model, get_preprocess_key
from sicm.training import evaluate
from sicm.training.metrics import find_optimal_thresholds, metrics_at_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble evaluation for the SICM classifier")
    parser.add_argument("--run_dirs", nargs="+", required=True,
                        help="Run directories to ensemble (each must have config.json + best.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--images_dir", type=str, default="data/raw")
    parser.add_argument("--split_file", type=str, default="data/splits/splits.csv")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_thresholds", type=int, default=201)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def load_run_probs(run_dir: Path, split: str, images_dir: str, split_file: str,
                   num_workers: int, device: torch.device
                   ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a checkpoint and return (labels, probs, image_ids) for the requested split."""
    config_path = run_dir / "config.json"
    ckpt_path = run_dir / "best.pt"
    for p in (config_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p.name} in {run_dir}")

    with open(config_path) as f:
        config = json.load(f)

    model_name = config["model"]
    preprocess_key = get_preprocess_key(model_name)

    train_loader, test_loader = create_dataloaders(
        split_file=split_file,
        images_dir=images_dir,
        model_key=preprocess_key,
        batch_size=config.get("batch_size", 16),
        num_workers=num_workers,
    )
    loader = train_loader if split == "train" else test_loader

    model = create_model(
        model_name,
        num_classes=2,
        pretrained=False,
        freeze_backbone=config.get("freeze_backbone", False),
        head_type=config.get("head_type", "linear"),
        head_dropout=config.get("dropout", 0.3),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Unweighted — only used internally by evaluate() for a loss stat we don't report here.
    criterion = nn.CrossEntropyLoss()

    results = evaluate(model, loader, criterion, device)
    print(f"  {model_name} ({run_dir.name}): AUC={results['auc']:.4f}  n={len(results['labels'])}")
    return (
        np.asarray(results["labels"], dtype=np.int64),
        np.asarray(results["probs"], dtype=np.float64),
        results["image_ids"],
    )


def _as_rounded_float(value: float) -> float | None:
    value = float(value)
    return None if not np.isfinite(value) else round(value, 6)


def main():
    args = parse_args()

    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("runs", f"ensemble_{timestamp}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Ensembling {len(args.run_dirs)} run(s) on split='{args.split}':")

    # Collect per-model predictions, keyed by image_id for alignment.
    per_image: dict[str, dict] = defaultdict(lambda: {"label": None, "probs": []})

    run_names = []
    individual_aucs = {}
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str).expanduser().resolve()
        run_names.append(run_dir.name)
        labels, probs, image_ids = load_run_probs(
            run_dir, args.split, args.images_dir, args.split_file,
            args.num_workers, device,
        )
        individual_aucs[run_dir.name] = float(roc_auc_score(labels, probs))
        for img_id, label, prob in zip(image_ids, labels.tolist(), probs.tolist()):
            per_image[img_id]["label"] = label
            per_image[img_id]["probs"].append(prob)

    n_models = len(args.run_dirs)
    complete = {k: v for k, v in per_image.items() if len(v["probs"]) == n_models}
    if len(complete) < len(per_image):
        print(f"Warning: {len(per_image) - len(complete)} image(s) not seen by all models — dropped.")

    image_ids_sorted = sorted(complete.keys())
    labels = np.array([complete[k]["label"] for k in image_ids_sorted], dtype=np.int64)
    probs_ensemble = np.array(
        [np.mean(complete[k]["probs"]) for k in image_ids_sorted], dtype=np.float64
    )

    auc = float(roc_auc_score(labels, probs_ensemble))
    fpr, tpr, _ = roc_curve(labels, probs_ensemble)

    threshold_results = find_optimal_thresholds(labels, probs_ensemble, num_thresholds=args.num_thresholds)
    metrics_df = threshold_results["metrics_df"]
    threshold_candidates = {
        "by_roc_youden":           threshold_results["by_roc_youden"],
        "by_max_f1":               threshold_results["by_max_f1"],
        "by_target_sensitivity_95": threshold_results["by_target_sensitivity_95"],
        "by_target_specificity_90": threshold_results["by_target_specificity_90"],
    }
    threshold_metrics = {
        key: (metrics_at_threshold(labels, probs_ensemble, thr) if thr is not None else None)
        for key, thr in threshold_candidates.items()
    }
    baseline = metrics_at_threshold(labels, probs_ensemble, threshold=0.5)

    roc_png = out_dir / f"ensemble_{args.split}_roc_curve.png"
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"Ensemble AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888", lw=1.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title(f"Ensemble ROC ({args.split} split, {n_models} models)")
    ax.grid(True, linestyle="--", lw=0.6, alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(roc_png); plt.close(fig)

    csv_path = out_dir / f"ensemble_{args.split}_threshold_sweep.csv"
    metrics_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 56)
    print(f"  Ensemble evaluation \u2014 {args.split} split  ({n_models} models)")
    print("=" * 56)
    for name, ind_auc in individual_aucs.items():
        print(f"  Individual AUC [{name}]: {ind_auc:.4f}")
    print(f"  Ensemble AUC-ROC : {auc:.4f}")
    print(f"  Accuracy @0.50   : {baseline['accuracy']:.4f}  ({int(baseline['tp']+baseline['tn'])}/{len(labels)})")
    print(f"  Sensitivity @0.50: {baseline['sensitivity']:.4f}")
    print(f"  Specificity @0.50: {baseline['specificity']:.4f}")
    threshold_name_map = {
        "by_roc_youden":            "ROC Youden J",
        "by_max_f1":                "Max F1",
        "by_target_sensitivity_95": "Sensitivity >= 0.95",
        "by_target_specificity_90": "Specificity >= 0.90",
    }
    print("  Optimal thresholds:")
    for key, label in threshold_name_map.items():
        thr = threshold_candidates[key]
        m = threshold_metrics[key]
        if thr is None or m is None:
            print(f"    - {label:<24}: not found")
        else:
            print(f"    - {label:<24}: thr={thr:.6f}  sens={m['sensitivity']:.4f}  spec={m['specificity']:.4f}  f1={m['f1']:.4f}")
    print(f"  ROC curve    : {roc_png}")
    print(f"  Threshold CSV: {csv_path}")
    print("=" * 56)

    optimal_thresholds_output = {}
    for key in threshold_candidates:
        thr = threshold_candidates[key]
        m = threshold_metrics[key]
        if thr is None or m is None:
            optimal_thresholds_output[key] = {"threshold": None, "metrics": None}
        else:
            m_out = {k: int(v) if k in {"tp","fp","tn","fn"} else _as_rounded_float(float(v))
                     for k, v in m.items() if k != "threshold"}
            optimal_thresholds_output[key] = {"threshold": _as_rounded_float(thr), "metrics": m_out}

    output = {
        "split": args.split,
        "n_models": n_models,
        "run_dirs": args.run_dirs,
        "individual_aucs": {k: round(v, 6) for k, v in individual_aucs.items()},
        "auc_roc": _as_rounded_float(auc),
        "accuracy": _as_rounded_float(float(baseline["accuracy"])),
        "sensitivity": _as_rounded_float(float(baseline["sensitivity"])),
        "specificity": _as_rounded_float(float(baseline["specificity"])),
        "f1": _as_rounded_float(float(baseline["f1"])),
        "confusion_matrix": {
            "tn": int(baseline["tn"]), "fp": int(baseline["fp"]),
            "fn": int(baseline["fn"]), "tp": int(baseline["tp"]),
        },
        "optimal_thresholds": optimal_thresholds_output,
        "n_samples": int(len(labels)),
        "roc_curve_png": str(roc_png),
        "threshold_sweep_csv": str(csv_path),
    }
    json_path = out_dir / f"ensemble_{args.split}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
