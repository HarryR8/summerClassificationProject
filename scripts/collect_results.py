"""Aggregate evaluation results from all run directories into structured outputs.

Adapted from the reference irc-classification-project's scripts/collect_results.py.
No structural changes were needed here — this script only reads
config.json / eval_test.json / history.json / threshold-sweep CSVs, none of
which reference val, so it works unchanged for the two-way (train/test)
split used by this project. The one thing dropped is the reference repo's
POSTER_PREFIXES / --poster_only feature, which filtered to six specific
run names from *their* poster — not meaningful here. --name_prefix below
is a generic replacement if you want an equivalent filter later.

Iterates runs/ subdirectories that have both config.json and eval_test.json,
parses config/eval/history/threshold-sweep data, and writes to results/:

  results/summary.csv                     — one row per run, all metrics
  results/roc_curves/<run_name>.npz        — fpr/tpr/thresholds arrays
  results/confusion_matrices/<run_name>.json — raw TP/FP/TN/FN
  results/training_curves/<run_name>.json  — full history.json contents

Usage
-----
    uv run python scripts/collect_results.py
    uv run python scripts/collect_results.py --name_prefix resnet18_
    uv run python scripts/collect_results.py --runs runs/resnet18_* runs/densenet121_*
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _nan(val):
    """Return None/NaN values as float nan, leave valid floats as-is."""
    if val is None:
        return float("nan")
    try:
        v = float(val)
        return v
    except (TypeError, ValueError):
        return float("nan")


def parse_config(config: dict) -> dict:
    """Extract flat config fields relevant for the summary CSV."""
    masks_dir = config.get("masks_dir")
    masks_trained = bool(masks_dir) and masks_dir not in ("", "null", "None")
    return {
        "model": config.get("model", ""),
        "lr": _nan(config.get("lr")),
        "weight_decay": _nan(config.get("weight_decay")),
        "batch_size": config.get("batch_size"),
        "freeze_backbone": config.get("freeze_backbone", False),
        "head_type": config.get("head_type", ""),
        "dropout": _nan(config.get("dropout")),
        "masks_trained": masks_trained,
        "epochs_config": config.get("epochs"),
        "param_total": config.get("param_total"),
        "param_trainable": config.get("param_trainable"),
    }


def parse_eval(eval_data: dict) -> dict:
    """Extract flat eval fields; handles both basic and extended formats."""
    row: dict = {
        "auc_roc": _nan(eval_data.get("auc_roc")),
        "accuracy": _nan(eval_data.get("accuracy")),
        "sensitivity": _nan(eval_data.get("sensitivity")),
        "specificity": _nan(eval_data.get("specificity")),
        "precision": _nan(eval_data.get("precision")),
        "npv": _nan(eval_data.get("npv")),
        "f1": _nan(eval_data.get("f1")),
        "n_samples": eval_data.get("n_samples"),
        "checkpoint_epoch": eval_data.get("checkpoint_epoch"),
    }

    # Confusion matrix at threshold=0.5
    cm = eval_data.get("confusion_matrix", {})
    row["cm_tn"] = cm.get("tn")
    row["cm_fp"] = cm.get("fp")
    row["cm_fn"] = cm.get("fn")
    row["cm_tp"] = cm.get("tp")

    # Optimal thresholds — extended format only
    opt = eval_data.get("optimal_thresholds", {})
    for key, short in [
        ("by_roc_youden", "youden"),
        ("by_max_f1", "maxf1"),
        ("by_target_sensitivity_95", "sens95"),
        ("by_target_specificity_90", "spec90"),
    ]:
        entry = opt.get(key, {}) or {}
        thr = entry.get("threshold")
        m = entry.get("metrics") or {}
        row[f"{short}_threshold"] = _nan(thr)
        row[f"{short}_sens"] = _nan(m.get("sensitivity"))
        row[f"{short}_spec"] = _nan(m.get("specificity"))
        row[f"{short}_f1"] = _nan(m.get("f1"))

    return row


def load_roc_from_sweep(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Reconstruct FPR/TPR/threshold arrays from a threshold-sweep CSV."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        warnings.warn(f"Could not read {csv_path}: {e}")
        return None

    required = {"tp", "fp", "tn", "fn", "threshold"}
    if not required.issubset(df.columns):
        warnings.warn(f"{csv_path} missing columns {required - set(df.columns)}")
        return None

    # fpr = fp/(fp+tn),  tpr = tp/(tp+fn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fpr = np.where(
            (df["fp"] + df["tn"]) > 0,
            df["fp"] / (df["fp"] + df["tn"]),
            0.0,
        )
        tpr = np.where(
            (df["tp"] + df["fn"]) > 0,
            df["tp"] / (df["tp"] + df["fn"]),
            0.0,
        )
    thresholds = df["threshold"].values

    return fpr.astype(np.float64), tpr.astype(np.float64), thresholds.astype(np.float64)


def process_run(run_dir: Path, results_dir: Path) -> tuple[dict | None, str | None]:
    """Process a single run directory.

    Returns (row_dict, None) on success, (None, reason_str) on failure.
    """
    if run_dir.name.startswith("."):
        return None, "hidden dir"

    run_name = run_dir.name
    config_path = run_dir / "config.json"

    # Resolve eval JSON — prefer eval_test.json, fall back to ensemble_test.json
    eval_path = run_dir / "eval_test.json"
    sweep_csv_name = "eval_test_threshold_sweep.csv"
    is_ensemble = False
    if not eval_path.exists():
        ensemble_path = run_dir / "ensemble_test.json"
        if ensemble_path.exists():
            eval_path = ensemble_path
            sweep_csv_name = "ensemble_test_threshold_sweep.csv"
            is_ensemble = True
        else:
            return None, "no eval JSON"

    try:
        config = json.load(open(config_path)) if config_path.exists() else {}
        with open(eval_path) as f:
            eval_data = json.load(f)
    except Exception as e:
        warnings.warn(f"[{run_name}] Failed to load JSON: {e}")
        return None, "JSON parse error"

    row = {"run_name": run_name, "run_type": "ensemble" if is_ensemble else "single", "is_ensemble": is_ensemble}
    row.update(parse_config(config))
    row.update(parse_eval(eval_data))

    # ---------- ROC curve from threshold sweep CSV ----------
    sweep_csv = run_dir / sweep_csv_name
    if sweep_csv.exists():
        roc_data = load_roc_from_sweep(sweep_csv)
        if roc_data is not None:
            fpr, tpr, thresholds = roc_data
            roc_out = results_dir / "roc_curves" / f"{run_name}.npz"
            roc_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(roc_out, fpr=fpr, tpr=tpr, thresholds=thresholds)
    else:
        warnings.warn(f"[{run_name}] {sweep_csv_name} missing — skipping ROC npz")

    # ---------- Confusion matrix JSON ----------
    cm = eval_data.get("confusion_matrix")
    if cm:
        cm_out = results_dir / "confusion_matrices" / f"{run_name}.json"
        cm_out.parent.mkdir(parents=True, exist_ok=True)
        with open(cm_out, "w") as f:
            json.dump(cm, f, indent=2)

    # ---------- Training curves JSON ----------
    history_path = run_dir / "history.json"
    if history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)
            tc_out = results_dir / "training_curves" / f"{run_name}.json"
            tc_out.parent.mkdir(parents=True, exist_ok=True)
            with open(tc_out, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            warnings.warn(f"[{run_name}] Failed to load history.json: {e}")

    return row, None


def main():
    parser = argparse.ArgumentParser(description="Aggregate run evaluation results")
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="runs",
        help="Root directory containing run subdirectories (default: runs/)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Output directory for aggregated results (default: results/)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Specific run directories to process (overrides --runs_dir scan)",
    )
    parser.add_argument(
        "--name_prefix",
        type=str,
        default=None,
        help="Restrict to runs whose directory name starts with this prefix",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build candidate run directory list
    if args.runs:
        candidates = [Path(r) for r in args.runs]
    else:
        candidates = sorted(
            [p for p in runs_dir.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )

    # Filter hidden dirs up front (e.g. .ipynb_checkpoints)
    candidates = [p for p in candidates if not p.name.startswith(".")]

    if args.name_prefix:
        candidates = [p for p in candidates if p.name.startswith(args.name_prefix)]

    rows = []
    skipped: dict[str, list[str]] = {}  # reason -> [run_name, ...]

    for run_dir in candidates:
        row, reason = process_run(run_dir, results_dir)
        if row is None:
            skipped.setdefault(reason, []).append(run_dir.name)
        else:
            rows.append(row)

    if not rows:
        print("No valid runs found.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    # Sort by AUC descending (NaN last)
    df = df.sort_values("auc_roc", ascending=False, na_position="last").reset_index(drop=True)

    summary_path = results_dir / "summary.csv"
    df.to_csv(summary_path, index=False)

    # ---------- Print report ----------
    print(f"\n{'='*60}")
    print(f"  collect_results.py — Summary")
    print(f"{'='*60}")
    total_skipped = sum(len(v) for v in skipped.values())
    print(f"  Runs processed : {len(rows)}")
    print(f"  Runs skipped   : {total_skipped}")
    if skipped:
        for reason, names in skipped.items():
            print(f"    {reason} ({len(names)}): {', '.join(names)}")
    print(f"\n  Results written to: {results_dir}/")
    print(f"    summary.csv ({len(df)} rows)")
    print(f"    roc_curves/   ({len(list((results_dir / 'roc_curves').glob('*.npz') if (results_dir / 'roc_curves').exists() else []))} files)")
    print(f"    confusion_matrices/ ({len(list((results_dir / 'confusion_matrices').glob('*.json') if (results_dir / 'confusion_matrices').exists() else []))} files)")
    print(f"    training_curves/ ({len(list((results_dir / 'training_curves').glob('*.json') if (results_dir / 'training_curves').exists() else []))} files)")

    # Top runs by AUC
    auc_col = df[["run_name", "model", "masks_trained", "freeze_backbone", "auc_roc", "is_ensemble"]].dropna(subset=["auc_roc"])
    if not auc_col.empty:
        print(f"\n  Top runs by AUC-ROC:")
        for _, r in auc_col.head(10).iterrows():
            flags = []
            if r.get("is_ensemble"):
                flags.append("ensemble")
            if r.get("masks_trained"):
                flags.append("masked")
            if r.get("freeze_backbone"):
                flags.append("frozen")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"    {r['auc_roc']:.4f}  {r['model']}{flag_str}  ({r['run_name']})")

    # Best by Youden sensitivity
    youden_col = df[["run_name", "model", "youden_sens", "youden_spec", "youden_threshold"]].dropna(subset=["youden_sens"])
    if not youden_col.empty:
        best_youden = youden_col.sort_values("youden_sens", ascending=False).iloc[0]
        print(
            f"\n  Best Youden sensitivity: {best_youden['youden_sens']:.4f} "
            f"(spec={best_youden['youden_spec']:.4f}, "
            f"thr={best_youden['youden_threshold']:.4f}) "
            f"— {best_youden['run_name']}"
        )

    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
