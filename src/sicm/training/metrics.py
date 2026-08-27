"""Binary classification metrics for the SICM evaluation pipeline.

Pure functions — no I/O, no side effects.  Safe to call from training loops,
evaluation scripts, or notebooks.

Public API
----------
metrics_at_threshold(y_true, y_score, threshold)
    Compute sensitivity, specificity, precision, NPV, F1, accuracy, and the
    raw TP/TN/FP/FN counts at a single fixed probability threshold.

sweep_thresholds(y_true, y_score, num_thresholds=201)
    Evaluate metrics_at_threshold at num_thresholds evenly-spaced points in
    [0, 1].  Returns (list_of_dicts, pd.DataFrame).

find_optimal_thresholds(y_true, y_score, num_thresholds=201)
    Run the full threshold sweep and select four clinically meaningful
    operating points:
      - by_roc_youden              max Youden J statistic on the ROC curve
      - by_max_f1                  threshold that maximises F1
      - by_target_sensitivity_95   highest specificity with sensitivity >= 0.95
      - by_target_specificity_90   highest sensitivity with specificity >= 0.90
    Returns a dict with the four thresholds and a "metrics_df" DataFrame.

Example
-------
    import numpy as np
    from sicm.training.metrics import metrics_at_threshold, find_optimal_thresholds

    y_true  = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.8, 0.9])

    # Metrics at a fixed cut-off
    m = metrics_at_threshold(y_true, y_score, threshold=0.5)
    print(m["sensitivity"], m["specificity"])   # 1.0  1.0

    # Find the best operating point automatically
    result = find_optimal_thresholds(y_true, y_score)
    print(result["by_roc_youden"])              # e.g. 0.45
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return 0 instead of raising for undefined ratios."""
    return float(numerator / denominator) if denominator else 0.0


def metrics_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Compute classification metrics at a fixed probability threshold.

    Returns:
        {
            "threshold", "sensitivity", "specificity", "precision", "npv",
            "f1", "accuracy", "tp", "fp", "tn", "fn"
        }
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    # Classify by thresholding cancerous-class (label=1) probabilities.
    y_pred = (y_score >= threshold).astype(np.int64)

    # Binary confusion matrix order with labels=[0,1] is: TN, FP, FN, TP.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Clinical-style metrics plus model-centric F1/accuracy.
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))
    specificity = _safe_divide(tn, tn + fp)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    npv = _safe_divide(tn, tn + fn)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)

    return {
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def sweep_thresholds(
    y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int = 201
) -> tuple[list[dict[str, float | int]], pd.DataFrame]:
    """Sweep thresholds uniformly over [0, 1] and compute metrics per step."""
    if num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2")

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    rows = [metrics_at_threshold(y_true, y_score, float(thr)) for thr in thresholds]
    return rows, pd.DataFrame(rows)


def find_optimal_thresholds(
    y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int = 201
) -> dict[str, Any]:
    """Find candidate thresholds by several criteria and return full sweep metrics.

    Returns:
        {
            "by_roc_youden": float | None,
            "by_max_f1": float | None,
            "by_target_sensitivity_95": float | None,
            "by_target_specificity_90": float | None,
            "metrics_df": pd.DataFrame,
        }
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    # Full table used both for optimization and CSV inspection.
    _, metrics_df = sweep_thresholds(y_true, y_score, num_thresholds=num_thresholds)

    # 1) ROC Youden J = TPR - FPR.
    thr_youden = None
    try:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        # Guard against single-class cases where sklearn can emit NaN rate arrays
        # (with warnings) instead of raising ValueError.
        finite_mask = np.isfinite(roc_thresholds) & np.isfinite(fpr) & np.isfinite(tpr)
        if np.any(finite_mask):
            youden_j = tpr[finite_mask] - fpr[finite_mask]
            if np.any(np.isfinite(youden_j)):
                finite_thresholds = roc_thresholds[finite_mask]
                thr_youden = float(finite_thresholds[int(np.nanargmax(youden_j))])
    except ValueError:
        thr_youden = None

    # 2) Max-F1 operating point from the threshold sweep.
    thr_max_f1 = None
    if not metrics_df.empty:
        max_f1_row = metrics_df.loc[int(metrics_df["f1"].idxmax())]
        thr_max_f1 = float(max_f1_row["threshold"])

    # 3) Meet sensitivity target first, then maximize specificity.
    thr_sens_95 = None
    sens_candidates = metrics_df[metrics_df["sensitivity"] >= 0.95]
    if not sens_candidates.empty:
        best_row = sens_candidates.sort_values(
            by=["specificity", "f1", "threshold"], ascending=[False, False, False]
        ).iloc[0]
        thr_sens_95 = float(best_row["threshold"])

    # 4) Meet specificity target first, then maximize sensitivity.
    thr_spec_90 = None
    spec_candidates = metrics_df[metrics_df["specificity"] >= 0.90]
    if not spec_candidates.empty:
        best_row = spec_candidates.sort_values(
            by=["sensitivity", "f1", "threshold"], ascending=[False, False, True]
        ).iloc[0]
        thr_spec_90 = float(best_row["threshold"])

    return {
        "by_roc_youden": thr_youden,
        "by_max_f1": thr_max_f1,
        "by_target_sensitivity_95": thr_sens_95,
        "by_target_specificity_90": thr_spec_90,
        "metrics_df": metrics_df,
    }
