"""Tests for sicm.training.metrics and the evaluate.py CLI integration.

Adapted from the reference irc-classification-project's tests/test_evaluate.py.
Groups 1-3 (metrics_at_threshold / sweep_thresholds / find_optimal_thresholds)
are unchanged — they test sicm.training.metrics, which is an unmodified copy
of the reference repo's metrics module. Group 4 (main() integration) is
adapted for the two-way (train/test, no val) split: create_dataloaders is
mocked to return a 2-tuple instead of 3, the val-split integration test is
dropped, and the fake checkpoint no longer carries a val_auc field (this
project's checkpoints store train_loss/train_auc instead — see train.py).
"""
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sicm.training.metrics import (
    find_optimal_thresholds,
    metrics_at_threshold,
    sweep_thresholds,
)

import scripts.evaluate as eval_module


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def balanced_labels_probs():
    """8-sample dataset: well-separated scores, TP=3, TN=2, FP=1, FN=1 at thr=0.5."""
    labels = np.array([0, 0, 0, 1, 1, 1, 1])
    probs  = np.array([0.1, 0.2, 0.6, 0.3, 0.7, 0.8, 0.9])
    return labels, probs


@pytest.fixture
def perfect_labels_probs():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probs  = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    return labels, probs


@pytest.fixture
def dummy_loader():
    """Small DataLoader returning dicts (matches real loader contract)."""
    images = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    image_ids = [f"img_{i:03d}" for i in range(8)]

    class DictDataset(TensorDataset):
        def __init__(self, images, labels, image_ids):
            super().__init__(images, labels)
            self.image_ids = image_ids

        def __getitem__(self, index):
            img, label = super().__getitem__(index)
            return {"image": img, "label": label, "image_id": self.image_ids[index]}

    return DataLoader(DictDataset(images, labels, image_ids), batch_size=4)


@pytest.fixture
def tmp_run_dir(tmp_path):
    config = {
        "model": "resnet18",
        "freeze_backbone": True,
        "head_type": "linear",
        "dropout": 0.3,
        "batch_size": 4,
        "num_workers": 0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "best.pt").write_bytes(b"")  # stub — torch.load is patched in tests
    return tmp_path


# ─── Group 1: metrics_at_threshold ───────────────────────────────────────────

class TestMetricsAtThreshold:

    def test_perfect_classifier(self, perfect_labels_probs):
        labels, probs = perfect_labels_probs
        m = metrics_at_threshold(labels, probs, threshold=0.5)
        assert m["accuracy"]    == pytest.approx(1.0)
        assert m["sensitivity"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)
        assert m["precision"]   == pytest.approx(1.0)
        assert m["f1"]          == pytest.approx(1.0)

    def test_known_confusion_matrix(self, balanced_labels_probs):
        labels, probs = balanced_labels_probs
        # At threshold=0.5: TN=2, FP=1, FN=1, TP=3
        m = metrics_at_threshold(labels, probs, threshold=0.5)
        assert m["tp"] == 3
        assert m["tn"] == 2
        assert m["fp"] == 1
        assert m["fn"] == 1
        assert m["accuracy"]    == pytest.approx(5 / 7)
        assert m["sensitivity"] == pytest.approx(3 / 4)
        assert m["specificity"] == pytest.approx(2 / 3)
        assert m["precision"]   == pytest.approx(3 / 4)
        assert m["f1"]          == pytest.approx(3 / 4)

    def test_threshold_in_output(self, balanced_labels_probs):
        labels, probs = balanced_labels_probs
        m = metrics_at_threshold(labels, probs, threshold=0.7)
        assert m["threshold"] == pytest.approx(0.7)

    def test_confusion_counts_are_int(self, balanced_labels_probs):
        labels, probs = balanced_labels_probs
        m = metrics_at_threshold(labels, probs, threshold=0.5)
        for key in ("tp", "tn", "fp", "fn"):
            assert isinstance(m[key], int)

    def test_all_predicted_negative_zero_division(self):
        # zero_division=0: sensitivity=0, precision=0, f1=0
        labels = np.array([0, 1, 0, 1])
        probs  = np.array([0.1, 0.2, 0.1, 0.2])
        m = metrics_at_threshold(labels, probs, threshold=0.5)
        assert m["sensitivity"] == pytest.approx(0.0)
        assert m["precision"]   == pytest.approx(0.0)
        assert m["f1"]          == pytest.approx(0.0)
        assert m["tp"] == 0

    def test_all_predicted_positive_zero_specificity(self):
        labels = np.array([0, 1, 0, 1])
        probs  = np.array([0.9, 0.9, 0.9, 0.9])
        m = metrics_at_threshold(labels, probs, threshold=0.5)
        assert m["specificity"] == pytest.approx(0.0)
        assert m["tn"] == 0

    def test_single_class_labels_zero_division(self):
        # All noncancerous: no positives → sensitivity/precision/f1 = 0
        labels = np.array([0, 0, 0, 0])
        probs  = np.array([0.1, 0.2, 0.3, 0.4])
        m = metrics_at_threshold(labels, probs, threshold=0.5)
        assert m["sensitivity"] == pytest.approx(0.0)
        assert m["npv"]         == pytest.approx(1.0)  # TN/(TN+FN)=4/4

    def test_npv_calculation(self, balanced_labels_probs):
        labels, probs = balanced_labels_probs
        # TN=2, FN=1 → NPV = 2/3
        m = metrics_at_threshold(labels, probs, threshold=0.5)
        assert m["npv"] == pytest.approx(2 / 3)


# ─── Group 2: sweep_thresholds ───────────────────────────────────────────────

class TestSweepThresholds:

    def test_returns_correct_number_of_rows(self, balanced_labels_probs):
        labels, probs = balanced_labels_probs
        rows, df = sweep_thresholds(labels, probs, num_thresholds=11)
        assert len(rows) == 11
        assert len(df) == 11

    def test_dataframe_has_expected_columns(self, balanced_labels_probs):
        labels, probs = balanced_labels_probs
        _, df = sweep_thresholds(labels, probs, num_thresholds=11)
        for col in ("threshold", "sensitivity", "specificity", "precision", "f1", "accuracy", "tp", "tn", "fp", "fn"):
            assert col in df.columns

    def test_thresholds_span_zero_to_one(self, balanced_labels_probs):
        labels, probs = balanced_labels_probs
        _, df = sweep_thresholds(labels, probs, num_thresholds=11)
        assert df["threshold"].min() == pytest.approx(0.0)
        assert df["threshold"].max() == pytest.approx(1.0)

    def test_raises_on_too_few_thresholds(self, balanced_labels_probs):
        labels, probs = balanced_labels_probs
        with pytest.raises(ValueError, match="num_thresholds"):
            sweep_thresholds(labels, probs, num_thresholds=1)


# ─── Group 3: find_optimal_thresholds ────────────────────────────────────────

class TestFindOptimalThresholds:

    def test_returns_expected_keys(self, balanced_labels_probs):
        labels, probs = balanced_labels_probs
        result = find_optimal_thresholds(labels, probs, num_thresholds=21)
        for key in ("by_roc_youden", "by_max_f1", "by_target_sensitivity_95",
                    "by_target_specificity_90", "metrics_df"):
            assert key in result

    def test_youden_threshold_in_unit_interval(self, balanced_labels_probs):
        labels, probs = balanced_labels_probs
        result = find_optimal_thresholds(labels, probs, num_thresholds=21)
        thr = result["by_roc_youden"]
        if thr is not None:
            assert 0.0 <= thr <= 1.0

    def test_single_class_youden_is_none(self):
        labels = np.array([0, 0, 0, 0])
        probs  = np.array([0.1, 0.2, 0.3, 0.4])
        result = find_optimal_thresholds(labels, probs, num_thresholds=11)
        assert result["by_roc_youden"] is None

    def test_sensitivity_95_threshold_achieves_target(self, perfect_labels_probs):
        labels, probs = perfect_labels_probs
        result = find_optimal_thresholds(labels, probs, num_thresholds=101)
        thr = result["by_target_sensitivity_95"]
        if thr is not None:
            m = metrics_at_threshold(labels, probs, thr)
            assert m["sensitivity"] >= 0.95

    def test_specificity_90_threshold_achieves_target(self, perfect_labels_probs):
        labels, probs = perfect_labels_probs
        result = find_optimal_thresholds(labels, probs, num_thresholds=101)
        thr = result["by_target_specificity_90"]
        if thr is not None:
            m = metrics_at_threshold(labels, probs, thr)
            assert m["specificity"] >= 0.90

    def test_metrics_df_rows_match_num_thresholds(self, balanced_labels_probs):
        labels, probs = balanced_labels_probs
        result = find_optimal_thresholds(labels, probs, num_thresholds=51)
        assert len(result["metrics_df"]) == 51


# ─── Group 4: main() integration ─────────────────────────────────────────────

def _make_fake_evaluate_fn(n=8):
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * (n // 2))
    probs  = rng.uniform(0.1, 0.9, size=n)
    probs[labels == 1] = np.clip(probs[labels == 1], 0.5, 0.95)
    probs[labels == 0] = np.clip(probs[labels == 0], 0.05, 0.49)
    image_ids = [f"img_{i:03d}" for i in range(n)]
    return MagicMock(return_value={
        "labels": labels, "probs": probs, "loss": 0.4, "auc": 0.85, "image_ids": image_ids,
    })


class TestMainIntegration:

    def _run_main(self, monkeypatch, tmp_run_dir, dummy_loader, split="test"):
        fake_model = nn.Linear(10, 2)
        # No val_auc — this project's checkpoints store train_loss/train_auc
        # (final training epoch; see train.py's module docstring for why
        # there's no val-selected "best" epoch).
        fake_ckpt  = {
            "model_state_dict": fake_model.state_dict(),
            "epoch": 5,
            "train_loss": 0.31,
            "train_auc": 0.91,
        }
        monkeypatch.setattr("sys.argv", [
            "evaluate.py", "--run_dir", str(tmp_run_dir),
            "--split", split, "--threshold_split", "same",
        ])
        with (
            # 2-tuple (train, test) — this dataset has no val split.
            patch.object(eval_module, "create_dataloaders",
                         return_value=(dummy_loader, dummy_loader)),
            patch.object(eval_module, "create_model", return_value=fake_model),
            patch.object(eval_module, "evaluate", _make_fake_evaluate_fn()),
            patch("torch.load", return_value=fake_ckpt),
            patch.object(fake_model, "load_state_dict"),
        ):
            eval_module.main()

    def test_main_creates_eval_json(self, monkeypatch, tmp_run_dir, dummy_loader):
        self._run_main(monkeypatch, tmp_run_dir, dummy_loader, split="test")
        assert (tmp_run_dir / "eval_test.json").exists()

    def test_main_train_split_key_in_json(self, monkeypatch, tmp_run_dir, dummy_loader):
        # Reference repo tested "val" here; this dataset's --split choices
        # are {"train", "test"} — train is the diagnostic split instead.
        self._run_main(monkeypatch, tmp_run_dir, dummy_loader, split="train")
        data = json.loads((tmp_run_dir / "eval_train.json").read_text())
        assert data["split"] == "train"

    def test_main_json_has_required_keys(self, monkeypatch, tmp_run_dir, dummy_loader):
        self._run_main(monkeypatch, tmp_run_dir, dummy_loader, split="test")
        data = json.loads((tmp_run_dir / "eval_test.json").read_text())
        for key in ("split", "checkpoint_epoch", "auc_roc", "accuracy",
                    "sensitivity", "specificity", "confusion_matrix",
                    "optimal_thresholds", "n_samples"):
            assert key in data, f"Missing key: {key}"

    def test_main_json_confusion_matrix_values_are_int(self, monkeypatch, tmp_run_dir, dummy_loader):
        self._run_main(monkeypatch, tmp_run_dir, dummy_loader, split="test")
        data = json.loads((tmp_run_dir / "eval_test.json").read_text())
        for key in ("tn", "fp", "fn", "tp"):
            assert isinstance(data["confusion_matrix"][key], int)

    def test_main_creates_roc_png(self, monkeypatch, tmp_run_dir, dummy_loader):
        self._run_main(monkeypatch, tmp_run_dir, dummy_loader, split="test")
        assert (tmp_run_dir / "eval_test_roc_curve.png").exists()

    def test_main_creates_threshold_sweep_csv(self, monkeypatch, tmp_run_dir, dummy_loader):
        self._run_main(monkeypatch, tmp_run_dir, dummy_loader, split="test")
        assert (tmp_run_dir / "eval_test_threshold_sweep.csv").exists()
