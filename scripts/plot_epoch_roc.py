"""Plot per-epoch ROC curves from epoch_test_preds.npz files.

Single-run mode — per-epoch curves overlaid, coloured by epoch number:
    uv run python scripts/plot_epoch_roc.py --run_dir runs/<model>_<timestamp>

Multi-run comparison — best-epoch ROC per run, one line per run:
    uv run python scripts/plot_epoch_roc.py \\
        --run_dirs runs/resnet18_* runs/densenet121_* runs/dinov3_large_*

Outputs
-------
Single-run : <run_dir>/epoch_roc_curves.png
Multi-run  : results/epoch_roc_comparison.png

Requires --eval_test_every_epoch to have been passed to train.py (that's
what writes epoch_test_preds.npz). The "(best)" epoch highlighted here is
for visual diagnostics only — this dataset's train.py has no val split,
so the checkpoint actually saved as best.pt/last.pt is always the FINAL
epoch, not whichever epoch has the highest test AUC here. If they differ
noticeably, that's a useful signal about training stability, but don't
substitute the "best" epoch's number for the final-epoch number when
reporting results — the former is chosen by looking at test data.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MPL_CONFIG_DIR = Path(__file__).resolve().parent.parent / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def load_npz(run_dir: Path) -> dict:
    npz_path = run_dir / "epoch_test_preds.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"epoch_test_preds.npz not found in {run_dir}")
    data = np.load(npz_path, allow_pickle=True)
    return {
        "probs": data["probs"],      # (N_epochs, N_test) float32
        "labels": data["labels"],    # (N_test,) int64
        "aucs": data["aucs"],        # (N_epochs,) float64
        "epochs": data["epochs"],    # (N_epochs,) int32
    }


def plot_single_run(run_dir: Path, out_path: Path):
    data = load_npz(run_dir)
    probs = data["probs"]
    labels = data["labels"]
    aucs = data["aucs"]
    epochs = data["epochs"]

    n_epochs = len(epochs)
    best_idx = int(np.argmax(aucs))

    # Load model name from config if available
    config_path = run_dir / "config.json"
    model_name = run_dir.name
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        model_name = cfg.get("model", run_dir.name)

    fig, ax = plt.subplots(figsize=(7.0, 6.5), dpi=200)
    cmap = plt.get_cmap("plasma")

    for i, (epoch, p) in enumerate(zip(epochs, probs)):
        fpr, tpr, _ = roc_curve(labels, p)
        color = cmap(i / max(n_epochs - 1, 1))
        lw = 2.5 if i == best_idx else 0.8
        alpha = 1.0 if i == best_idx else 0.4
        label = f"Epoch {epoch} — AUC={aucs[i]:.4f} (best)" if i == best_idx else None
        ax.plot(fpr, tpr, color=color, lw=lw, alpha=alpha, label=label)

    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"Per-Epoch Test ROC — {model_name}\n({n_epochs} epochs shown)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=9)

    # Colorbar for epoch number
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=int(epochs[0]), vmax=int(epochs[-1])))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Epoch", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_multi_run(run_dirs: list[Path], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 6.5), dpi=200)
    colors = plt.get_cmap("tab10").colors

    for i, run_dir in enumerate(run_dirs):
        try:
            data = load_npz(run_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e} — skipping")
            continue

        probs = data["probs"]
        labels = data["labels"]
        aucs = data["aucs"]
        epochs = data["epochs"]
        best_idx = int(np.argmax(aucs))

        config_path = run_dir / "config.json"
        label_name = run_dir.name
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            label_name = cfg.get("model", run_dir.name)

        fpr, tpr, _ = roc_curve(labels, probs[best_idx])
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2.0,
                label=f"{label_name}  AUC={aucs[best_idx]:.4f} (ep {epochs[best_idx]})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Best-Epoch Test ROC — Multi-Run Comparison")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot per-epoch ROC curves from epoch_test_preds.npz")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_dir", type=str,
                       help="Single run directory (produces per-epoch curves)")
    group.add_argument("--run_dirs", type=str, nargs="+",
                       help="Multiple run directories (produces comparison plot)")
    parser.add_argument("--out", type=str, default=None,
                        help="Override output file path")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        out_path = Path(args.out).expanduser().resolve() if args.out else run_dir / "epoch_roc_curves.png"
        plot_single_run(run_dir, out_path)
    else:
        run_dirs = [Path(d).expanduser().resolve() for d in args.run_dirs]
        out_path = Path(args.out).expanduser().resolve() if args.out else Path("results/epoch_roc_comparison.png")
        plot_multi_run(run_dirs, out_path)


if __name__ == "__main__":
    main()
