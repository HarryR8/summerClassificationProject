"""
Grid search functionality for hyperparameter tuning of the SICM classifier.

Adapted from the reference irc-classification-project's scripts/search.py.

*** Read this before using the "Best AUC" output below ***
The reference script picks the winning hyperparameter combo by the
highest val_auc reached during each run. This dataset's train.py has no
val split (see src/sicm/data/prepare_data.py), so there is no leakage-free
per-config signal to search over here. This script instead reports each
config's FINAL-epoch train_auc/train_loss (matching whatever checkpoint
train.py actually saves) — but selecting hyperparameters by train
performance systematically favours configs that overfit hardest, which is
close to the opposite of what you want. Comparing many configs against
test AUC would be worse, not better — that's hyperparameter search
directly on the number you're supposed to report, across many candidates,
which is a stronger form of the same leakage.

If you want a defensible hyperparameter search, the honest fix is k-fold
(or leave-one-session-out) cross-validation *within the train split only*,
never touching test until one final config is chosen. This script doesn't
implement that — ask if you want it built; it's a bigger restructuring
than swapping which field gets read from history.json.

Treat the ranking below as "which configs converge/fit fastest", not
"which config will generalise best".

This module can be imported into train.py or run as a standalone script.

Example usage:
    # As standalone script
    python scripts/search.py --models efficientnet_b0 dinov3_base --epochs 10

    # Import into train.py (modify train.py to use search functionality)
    from scripts.search import run_grid_search
    results = run_grid_search(models=['efficientnet_b0'], param_grid=param_grid, epochs=10)
"""

import argparse
import json
import math
import os
import subprocess
import sys
from itertools import product
from pathlib import Path


def get_default_param_grids():
    """Get default parameter grids for grid search based on model type."""
    return {
        "efficientnet_b0": {
            "lr": [1e-4, 5e-4, 1e-3],
            "weight_decay": [1e-5, 1e-4, 1e-3],
            "batch_size": [8, 16],
        },
        "densenet121": {
            "lr": [1e-4, 5e-4, 1e-3],
            "weight_decay": [1e-5, 1e-4, 1e-3],
            "batch_size": [8, 16],
        },
        "dinov3_base": {
            "lr": [1e-5, 5e-5],
            "weight_decay": [1e-2, 5e-3, 1e-3],
            "batch_size": [8, 16],
        },
        "resnet18": {
            "lr": [1e-4, 5e-4, 1e-3],
            "weight_decay": [1e-5, 1e-4, 1e-3],
            "batch_size": [8, 16],
        },
        "resnet50": {
            "lr": [1e-4, 5e-4, 1e-3],
            "weight_decay": [1e-5, 1e-4, 1e-3],
            "batch_size": [8, 16],
        },
        "dinov3_large": {
            "lr": [5e-6, 1e-5, 5e-5],
            "weight_decay": [1e-3, 1e-2, 5e-2],
            "batch_size": [4, 8],
        },
        "clip_vit_base": {
            "lr": [1e-4, 5e-4, 1e-3],
            "weight_decay": [1e-5, 1e-4, 1e-3],
            "batch_size": [8, 16],
        },
        "dinov2_base": {
            "lr": [1e-5, 5e-5],
            "weight_decay": [1e-2, 5e-3, 1e-3],
            "batch_size": [8, 16],
        },
        "dinov2_large": {
            "lr": [5e-6, 1e-5, 5e-5],
            "weight_decay": [1e-3, 1e-2, 5e-2],
            "batch_size": [4, 8],
        },
    }


def submit_slurm_job(cmd, job_name, logs_dir, slurm_time, slurm_mem, slurm_gres, slurm_partition, repo_root):
    """Submit a single training run as a SLURM job via sbatch."""
    import shlex
    logs_dir = os.path.abspath(logs_dir)
    os.makedirs(logs_dir, exist_ok=True)
    sbatch_args = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={logs_dir}/{job_name}_%j.out",
        f"--error={logs_dir}/{job_name}_%j.err",
        f"--time={slurm_time}",
        f"--mem={slurm_mem}",
        f"--chdir={repo_root}",
    ]
    if slurm_gres:
        sbatch_args.append(f"--gres={slurm_gres}")
    if slurm_partition:
        sbatch_args.append(f"--partition={slurm_partition}")
    pythonpath_prefix = f"export PYTHONPATH={repo_root / 'src'}:$PYTHONPATH && "
    wrap_cmd = pythonpath_prefix + " ".join(shlex.quote(str(a)) for a in cmd)
    sbatch_args += ["--wrap", wrap_cmd]

    result = subprocess.run(sbatch_args, capture_output=True, text=True)
    job_id = result.stdout.strip().split()[-1] if result.returncode == 0 else None
    return result.returncode == 0, job_id, result.stderr.strip()


def run_single_training(model, params, epochs, output_base_dir="runs/search", seed=42, python_cmd="",
                        images_dir=None,
                        head_type="linear", dropout=0.3, patience=0,
                        freeze_backbone=False,
                        slurm=False, slurm_time="4:00:00", slurm_mem="32G",
                        slurm_gres="gpu:1", slurm_partition=""):
    """Run a single training run with given parameters."""
    param_str = "_".join([f"{k}={v}" for k, v in params.items()])
    output_dir = os.path.join(output_base_dir, f"{model}_{param_str}")

    # Build command — default to sys.executable so the subprocess uses the same
    # venv that is already active, avoiding "No module named 'sicm'" errors.
    cmd = (python_cmd.split() if python_cmd else [sys.executable]) + [
        "scripts/train.py",
        "--model", model,
        "--epochs", str(epochs),
        "--batch_size", str(params["batch_size"]),
        "--lr", str(params["lr"]),
        "--weight_decay", str(params["weight_decay"]),
        "--output_dir", output_dir,
        "--seed", str(seed),
        "--head_type", head_type,
        "--dropout", str(dropout),
    ]

    if patience > 0:
        cmd += ["--patience", str(patience)]
    if freeze_backbone:
        cmd += ["--freeze_backbone"]
    if images_dir:
        cmd += ["--images_dir", images_dir]

    repo_root = Path(__file__).parent.parent

    print(f"{'Submitting' if slurm else 'Running'}: {' '.join(cmd)}")

    if slurm:
        job_name = f"gs_{model}_{param_str}"[:64]
        logs_dir = os.path.join(output_base_dir, "logs")
        success, job_id, err = submit_slurm_job(
            cmd, job_name, logs_dir, slurm_time, slurm_mem, slurm_gres, slurm_partition, repo_root
        )
        return {
            "model": model,
            "params": params,
            "output_dir": output_dir,
            "success": success,
            "job_id": job_id,
            "final_train_auc": None,  # not yet available (jobs run asynchronously)
            "error": err if not success else None,
        }

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(cmd, cwd=repo_root, env=env)
        success = result.returncode == 0

        # Read final-epoch train_auc from history.json (no val_auc exists — see module docstring).
        final_train_auc = None
        history_path = os.path.join(output_dir, "history.json")
        if success and os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            if history:
                final_train_auc = history[-1]["train_auc"]

        return {
            "model": model,
            "params": params,
            "output_dir": output_dir,
            "success": success,
            "final_train_auc": final_train_auc,
        }

    except Exception as e:
        return {
            "model": model,
            "params": params,
            "output_dir": output_dir,
            "success": False,
            "final_train_auc": None,
            "error": str(e)
        }


def run_grid_search(models, param_grid=None, epochs=10, output_base_dir="runs/search", seed=42, python_cmd="",
                    images_dir=None,
                    head_type="linear", dropout=0.3, patience=0,
                    freeze_backbone=False,
                    part=1, num_parts=1,
                    slurm=False, slurm_time="4:00:00", slurm_mem="32G",
                    slurm_gres="gpu:1", slurm_partition=""):
    """Run grid search over specified models and parameter combinations."""
    if param_grid is None:
        param_grid = get_default_param_grids()

    results = []

    total_runs = 0
    for m in models:
        if m in param_grid:
            all_combos = list(product(*param_grid[m].values()))
            chunk_size = math.ceil(len(all_combos) / num_parts)
            start = (part - 1) * chunk_size
            total_runs += len(all_combos[start:start + chunk_size])
    run_idx = 0

    for model in models:
        if model not in param_grid:
            print(f"Warning: No parameter grid defined for {model}, skipping")
            continue

        grid = param_grid[model]
        param_names = list(grid.keys())
        param_values = list(grid.values())

        all_combos = list(product(*param_values))
        chunk_size = math.ceil(len(all_combos) / num_parts)
        start = (part - 1) * chunk_size
        combos_to_run = all_combos[start:start + chunk_size]

        for combination in combos_to_run:
            params = dict(zip(param_names, combination))
            run_idx += 1

            print(f"\n--- {'Submitting' if slurm else 'Running'} [{run_idx}/{total_runs}] {model} with params: {params} ---")
            result = run_single_training(
                model, params, epochs, output_base_dir, seed, python_cmd, images_dir,
                head_type=head_type, dropout=dropout, patience=patience,
                freeze_backbone=freeze_backbone,
                slurm=slurm, slurm_time=slurm_time, slurm_mem=slurm_mem,
                slurm_gres=slurm_gres, slurm_partition=slurm_partition,
            )
            results.append(result)

            if slurm:
                if result["success"]:
                    print(f"Submitted job ID: {result['job_id']}")
                else:
                    print(f"Submission failed: {result.get('error', 'Unknown error')}")
            elif result["success"] and result["final_train_auc"] is not None:
                print(f"Final train AUC: {result['final_train_auc']:.4f}  (NOT a generalisation estimate — see module docstring)")
            else:
                print(f"Failed: {result.get('error', 'Unknown error')}")

    os.makedirs(output_base_dir, exist_ok=True)
    summary_file = os.path.join(output_base_dir, "grid_search_results.json")

    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("GRID SEARCH SUMMARY")
    print(f"{'='*50}")

    successful_runs = [r for r in results if r["success"]]
    if slurm:
        submitted = [r for r in successful_runs if r.get("job_id")]
        print(f"Submitted {len(submitted)} SLURM job(s).")
        for r in submitted:
            print(f"  job {r['job_id']}: {r['model']} {r['params']}")
    else:
        auc_runs = [r for r in successful_runs if r.get("final_train_auc") is not None]
        if auc_runs:
            best_result = max(auc_runs, key=lambda x: x["final_train_auc"])
            print(f"Highest final train AUC: {best_result['final_train_auc']:.4f}  (train-set fit, not a generalisation estimate)")
            print(f"Params: {best_result['params']}")
            print(f"Model: {best_result['model']}")
            print(f"Output dir: {best_result['output_dir']}")
            print("\nTo actually compare configs on held-out data, run scripts/evaluate.py "
                  "on each run_dir and compare eval_test.json — but do that ONCE you've "
                  "picked a config some other way (e.g. train-set convergence behaviour), "
                  "not as the search criterion itself.")
        else:
            print("No successful runs completed.")

    print(f"Total runs: {len(results)}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Results saved to: {summary_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run grid search for the SICM classifier")
    parser.add_argument("--models", nargs="+", default=["resnet18"],
                        help="Models to include in grid search")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs per training run")
    parser.add_argument("--output_dir", type=str, default="runs/search",
                        help="Base directory for search outputs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--python_cmd", type=str, default="",
                        help="Python command to invoke train.py (default: sys.executable)")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Path to images directory")
    parser.add_argument("--head_type", type=str, default="linear",
                        help="Classification head type: 'linear', 'mlp', or 'mlp_deep' (default: linear)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate for MLP head (default: 0.3)")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience on TRAIN loss, in epochs; 0 disables (default: 0)")
    parser.add_argument("--slurm", action="store_true",
                        help="Submit each run as a SLURM job instead of running sequentially")
    parser.add_argument("--slurm_time", type=str, default="4:00:00",
                        help="SLURM wall-time per job (default: 4:00:00)")
    parser.add_argument("--slurm_mem", type=str, default="32G",
                        help="SLURM memory per job (default: 32G)")
    parser.add_argument("--slurm_gres", type=str, default="gpu:1",
                        help="SLURM generic resources, e.g. 'gpu:1' (default: gpu:1)")
    parser.add_argument("--slurm_partition", type=str, default="",
                        help="SLURM partition/queue name (optional)")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone weights during training (default: False)")
    parser.add_argument("--part", type=int, default=1,
                        help="Which partition to run (1-indexed). Use with --num_parts.")
    parser.add_argument("--num_parts", type=int, default=1,
                        help="Total number of partitions to split the config list into (default: 1 = no split).")

    args = parser.parse_args()

    print(f"Starting grid search for models: {args.models}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Output base dir: {args.output_dir}")
    print(f"Head type: {args.head_type}, dropout: {args.dropout}, patience: {args.patience}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    if args.num_parts > 1:
        print(f"Partition: {args.part}/{args.num_parts}")
    print(f"Python command: {args.python_cmd or sys.executable}")
    if args.slurm:
        print(f"SLURM mode: time={args.slurm_time}, mem={args.slurm_mem}, "
              f"gres={args.slurm_gres}, partition={args.slurm_partition or '(default)'}")

    run_grid_search(
        models=args.models,
        epochs=args.epochs,
        output_base_dir=args.output_dir,
        seed=args.seed,
        python_cmd=args.python_cmd,
        images_dir=args.images_dir,
        head_type=args.head_type,
        dropout=args.dropout,
        patience=args.patience,
        freeze_backbone=args.freeze_backbone,
        part=args.part,
        num_parts=args.num_parts,
        slurm=args.slurm,
        slurm_time=args.slurm_time,
        slurm_mem=args.slurm_mem,
        slurm_gres=args.slurm_gres,
        slurm_partition=args.slurm_partition,
    )


if __name__ == "__main__":
    main()
