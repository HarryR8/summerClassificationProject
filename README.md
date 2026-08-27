# SICM Cancerous / Noncancerous Cell Classifier

Adapted from [irc-classification-project](https://github.com/HarryR8/irc-classification-project)
(BUS-BRA breast-ultrasound benign/malignant classifier). Same modular
transfer-learning pipeline — model factory, preprocessing registry,
training loop, metrics library, threshold analysis, ensembling, results
aggregation, HPC batch scripts — retargeted at SICM (Scanning Ion
Conductance Microscopy) cell height-map scans instead of ultrasound images.

## Read this first: two structural differences from the reference repo

**1. Two-way split (train/test), not three-way (train/val/test).**
`cases.csv`'s `session` column (the day/plate a batch of scans was
collected on) is confounded with `label`: every session is 100% one
class (2 sessions entirely cancerous, 2 entirely noncancerous). With
only 4 independent sessions total, there aren't enough to keep every
split session-disjoint *and* class-balanced in a three-way split (that
needs 6+ sessions minimum). So this pipeline holds out whole sessions
for **test** only, and everything else is **train** — no val. Full
reasoning in `src/sicm/data/prepare_data.py`'s module docstring, which
is worth reading before you trust any number this produces. Practical
knock-on effects, all documented in their respective files:
- `train.py` has no val-based checkpoint selection — it just trains for
  `--epochs` and saves the final epoch as `best.pt`/`last.pt` (identical).
  `--patience` early-stops on **train** loss instead of val AUC.
- `evaluate.py` picks its four "optimal threshold" candidates from the
  **train** split's predictions by default (`--threshold_split train`),
  not a clean held-out set — treat those as exploratory, report the
  threshold=0.5 metrics as primary.
- `search.py` (grid search) has no leakage-free way to rank
  hyperparameter configs — see its module docstring before using it.

**2. `session` is confounded with `label`.** Even with session-disjoint
test, the model only ever sees 2 independent "batches" during training.
It's entirely possible for it to learn day-specific scanning artifacts
(background level, calibration drift) rather than real cancerous/
noncancerous morphology, and there's no way to fully rule that out with
this data. Treat test-set AUC as an **optimistic upper bound** until
scans from more independent sessions exist, and say so in any write-up.
`data/splits/split_info.json` records exactly which sessions ended up
where so this is auditable.

## Everything you need to do, in order

1. **Push this to GitHub** (if you haven't already):
   ```bash
   cd summerClassificationProject
   git remote add origin https://github.com/HarryR8/summerClassificationProject.git
   git push -u origin main
   ```
2. **Add your PNGs** — drop your converted images into
   `data/raw/cancerous/` and `data/raw/noncancerous/`, matching the
   `filename` column already in `data/raw/cases.csv` (e.g.
   `cancerous/SICM_050224_1748_035t.png`) — exactly the `--output`
   layout `scripts/sicm_img_converter.py` already produces. Both folders
   are gitignored (see `.gitignore`) so the PNGs themselves won't get
   pushed to GitHub — only `cases.csv`/`manifest.csv` and the generated
   splits are tracked.
3. **Install locally** and verify the pipeline end-to-end (Setup +
   Training sections below) on a laptop-friendly model (`resnet18`) —
   catch any path/environment problems here, where iteration is fast,
   before touching the HPC queue.
4. **Run the full comparison matrix on Imperial's HPC** — see "Running
   on Imperial College HPC" below. This mirrors the reference project's
   own workflow almost exactly (same `uv`/PBS approach, same
   9-model x condition matrix, same script names), scaled to this
   dataset's much smaller size.
5. **Evaluate, ensemble, aggregate** — `scripts/evaluate.py` per run
   (or `scripts/run_all_evals.sh` for all of them), optionally
   `scripts/ensemble_eval.py` across your best runs, then
   `scripts/collect_results.py` to pull everything into one
   `results/summary.csv`.
6. **Read the caveats above again** before writing up any number —
   they apply to every run this pipeline produces, however good the
   AUC looks.

## Repo structure

```
summerClassificationProject/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── test_checkpoint_loading.py # Standalone checkpoint save/load smoke test
├── test_factory_smoke.py      # Standalone model factory smoke test
├── test_grid.py               # Standalone grid-search combination printout
├── tests/                     # pytest suite (unit + evaluate.py integration tests)
├── scripts/
│   ├── sicm_img_converter.py # raw .img -> normalised 224x224 PNG (your uploaded script, unchanged)
│   ├── train.py               # CLI training entrypoint (single model/config)
│   ├── train_cnn.sh            # Batch: resnet18/50, densenet121, clip_vit_base (A/B each)
│   ├── train_dino.sh           # Batch: dinov2/3 base+large (A/B each)
│   ├── train_all.sh            # Runs both of the above
│   ├── evaluate.py             # Evaluate a checkpoint
│   ├── run_all_evals.sh        # Evaluate every run in runs/ automatically
│   ├── search.py               # Grid search (read the caveats first)
│   ├── sanity_dataloader.py    # Verify batch shapes/dtypes
│   ├── ensemble_eval.py        # Ensemble inference across multiple checkpoints
│   ├── collect_results.py      # Aggregate run evaluations -> results/summary.csv
│   ├── plot_epoch_roc.py       # Plot per-epoch ROC curves (diagnostic)
│   └── hpc/                    # PBS job scripts for Imperial College HPC — see below
│       ├── train_all_hpc.pbs / train_cnn_hpc.pbs / train_dino_hpc.pbs
│       ├── evaluate_hpc.pbs / ensemble_eval_hpc.pbs / run_all_evals_hpc.pbs
│       └── search_resnet18_hpc.pbs, search_dinov2_large_hpc_part{1,2,3}.pbs (examples)
├── src/sicm/
│   ├── data/
│   │   ├── prepare_data.py   # Load cases.csv, create SESSION-level train/test splits
│   │   ├── dataset.py        # Model-agnostic PyTorch Dataset (returns PIL.Image)
│   │   ├── preprocessing.py  # Backbone-specific preprocessing registry (unchanged from reference)
│   │   └── loaders.py        # Collate functions + DataLoader factory (train/test)
│   ├── models/
│   │   ├── factory.py        # Model registry + create_model / create_backbone (unchanged)
│   │   └── heads.py          # linear / mlp / mlp_deep heads (unchanged)
│   └── training/
│       ├── trainer.py        # train_one_epoch + evaluate (unchanged)
│       └── metrics.py        # metrics_at_threshold, find_optimal_thresholds (unchanged)
├── data/
│   ├── raw/
│   │   ├── cases.csv         # ← already here (your uploaded file)
│   │   ├── manifest.csv      # ← already here (your uploaded file, used for QC filtering)
│   │   ├── cancerous/        # ← put your converted PNGs here (gitignored)
│   │   └── noncancerous/     # ← put your converted PNGs here (gitignored)
│   └── splits/                # ← already generated from cases.csv; regenerate anytime with prepare_data.py
│       ├── splits.csv
│       ├── session_splits.csv
│       └── split_info.json
└── runs/                      # ← created by train.py (gitignored)
```

Modules marked "(unchanged)" above are copied from the reference repo
essentially as-is — they're model-agnostic and had no BUS-BRA-specific
logic to adapt. Everything is smoke-tested — see "Tests" below.

## Setup (local)

```bash
cd summerClassificationProject
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e ".[clip]"   # optional, only if you want clip_vit_base
```

Regenerate the splits anytime after editing `cases.csv`:

```bash
python -m src.sicm.data.prepare_data --cases_csv data/raw/cases.csv --images_dir data/raw
```

Current split (150 scans after QC-flag exclusion, 29 dropped):

| split | images | sessions | cancerous | noncancerous |
|---|---|---|---|---|
| train | 86 | 2024_02_05_2, 2024_02_06 | 46 | 40 |
| test  | 64 | 2024_02_09, 2024_02_22_2 | 42 | 22 |

Verify the pipeline (no download needed for `imagenet_cnn`):

```bash
python scripts/sanity_dataloader.py --model_key imagenet_cnn \
  --split_file data/splits/splits.csv --images_dir data/raw
```

## Training

Single model:
```bash
python scripts/train.py --model resnet18 --epochs 30 --batch_size 16
```

Full comparison matrix (9 models x condition A/B — see "What changed" below):
```bash
bash scripts/train_all.sh        # or train_cnn.sh / train_dino.sh separately
```

**Batch size**: default is 16 (down from the reference repo's 32); even
so, `drop_last=True` on the train loader still throws away a meaningful
chunk each epoch (86 train images, batch_size 16 -> 5 full batches, 6
images dropped per epoch). Consider `--batch_size 8` if you want less
data thrown away, at the cost of noisier batch statistics.

Model registry, head types, and per-model default hyperparameters are
identical to the reference repo (resnet18/50, efficientnet_b0,
densenet121, dinov2_*, dinov3_*, clip_vit_base). Given ~86 training
images, the big ViT foundation models — dinov2_large/dinov3_large in
particular — are far more prone to overfitting here than they were on
BUS-BRA's 1300+ training images; frozen-backbone (condition A) is the
safer default for those, full fine-tuning (condition B) is included for
comparison but don't expect it to win.

## Evaluation

```bash
python scripts/evaluate.py --run_dir runs/resnet18_<timestamp>
```

Writes `eval_test.json`, `eval_test_roc_curve.png`,
`eval_test_threshold_sweep.csv` to the run directory. Use `--split train`
for a diagnostic look at train-set fit (never report this as
generalisation performance). To evaluate every completed run at once:

```bash
bash scripts/run_all_evals.sh
```

## Running on Imperial College HPC

This mirrors the reference project's own workflow: `uv`-managed
dependencies, PBS job scripts, the same 9-model comparison matrix, same
script names under `scripts/hpc/`. What's different is scaled down for
this dataset's much smaller size (see each script's header comments) —
walltime estimates below are starting points, not guarantees, since I
haven't been able to actually time a real epoch on your images.

### 0. Prerequisites

You need an active Imperial HPC account (PhD/research students: ask your
supervisor to register you — see [Get Access](https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/get-access/)).
Imperial's HPC uses the **PBS Pro** scheduler (`qsub`/`qstat`, not SLURM)
— full docs at the [RCS User Guide](https://icl-rcs-user-guide.readthedocs.io/en/latest/).

### 1. Get the code and data onto the HPC

SSH into a login node (must be on the college network or connected via
[Unified Access](https://www.imperial.ac.uk/admin-services/ict/self-service/connect-and-use-it/wifi-and-remote-access/unified-access/) off-campus):

```bash
ssh <your_college_username>@login.cx3.hpc.imperial.ac.uk
```

Your `$HOME` directory (930 GB quota) is where you land and is backed by
the Research Data Store — plenty of room for this project. Clone your repo there:

```bash
git clone https://github.com/HarryR8/summerClassificationProject.git ~/summerClassificationProject
cd ~/summerClassificationProject
```

The PNGs are gitignored, so `git clone` won't bring them — transfer them
separately with `rsync` or `scp` from your Mac (run this from your Mac,
not the HPC):

```bash
rsync -avz data/raw/cancerous/    <username>@login.cx3.hpc.imperial.ac.uk:~/summerClassificationProject/data/raw/cancerous/
rsync -avz data/raw/noncancerous/ <username>@login.cx3.hpc.imperial.ac.uk:~/summerClassificationProject/data/raw/noncancerous/
```

(For much larger datasets in future, Imperial recommend
[Globus](https://icl-rcs-user-guide.readthedocs.io/en/latest/rds/transferringdata/globus/)
over rsync/scp — not necessary at this dataset's size.)

### 2. Set up the environment (on the login node, not in a job)

The reference project used [`uv`](https://docs.astral.sh/uv/) directly
rather than Imperial's officially-documented conda route, and that's what
every script in `scripts/hpc/` assumes (`python scripts/train.py` inside
a job that already has `uv`'s venv on `PATH` — see the note at the end of
this step). Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env    # or restart your shell
```

Then, from the project root:
```bash
uv sync
```

This builds `.venv` from `pyproject.toml`. **Do this on the login node,
not inside a PBS job** — Imperial's best-practice guidance is not to do
package installation from within batch jobs, and it means jobs start
running immediately instead of spending queue time installing packages.

If `uv` isn't available or doesn't work for you for any reason, the
Imperial-documented fallback is conda — see the
[PyTorch application guide](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/applications/guides/pytorch/)
and swap `python scripts/train.py` for `uv run python scripts/train.py`
throughout, or activate your conda env at the top of each `.pbs` script instead.

The PBS scripts in `scripts/hpc/` don't have a `module load` line — that
matches what actually worked for the reference project. If a fresh
`.pbs` job can't find `python`/`uv` on `PATH`, add
`module load Python/3.12.3-GCCcore-13.3.0` (or run `module avail Python`
to see what's currently offered) near the top, before `cd "$PROJECT_DIR"`.

**Pre-cache pretrained weights before submitting jobs.** Compute nodes
are meant for computation, not on-demand downloads — the safest and
fastest approach is to warm the cache once on the login node so jobs
never need network access mid-run:

```bash
python -c "
from sicm.models import create_model
for m in ['resnet18','resnet50','densenet121','efficientnet_b0','clip_vit_base','dinov2_base','dinov2_large']:
    create_model(m, num_classes=2)
    print(m, 'cached')
"
```

DINOv3 checkpoints are gated on HuggingFace and need a token — see
`scripts/sanity_dataloader.py --model_key dinov3` for the account/access
steps, then on the HPC:

```bash
echo "hf_your_token_here" > ~/.hf_token && chmod 600 ~/.hf_token
export HF_TOKEN=$(cat ~/.hf_token)
python -c "
from sicm.models import create_model
for m in ['dinov3_base','dinov3_large']:
    create_model(m, num_classes=2)
    print(m, 'cached')
"
```

Every `.pbs` script that trains/evaluates a DINOv3 model already does
`export HF_TOKEN=$(cat ~/.hf_token)` for you — just make sure that file
exists first.

### 3. Submit jobs

```bash
cd ~/summerClassificationProject
qsub scripts/hpc/train_all_hpc.pbs          # everything, one job (~8h estimate)
# or, to parallelise and get results faster:
qsub scripts/hpc/train_cnn_hpc.pbs          # resnet/densenet/clip (~4h estimate)
qsub scripts/hpc/train_dino_hpc.pbs         # dinov2/dinov3 (~6h estimate)
```

Each prints a job ID (e.g. `403036.pbs-7`). Monitor with:
```bash
qstat -u <your_college_username>
```
`R` = running, `Q` = queued. Output/errors land in `logs/` (created
automatically) as `<name>.out` / `<name>.err`. Cancel a job with
`qdel <job_id>` if needed.

**These walltimes are estimates, not measurements** — I don't have your
actual images so I can't time a real epoch. This dataset's train split
(~86 images) is roughly 15x smaller than the reference project's, and
epoch counts here are already scaled down (30-50 vs their 50-100), so
these jobs should run considerably faster than the reference project's
own 16-48h budgets — but "should" isn't "will." Submit
`scripts/hpc/train_cnn_hpc.pbs` first (cheapest, fastest models), check
`logs/train_cnn.out` for per-epoch timing once it starts, and adjust the
`#PBS -l walltime=` line in the other scripts before submitting them if
your numbers look very different from mine.

### 4. GPU selection

All the `.pbs` scripts request `ngpus=1` without pinning a specific
card, which currently defaults to an **L40S (48GB)** — plenty for every
model in this project, including dinov2_large/dinov3_large. If you ever
want a different card, add `:gpu_type=A100` or `:gpu_type=RTX6000` to
the `#PBS -l select=...` line (see the
[GPU Jobs guide](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/queues/gpu-jobs/)
for current specs/availability) — RTX6000 has less VRAM (24GB) but is
often less contended if you're just running the smaller CNNs.

### 5. Grid search on HPC

`scripts/hpc/search_resnet18_hpc.pbs` is a template for a single-job
search (18 configs via `scripts/search.py`'s default grid). For the two
large ViT models, split the grid across parallel jobs instead of one
long one — `scripts/hpc/search_dinov2_large_hpc_part{1,2,3}.pbs`
demonstrates the pattern (`--part N --num_parts 3`, each covering 6 of
18 configs). Copy either pattern for the other six models, changing
`--models`/`#PBS -N`/`-o`/`-e`. **Read `scripts/search.py`'s module
docstring before trusting its "Highest final train AUC" output** — with
no val split, that ranks configs by training-set fit, not generalisation.

### 6. Bring results back

```bash
# from your Mac:
rsync -avz <username>@login.cx3.hpc.imperial.ac.uk:~/summerClassificationProject/runs/    ./runs/
rsync -avz <username>@login.cx3.hpc.imperial.ac.uk:~/summerClassificationProject/results/ ./results/
```

Then run `scripts/collect_results.py` locally (or on the HPC before
copying back — either works) to build `results/summary.csv`.

### Troubleshooting

- **"command not found: python" / "uv" in a job** — add the `module
  load` line mentioned in step 2, or double check `uv sync` actually
  completed on the login node first.
- **CUDA out of memory** on dinov2_large/dinov3_large condition B — drop
  `--batch_size` further in `train_dino.sh`, or request `gpu_type=A100`
  for more headroom.
- **HuggingFace 401/403 errors for DINOv3** — your HF account needs
  approved access to each gated checkpoint repo, not just a token; see
  `scripts/sanity_dataloader.py`'s error message for the exact repos.
- **General HPC issues** — [FAQ and Common Issues](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/faq/),
  or [book a clinic slot](https://icl-rcs-user-guide.readthedocs.io/en/latest/support/attend-a-clinic/) with RCS.

## Ensembling, results aggregation

Same CLIs as the reference repo (`ensemble_eval.py`, `collect_results.py`,
`plot_epoch_roc.py`) — see each script's docstring for the specific
two-way-split adaptations.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v                        # pytest suite
python test_factory_smoke.py            # standalone model factory smoke test
python test_checkpoint_loading.py       # standalone checkpoint save/load smoke test
python test_grid.py                     # standalone grid-search combination printout
```

`tests/test_evaluate.py`, `tests/test_heads.py`, `tests/test_factory.py`,
`tests/test_trainer.py`, and the three root-level scripts are all ported
from the reference repo's own test suite. Everything that doesn't touch a
real pretrained backbone (metrics, heads, `list_models`, invalid-name
handling, the `evaluate.py` integration tests, `test_grid.py`) was run
in the sandbox that built this project and passed — 44 tests. Everything
that does need a pretrained backbone (`create_model(...)` with its
default `pretrained=True`, i.e. most of `test_factory.py` and
`test_trainer.py`, plus `test_checkpoint_loading.py` and
`test_factory_smoke.py`) couldn't be exercised there because that sandbox
has no route to huggingface.co — every one of those failures was the
identical `403`/`LocalEntryNotFoundError`, not a code issue. They should
pass wherever you have a normal internet connection — including on the
HPC login node, per step 2 above.

## What's genuinely different from a straightforward port

Everything in `models/` and `training/` is an unmodified copy (fully
dataset-agnostic in the reference repo already). What actually changed:

- `dataset.py` — renamed, masks_dir disabled (no lesion-mask equivalent
  for SICM scans), otherwise identical shape.
- `prepare_data.py` — rewritten: session-level (not patient-level)
  grouping, two-way split, QC-flag exclusion (`manifest.csv`,
  `exclude_qc_flagged=True` by default, drops 29 corrupted-looking
  scans — pass `--include_qc_flagged` to keep them).
- `loaders.py` — returns `(train_loader, test_loader)`, a 2-tuple.
- `train.py` — no val loop; final-epoch checkpoint instead of
  best-val-AUC checkpoint; train-loss-based `--patience`; class weights
  computed dynamically from the actual train split
  (`--class_weight_mode balanced`, default) instead of BUS-BRA's
  hardcoded `[0.32, 0.68]`.
- `evaluate.py` / `ensemble_eval.py` — `--split {train,test}` instead of
  `{val,test}`; `--threshold_split` defaults to `train`.
- `search.py` — reports final train AUC per config instead of best val
  AUC, with a caveat that this isn't an unbiased generalisation estimate.
- `collect_results.py` / `plot_epoch_roc.py` — no BUS-BRA-specific logic
  to change; only cosmetic (dropped the poster-prefix filter, added a
  docstring note).
- `tests/` — ported with renamed imports; `tests/test_evaluate.py`'s
  integration tests adapted for the 2-tuple dataloader and dropped-val
  checkpoint format; `test_checkpoint_loading.py` rewritten to be
  self-contained (the reference version loaded checkpoints from
  hardcoded `/tmp/...` paths left over from a prior run).
- `train_cnn.sh` / `train_dino.sh` / `train_all.sh` — 2 conditions (A/B)
  instead of 3 (no masks), epoch/patience budgets scaled down for this
  dataset's much smaller train split — see each script's header.
- `run_all_evals.sh` — walks `runs/` automatically instead of a
  hand-maintained list of the reference project's own timestamped run
  names, which were meaningless for a fresh project.
- `scripts/hpc/*.pbs` — same PBS structure and `uv`-based approach as
  the reference project's own HPC scripts, with generic `$HOME`-based
  paths instead of a hardcoded username, scaled-down walltime estimates,
  and the pretrained-weight pre-caching step made explicit (see "Running
  on Imperial College HPC" above) rather than assumed.
