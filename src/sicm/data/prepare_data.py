"""
Prepare SICM dataset: load cases.csv, create SESSION-level train/test splits.

SICM structure (differs from BUS-BRA's patient-level structure):
- cases.csv: ID, Case, Pathology, label, session, split, session_leak_flag, filename
  (produced alongside the converted PNGs — see sicm_img_converter.py)
- Images: cancerous/<ID>.png, noncancerous/<ID>.png
- "Case" is 1:1 with each scan — there is no repeat imaging of the same
  physical sample the way BUS-BRA has multiple images per patient.
- "session" (the day/plate a batch of scans was collected on) is the
  variable that actually needs group-level splitting: within a session,
  scans share the same day's instrument calibration, background/noise
  floor, and physical sample prep, so image-level random splitting risks
  the model learning session-specific artifacts rather than real
  cancerous/noncancerous morphology.

*** IMPORTANT — read this before trusting any reported metric ***
In the data supplied, `session` is a near-perfect predictor of `label`:
every session in cases.csv is 100% one class (2 sessions are entirely
cancerous, 2 are entirely noncancerous). With only 4 independent sessions
(2 per class) total, session and label are CONFOUNDED, not just
correlated. That is also why this pipeline uses a two-way (train/test)
split rather than three-way (train/val/test): three session-disjoint
splits would need at least 2 sessions per class per split (6+ sessions
minimum) just to keep both classes represented everywhere, and this
dataset doesn't have that. Two-way is the most that can be made honest:
  - TEST is session-held-out (>=1 full session per class, never seen
    during training). This is the number that matters.
  - TRAIN gets everything else, including all remaining sessions.
There is deliberately no validation split — see train.py's docstring for
what that means for checkpoint/epoch selection (short version: no
early-stopping-on-val, and no threshold tuning on test).
Treat any test-set AUC from this pipeline as an optimistic upper bound
until scans from more independent sessions are available, and say so
explicitly in any write-up.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


def _select_group_holdout(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    target_ratio: float,
    always_keep_one_group: bool = True,
) -> set:
    """Greedily hold out whole groups (smallest first) per class until the
    held-out fraction of that class's images reaches `target_ratio`, while
    always leaving at least one group per class in the training pool.

    Smallest-first ordering maximises how much data stays available for
    training, which matters when — as here — there are only a handful of
    independent groups (sessions) to begin with.
    """
    held_out = set()
    for _, class_df in df.groupby(label_col):
        counts = class_df.groupby(group_col).size().sort_values()  # ascending
        group_names = list(counts.index)
        total = int(counts.sum())
        cum = 0
        chosen = []
        max_holdout = len(group_names) - 1 if always_keep_one_group else len(group_names)
        for g in group_names:
            if len(chosen) >= max_holdout:
                break
            if total > 0 and cum / total >= target_ratio:
                break
            chosen.append(g)
            cum += int(counts[g])
        held_out.update(chosen)
    return held_out


def create_session_splits(
    cases_csv: Path,
    output_dir: Path,
    test_session_ratio: float = 0.3,
    exclude_qc_flagged: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create train/test splits for the SICM dataset (two-way — see module
    docstring for why there is no val split here).

    TEST is SESSION-held-out: whole sessions (>=1 per class, chosen to
    approximate `test_session_ratio` of each class's images while always
    leaving >=1 session per class in train) are removed entirely. No image
    from a test session appears anywhere in train. Everything else is train.

    With only 4 sessions total (2 cancerous, 2 noncancerous) in the
    supplied cases.csv, this currently holds out exactly the smaller
    session from each class. If more sessions are added later this
    function will automatically spread the holdout across more of them,
    and `test_session_ratio` starts to matter more.

    Parameters
    ----------
    cases_csv : Path
        Path to cases.csv (columns: ID, Case, Pathology, label, session,
        filename, plus whatever else prior tooling added — split and
        session_leak_flag, if present, are ignored and recomputed here).
    output_dir : Path
        Where to write splits.csv / session_splits.csv / split_info.json.
    test_session_ratio : float
        Target fraction of each class's images to hold out as whole
        sessions for test. With only 2 sessions per class this currently
        just selects the smaller session per class regardless of the
        exact value, as long as it's > 0 (see _select_group_holdout).
    exclude_qc_flagged : bool
        If True (default), drop rows manifest.csv flagged qc_flag=True
        (scans quality_flag() identified as likely tip-crash / stage-
        collision artifacts) before splitting. These are corrupted-looking
        scans, not merely noisy ones — mixing them in adds label-irrelevant
        noise on top of an already tiny dataset. Pass False to keep them
        (e.g. to run a sensitivity analysis comparing with/without).
    seed : int
        Kept for interface parity / forward-compatibility (e.g. if a
        future version reintroduces an image-level split somewhere).
        Session-holdout itself is deterministic given the data, not
        randomised, so this currently has no effect.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cases_csv)
    n_before = len(df)
    print(f"Loaded {n_before} scans from {cases_csv}")
    print(f"Pathology distribution:\n{df['Pathology'].value_counts()}\n")

    required_cols = {"ID", "Case", "Pathology", "label", "session", "filename"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"cases.csv is missing required column(s): {missing}")

    # Drop any pre-existing split/session_leak_flag columns — we recompute both.
    df = df.drop(columns=[c for c in ("split", "session_leak_flag") if c in df.columns])

    n_qc_dropped = 0
    if exclude_qc_flagged:
        manifest_path = cases_csv.parent / "manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)[["filename", "qc_flag"]].copy()
            df["_basename"] = df["filename"].apply(lambda p: Path(p).name)
            df = df.merge(manifest, left_on="_basename", right_on="filename", how="left", suffixes=("", "_man"))
            flagged = df["qc_flag"].fillna(False)
            n_qc_dropped = int(flagged.sum())
            df = df[~flagged].drop(columns=["_basename", "filename_man", "qc_flag"], errors="ignore").reset_index(drop=True)
            print(f"Excluded {n_qc_dropped} qc_flag=True scan(s) (see manifest.csv) — "
                  f"{len(df)} remain. Pass exclude_qc_flagged=False to keep them.\n")
        else:
            print(f"exclude_qc_flagged=True but no manifest.csv found next to {cases_csv} — skipping QC filter.\n")

    mixed_sessions = df.groupby("session")["label"].nunique()
    mixed_sessions = mixed_sessions[mixed_sessions > 1]
    if len(mixed_sessions) > 0:
        print(f"Note: {len(mixed_sessions)} session(s) contain both classes: "
              f"{list(mixed_sessions.index)}")

    # ---- Session-level holdout for test -------------------------------------
    test_sessions = _select_group_holdout(
        df, group_col="session", label_col="label", target_ratio=test_session_ratio,
    )
    print(f"Test sessions (held out entirely): {sorted(test_sessions)}")

    out_df = df.copy()
    out_df["split"] = out_df["session"].apply(lambda s: "test" if s in test_sessions else "train")

    if out_df.loc[out_df["split"] == "train", "label"].nunique() < 2:
        raise ValueError(
            "After holding out test sessions, train has only one class — "
            "cannot train a binary classifier. Lower test_session_ratio or "
            "add more sessions."
        )
    if out_df.loc[out_df["split"] == "test", "label"].nunique() < 2:
        print("WARNING: test split has only one class present. AUC/sensitivity/"
              "specificity will be undefined or degenerate for this split.")

    # session_leak_flag: recomputed for transparency. With a pure session
    # holdout this should always be False (a session belongs entirely to
    # train or entirely to test) — kept as an explicit assertion so any
    # future change to the splitting logic can't silently reintroduce leakage.
    out_df["session_leak_flag"] = out_df.groupby("session")["split"].transform(
        lambda s: s.nunique() > 1
    )
    assert not out_df["session_leak_flag"].any(), (
        "Internal error: a session ended up split across train and test."
    )

    # ---- Statistics ----------------------------------------------------------
    print("\n--- Split statistics ---")
    for split in ["train", "test"]:
        split_df = out_df[out_df["split"] == split]
        sessions = sorted(split_df["session"].unique())
        canc = (split_df["label"] == 1).sum()
        noncanc = (split_df["label"] == 0).sum()
        print(f"{split:5s}: {len(split_df):3d} images from {len(sessions)} session(s) {sessions} "
              f"(cancerous={canc}, noncancerous={noncanc})")

    # ---- Save splits.csv ------------------------------------------------------
    cols = ["ID", "Case", "Pathology", "label", "session", "split", "session_leak_flag", "filename"]
    out_df[cols].to_csv(output_dir / "splits.csv", index=False)
    print(f"\nSaved: {output_dir / 'splits.csv'}")

    # ---- Save session-level splits (for auditing) -----------------------------
    session_splits = out_df.groupby("session").agg(
        label=("label", "first"), split=("split", "first"), n_images=("ID", "size")
    ).reset_index()
    session_splits.to_csv(output_dir / "session_splits.csv", index=False)
    print(f"Saved: {output_dir / 'session_splits.csv'}")

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_scans_in_cases_csv": n_before,
        "n_qc_flagged_excluded": n_qc_dropped,
        "n_scans_used": len(out_df),
        "test_sessions": sorted(test_sessions),
        "splits": {s: int((out_df["split"] == s).sum()) for s in ["train", "test"]},
        "split_strategy": "two-way (train/test only, no val — see prepare_data.py module docstring)",
        "warning": (
            "session is confounded with label in this dataset (every session is "
            "100% one class) — test is session-disjoint from train, which is the "
            "best available guard against the model learning session-specific "
            "artifacts instead of real morphology. There is no held-out val "
            "split; see train.py for how model selection and evaluate.py for "
            "how threshold selection are handled without one."
        ),
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {output_dir / 'split_info.json'}")

    return out_df


def verify_images(df: pd.DataFrame, images_dir: Path):
    """Check that every referenced PNG actually exists in images_dir."""
    missing = []
    for _, row in df[["ID", "filename"]].iterrows():
        if not (images_dir / row["filename"]).exists():
            missing.append(row["ID"])

    if missing:
        print(f"\n\u26a0 Missing {len(missing)} image(s), e.g. {missing[:5]}")
    else:
        print(f"\n\u2713 All {len(df)} images found in {images_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases_csv", type=str, default="data/raw/cases.csv",
                        help="Path to cases.csv")
    parser.add_argument("--images_dir", type=str, default="data/raw",
                        help="Directory containing cancerous/ and noncancerous/ PNG subfolders")
    parser.add_argument("--output_dir", type=str, default="data/splits",
                        help="Directory to write splits.csv etc. to")
    parser.add_argument("--test_session_ratio", type=float, default=0.3)
    parser.add_argument("--include_qc_flagged", action="store_true", default=False,
                        help="Keep manifest.csv qc_flag=True scans (default: excluded)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases_csv = Path(args.cases_csv)
    if not cases_csv.exists():
        print(f"Error: {cases_csv} not found")
        return

    df = create_session_splits(
        cases_csv=cases_csv,
        output_dir=Path(args.output_dir),
        test_session_ratio=args.test_session_ratio,
        exclude_qc_flagged=not args.include_qc_flagged,
        seed=args.seed,
    )
    verify_images(df, Path(args.images_dir))


if __name__ == "__main__":
    main()
