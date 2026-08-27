"""Sanity-check script for the SICM data pipeline.

Run from the repo root:
    python scripts/sanity_dataloader.py --model_key imagenet_cnn
    python scripts/sanity_dataloader.py --model_key clip
    python scripts/sanity_dataloader.py --model_key dinov2
    python scripts/sanity_dataloader.py --model_key dinov3

For each split (train, test — no val, see prepare_data.py) the script
prints the batch tensor shape/dtype/range and verifies that metadata
lists have the expected length.

Note: --model_key dinov3 requires a HuggingFace token and access to the
Meta DINOv3 checkpoints. Run with --model_key imagenet_cnn first to verify
the rest of the pipeline without any HuggingFace dependencies.
"""

import argparse
import sys
from pathlib import Path

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sicm.data.loaders import create_dataloaders

_DINOV3_REPOS = [
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
]


def _check_hf_token() -> None:
    """Verify HF_TOKEN is set; try ~/.hf_token as fallback. Print guidance and exit if missing."""
    import os

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        token_file = Path.home() / ".hf_token"
        if token_file.exists():
            token = token_file.read_text().strip()
            if token:
                os.environ["HF_TOKEN"] = token
                print(f"  [HF] Loaded token from {token_file}")

    if not token:
        print(
            "\n[ERROR] HF_TOKEN is not set.\n"
            "\nDINOv3 checkpoints are hosted on HuggingFace and require:\n"
            "  1. A HuggingFace account  ->  https://huggingface.co/join\n"
            "  2. A read-access token    ->  https://huggingface.co/settings/tokens\n"
            "  3. Approved access to each DINOv3 checkpoint repo:\n"
        )
        for repo in _DINOV3_REPOS:
            print(f"       https://huggingface.co/{repo}")
        print(
            "\nThen export your token before running:\n"
            "  export HF_TOKEN=\"hf_your_token_here\"\n"
            "\nOr write it to ~/.hf_token (used automatically by this script):\n"
            "  echo \"hf_your_token_here\" > ~/.hf_token && chmod 600 ~/.hf_token\n"
        )
        sys.exit(1)


def check_batch(batch: dict, split: str) -> None:
    """Print key statistics for one batch."""
    img    = batch["image"]
    label  = batch["label"]
    cases  = batch["case"]
    ids    = batch["image_id"]

    print(f"\n  [{split}]")
    print(f"    image  shape={tuple(img.shape)}  dtype={img.dtype}  "
          f"min={img.min():.3f}  max={img.max():.3f}")
    print(f"    label  shape={tuple(label.shape)}  dtype={label.dtype}  "
          f"unique={label.unique().tolist()}")
    print(f"    case   type={type(cases).__name__}  len={len(cases)}  "
          f"example={cases[0]!r}")
    print(f"    id     type={type(ids).__name__}   len={len(ids)}  "
          f"example={ids[0]!r}")

    B = img.shape[0]
    assert label.shape == (B,),            f"label shape mismatch: {label.shape}"
    assert len(cases) == B,                f"case list length mismatch: {len(cases)}"
    assert len(ids)   == B,                f"image_id list length mismatch: {len(ids)}"
    assert img.is_floating_point(), "image tensor must be float"
    print(f"    \u2713 assertions passed")


def main():
    parser = argparse.ArgumentParser(description="Sanity-check SICM dataloaders")
    parser.add_argument(
        "--model_key", default="imagenet_cnn",
        choices=["imagenet_cnn", "clip", "dinov2", "dinov3"],
        help="Backbone key to test",
    )
    parser.add_argument("--split_file",  default="data/splits/splits.csv")
    parser.add_argument("--images_dir",  default="data/raw")
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="0 = main process (easier to debug)")
    parser.add_argument("--size",        type=int, default=224)
    args = parser.parse_args()

    if args.model_key == "dinov3":
        _check_hf_token()

    print(f"Testing model_key='{args.model_key}'  size={args.size}")
    print(f"split_file : {args.split_file}")
    print(f"images_dir : {args.images_dir}")

    train_loader, test_loader = create_dataloaders(
        split_file=args.split_file,
        images_dir=args.images_dir,
        model_key=args.model_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        size=args.size,
    )

    for split, loader in [("train", train_loader), ("test", test_loader)]:
        batch = next(iter(loader))
        check_batch(batch, split)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
