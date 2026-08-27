"""PyTorch Dataset for the SICM cancerous/noncancerous cell-scan classifier.

Model-agnostic: __getitem__ returns raw PIL.Image + metadata.
Preprocessing (resize, normalize, augment) is handled externally via
preprocessing.py so any backbone (ImageNet CNN, CLIP, DINOv2, …) can
inject its own transform through the collate function.

This mirrors busbra.data.dataset.BUSBRADataset from the reference
irc-classification-project repo as closely as the dataset allows. The
main differences from BUS-BRA:
  - Images are single-channel height-map renders (converted from raw
    .img files by sicm_img_converter.py), not RGB photographs — they are
    still opened with .convert("RGB") so every backbone gets a 3-channel
    input, the channel is just a replicated grayscale value.
  - There is no lesion segmentation mask equivalent for SICM scans, so
    `masks_dir` is accepted for interface parity with the reference
    pipeline's CLI flags but is not supported — passing it raises
    immediately rather than silently doing nothing.
  - "Case" in the SICM cases.csv is 1:1 with each scan (no repeat
    imaging per physical sample the way BUS-BRA has multiple images per
    patient); the grouping variable that matters for SICM is "session"
    (the day/plate a batch of scans was collected on). See
    prepare_data.py for why session — not Case — drives the split.
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SICMDataset(Dataset):
    """SICM cancerous/noncancerous cell height-map dataset.

    Returns raw PIL.Image objects so that model-specific preprocessing
    can be applied outside the Dataset (see preprocessing.py).
    """

    def __init__(
        self,
        split_file: str | Path,          # path to splits.csv (ID, Case, label, split, …)
        images_dir: str | Path,          # directory containing <filename> PNGs (cancerous/, noncancerous/ subfolders)
        split: str,                      # one of "train", "val", "test"
        masks_dir: str | Path | None = None,  # unsupported for SICM — kept for CLI/API parity only
    ):
        if masks_dir is not None:
            raise NotImplementedError(
                "masks_dir is not supported for the SICM dataset — there is no "
                "lesion-segmentation-mask equivalent for these scans. Leave "
                "--masks_dir unset (the default)."
            )

        self.images_dir = Path(images_dir)

        df = pd.read_csv(split_file)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples for split '{split}'")
        print(f"Loaded {split}: {len(self.df)} images")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample.

        Returns
        -------
        dict with keys:
            "image"    : PIL.Image in RGB mode  (no resize/normalize)
            "label"    : int  — 0 noncancerous, 1 cancerous
            "case"     : str  — Case ID (1:1 with a scan for SICM data)
            "image_id" : str  — image ID (used to build the filename)
        """
        row = self.df.iloc[idx]

        # "filename" already includes the cancerous/noncancerous subfolder,
        # e.g. "noncancerous/SICM_090224_1359_000t.png" — matches the
        # --output layout produced by sicm_img_converter.py.
        img_path = self.images_dir / row["filename"]

        # Convert to RGB so every backbone receives a 3-channel image, even
        # though the source PNG is single-channel (mode 'L') grayscale from
        # sicm_img_converter.render() — convert() replicates the one channel
        # across R/G/B.
        image = Image.open(img_path).convert("RGB")

        return {
            "image": image,
            "label": int(row["label"]),
            "case": str(row["Case"]),
            "image_id": str(row["ID"]),
        }
