"""Model-specific preprocessing registry for SICM.

Usage
-----
    from sicm.data.preprocessing import get_preprocess

    preprocess = get_preprocess("imagenet_cnn", split="train", size=224)
    tensor = preprocess(pil_image)   # torch.FloatTensor (3, H, W)

Supported model_keys
--------------------
    "imagenet_cnn"  — Albumentations letterbox + ImageNet mean/std.
                      Training split adds flip/rotate/brightness augmentation.
    "clip"          — HuggingFace CLIPProcessor (openai/clip-vit-base-patch32).
    "dinov2"        — HuggingFace AutoImageProcessor (facebook/dinov2-base).
    "dinov3"        — HuggingFace AutoImageProcessor (facebook/dinov3-vitb16-pretrain-lvd1689m).
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Helper: PIL → numpy (uint8 HWC) for Albumentations (expects numpy input)
# ---------------------------------------------------------------------------

def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to uint8 HWC numpy array."""
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Lesion cropping: load image + mask and crop to the lesion bounding box
# ---------------------------------------------------------------------------

def load_and_crop_to_lesion(
    image_path: str | Path,
    mask_path: str | Path,
    padding: int = 20,
) -> np.ndarray:
    """Load an image and crop it to the lesion bounding box from its mask.

    Parameters
    ----------
    image_path : str or Path
        Path to the RGB ultrasound image.
    mask_path : str or Path
        Path to the binary segmentation mask (white lesion on black background).
    padding : int
        Pixels to expand the bounding box on each side, clamped to image bounds.

    Returns
    -------
    np.ndarray
        Cropped image as (H, W, 3) uint8 array.  If no white pixels are found
        in the mask, the full image is returned unchanged and a warning is logged.
    """
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask  = np.array(Image.open(mask_path).convert("L"),   dtype=np.uint8)

    binary = mask >= 127
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)

    if not rows.any():
        warnings.warn(
            f"No lesion pixels found in mask '{mask_path}'; returning full image.",
            stacklevel=2,
        )
        return image

    h, w = image.shape[:2]
    y_min, y_max = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    x_min, x_max = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])

    y_min = max(0, y_min - padding)
    y_max = min(h - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(w - 1, x_max + padding)

    return image[y_min : y_max + 1, x_min : x_max + 1]


# ---------------------------------------------------------------------------
# Picklable callable classes for DataLoader multiprocessing workers
# ---------------------------------------------------------------------------

class _ImagenetCNNPreprocess:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, img):
        return self.pipeline(image=_pil_to_numpy(img))["image"]


class _CLIPPreprocess:
    def __init__(self, processor, augment: bool):
        self.processor = processor
        self.augment = augment

    def __call__(self, img):
        if self.augment and random.random() < 0.5:
            import torchvision.transforms.functional as TF
            img = TF.hflip(img)
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


class _DinoPreprocess:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


# ---------------------------------------------------------------------------
# imagenet_cnn: Albumentations letterbox + ImageNet normalisation
# ---------------------------------------------------------------------------

def _make_imagenet_cnn_preprocess(split: str, size: int) -> Callable:
    """Return a callable that accepts a PIL.Image and returns (3,H,W) tensor.

    Training: letterbox → flip / rotate / brightness → normalize → tensor.
    Val/test: letterbox → normalize → tensor.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Letterbox: scale longest side to `size`, pad remainder with black
    letterbox = [
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, fill=0),
    ]

    # Normalisation for ImageNet pre-trained CNNs (e.g. ResNet50), expects input in [0, 1] range (Albumentations handles this internally when using ToTensorV2)
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Augmentations applied to `train` split (flip, rotate, brightness/contrast) to artificially increase data diversity and reduce overfitting.
    if split == "train":
        pipeline = A.Compose([
            *letterbox,
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, border_mode=0, fill=0, p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.ElasticTransform(alpha=50, sigma=5, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            normalize,
            ToTensorV2(),
        ])
    # No augmentations to `val` and `test` splits— just resize and normalize. Want deterministic, consistent evaluation.
    else:
        pipeline = A.Compose([
            *letterbox,
            normalize,
            ToTensorV2(),
        ])

    return _ImagenetCNNPreprocess(pipeline)


# ---------------------------------------------------------------------------
# CLIP: HuggingFace CLIPProcessor
# ---------------------------------------------------------------------------

# Default public CLIP checkpoint — override by passing model_name kwarg if
# you call _make_clip_preprocess directly.
_CLIP_DEFAULT = "openai/clip-vit-base-patch32"


def _make_clip_preprocess(split: str, size: int, model_name: str = _CLIP_DEFAULT) -> Callable:
    """Return a callable that uses CLIPProcessor to preprocess a PIL.Image.

    CLIPProcessor handles its own resize/crop and normalisation internally,
    so we do not apply any additional transforms.  For the training split we
    optionally add a horizontal flip (cheap, keeps pixel statistics intact).
    Size is passed as a hint but CLIPProcessor may ignore it if the model
    has a fixed resolution.
    """
    from transformers import CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name)
    return _CLIPPreprocess(processor, augment=(split == "train"))


# ---------------------------------------------------------------------------
# DINOv2 / DINOv3: HuggingFace AutoImageProcessor
# ---------------------------------------------------------------------------

_DINO_CHECKPOINTS = {
    "dinov2": "facebook/dinov2-base",
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}


def _make_dino_preprocess(model_key: str, split: str, size: int) -> Callable:
    """Return a callable that uses AutoImageProcessor for DINOv2/v3.

    AutoImageProcessor applies model-specific resize + normalisation.
    No extra augmentation is added (ViT models are sensitive to distribution
    shift; strong augmentation should be done at the training-loop level if
    needed via separate torchvision transforms before calling this pipeline).
    """
    from transformers import AutoImageProcessor

    checkpoint = _DINO_CHECKPOINTS[model_key]
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    return _DinoPreprocess(processor)


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

def get_preprocess(model_key: str, split: str, size: int = 224) -> Callable:
    """Return a preprocessing callable for the requested backbone.

    Parameters
    ----------
    model_key : str
        One of "imagenet_cnn", "clip", "dinov2", "dinov3".
    split : str
        One of "train", "val", "test".  Controls whether augmentation
        is applied (training only, and only for imagenet_cnn / clip).
    size : int
        Target spatial resolution.  Used by the imagenet_cnn pipeline;
        HuggingFace processors derive their own resolution from the
        model config but we pass it through for forward-compat.

    Returns
    -------
    Callable[[PIL.Image], torch.FloatTensor]
        A function that takes a PIL.Image (RGB) and returns a
        (3, H, W) float32 tensor, ready to be stacked into a batch.
    """
    if model_key == "imagenet_cnn":
        return _make_imagenet_cnn_preprocess(split=split, size=size)
    elif model_key == "clip":
        return _make_clip_preprocess(split=split, size=size)
    elif model_key in ("dinov2", "dinov3"):
        return _make_dino_preprocess(model_key=model_key, split=split, size=size)
    else:
        raise ValueError(
            f"Unknown model_key '{model_key}'. "
            "Choose from: 'imagenet_cnn', 'clip', 'dinov2', 'dinov3'."
        )
