"""Model factory for SICM binary classification.

Usage
-----
    from sicm.models.factory import create_model, create_backbone, get_preprocess_key

    # Full fine-tuning (default)
    model = create_model("resnet18", num_classes=2, pretrained=True)

    # Frozen backbone + MLP head
    model = create_model("resnet18", freeze_backbone=True, head_type="mlp")

    # Backbone only (for pre-computing embeddings)
    backbone, embed_dim = create_backbone("resnet18")

Model keys and their preprocessing counterparts
------------------------------------------------
    "resnet18"        → timm / "imagenet_cnn" / embed 512
    "resnet50"        → timm / "imagenet_cnn" / embed 2048
    "efficientnet_b0" → timm / "imagenet_cnn" / embed 1280
    "densenet121"     → timm / "imagenet_cnn" / embed 1024
    "dinov2_small"    → HuggingFace / "dinov2" / embed 384
    "dinov2_base"     → HuggingFace / "dinov2" / embed 768
    "dinov2_large"    → HuggingFace / "dinov2" / embed 1024
    "dinov2_base_reg" → HuggingFace / "dinov2" / embed 768  (with registers)
    "dinov2_large_reg"→ HuggingFace / "dinov2" / embed 1024 (with registers)
    "dinov3_small"    → HuggingFace / "dinov3" / embed 384
    "dinov3_base"     → HuggingFace / "dinov3" / embed 768
    "dinov3_large"    → HuggingFace / "dinov3" / embed 1024
    "clip_vit_base"   → open_clip / "clip"   / embed 512
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .heads import create_head as _create_head
except ImportError:
    from heads import create_head as _create_head  # when run directly as a script

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Registry of available models and their configurations.  Each entry maps a
# model name (e.g. "resnet18") to a config dict that specifies:
#   - "type":          how the model is created in create_model()
#   - "preprocess_key": the key for data/preprocessing.py
#   - "embedding_dim": output size of the backbone (num_classes=0 mode)
MODEL_REGISTRY: dict[str, dict] = {
    # --- ImageNet CNNs via timm -------------------------------------------
    "resnet18": {
        "type": "timm",
        "timm_name": "resnet18",
        "embedding_dim": 512,
        "preprocess_key": "imagenet_cnn",
    },
    "resnet50": {
        "type": "timm",
        "timm_name": "resnet50",
        "embedding_dim": 2048,
        "preprocess_key": "imagenet_cnn",
    },
    "efficientnet_b0": {
        "type": "timm",
        "timm_name": "efficientnet_b0",
        "embedding_dim": 1280,
        "preprocess_key": "imagenet_cnn",
    },
    "densenet121": {
        "type": "timm",
        "timm_name": "densenet121",
        "embedding_dim": 1024,
        "preprocess_key": "imagenet_cnn",
    },
    # --- DINOv2 (patch size 14, 2023) via HuggingFace --------------------
    "dinov2_small": {
        "type": "dinov2",
        "hf_name": "facebook/dinov2-small",
        "embedding_dim": 384,
        "preprocess_key": "dinov2",
    },
    "dinov2_base": {
        "type": "dinov2",
        "hf_name": "facebook/dinov2-base",
        "embedding_dim": 768,
        "preprocess_key": "dinov2",
    },
    "dinov2_large": {
        "type": "dinov2",
        "hf_name": "facebook/dinov2-large",
        "embedding_dim": 1024,
        "preprocess_key": "dinov2",
    },
    # --- DINOv2 with registers (cleaner attention maps) ------------------
    "dinov2_base_reg": {
        "type": "dinov2",
        "hf_name": "facebook/dinov2-with-registers-base",
        "embedding_dim": 768,
        "preprocess_key": "dinov2",
    },
    "dinov2_large_reg": {
        "type": "dinov2",
        "hf_name": "facebook/dinov2-with-registers-large",
        "embedding_dim": 1024,
        "preprocess_key": "dinov2",
    },
    # --- DINOv3 (patch size 16, August 2025, SOTA) via HuggingFace ------
    "dinov3_small": {
        "type": "dinov3",
        "hf_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "embedding_dim": 384,
        "preprocess_key": "dinov3",
    },
    "dinov3_base": {
        "type": "dinov3",
        "hf_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "embedding_dim": 768,
        "preprocess_key": "dinov3",
    },
    "dinov3_large": {
        "type": "dinov3",
        "hf_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "embedding_dim": 1024,
        "preprocess_key": "dinov3",
    },
    # --- CLIP ViT ---------------------------------------------------------
    "clip_vit_base": {
        "type": "clip",
        "clip_name": "ViT-B-32",
        "embedding_dim": 512,
        "preprocess_key": "clip",
    },
}


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _create_timm_classifier(config: dict, num_classes: int, pretrained: bool) -> nn.Module:
    """Full timm classifier with a timm-managed head (num_classes > 0)."""
    import timm

    # timm replaces the final classification head automatically when
    # num_classes differs from the pretrained checkpoint's 1000-class head.
    return timm.create_model(
        config["timm_name"],
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=3,
    )


def _create_timm_backbone(config: dict, pretrained: bool) -> tuple[nn.Module, int]:
    """timm feature extractor with the classification head removed (num_classes=0)."""
    import timm

    model = timm.create_model(
        config["timm_name"],
        pretrained=pretrained,
        num_classes=0,   # returns (B, embedding_dim) instead of logits
        in_chans=3,
    )
    return model, config["embedding_dim"]


def _create_dino_backbone(config: dict, pretrained: bool) -> tuple[nn.Module, int]:
    """Create DINOv2 or DINOv3 backbone via HuggingFace transformers.

    Wraps the model to extract the CLS token (position 0) from
    last_hidden_state, giving a (B, embedding_dim) feature vector.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(config["hf_name"])
    embedding_dim = config["embedding_dim"]

    class DinoBackbone(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            outputs = self.model(x)
            return outputs.last_hidden_state[:, 0, :]  # CLS token

    return DinoBackbone(model), embedding_dim


class _DINOv3Wrapper(nn.Module):
    """Wraps HuggingFace DINOv3 AutoModel to return pooler_output as a plain tensor."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).pooler_output


def _create_dinov3_backbone(config: dict, pretrained: bool) -> tuple[nn.Module, int]:
    """Create DINOv3 backbone via HuggingFace AutoModel."""
    from transformers import AutoModel, AutoConfig
    if pretrained:
        model = AutoModel.from_pretrained(config["hf_name"])
    else:
        cfg = AutoConfig.from_pretrained(config["hf_name"])
        model = AutoModel.from_config(cfg)
    return _DINOv3Wrapper(model), config["embedding_dim"]


def _create_clip_backbone(config: dict, pretrained: bool) -> tuple[nn.Module, int]:
    """Create CLIP visual encoder via open_clip.

    Requires: uv pip install -e ".[clip]"
    """
    import open_clip
    pretrained_tag = "openai" if pretrained else None
    model, _, _ = open_clip.create_model_and_transforms(config["clip_name"], pretrained=pretrained_tag)
    visual = model.visual  # Extract vision encoder only
    return visual, config["embedding_dim"]

# ---------------------------------------------------------------------------
# BackboneWithHead wrapper
# ---------------------------------------------------------------------------

class BackboneWithHead(nn.Module):
    """Combines a (possibly frozen) backbone with a classification head.

    forward(x) returns logits regardless of whether the backbone is frozen
    or fully trainable.  get_features(x) returns raw backbone embeddings.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

        if freeze_backbone:
            freeze_module(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → logits (B, num_classes)."""
        features = self.backbone(x)   # (B, embedding_dim)
        return self.head(features)    # (B, num_classes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone embeddings without the classification head."""
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    head_type: str = "linear",
    head_hidden_dim: int = 256,
    head_dropout: float = 0.3,
) -> nn.Module:
    """Create a classification model.

    Parameters
    ----------
    model_name:
        Key from MODEL_REGISTRY.
    num_classes:
        Number of output classes (2 for binary noncancerous/cancerous).
    pretrained:
        Load pretrained weights when True.
    freeze_backbone:
        If True, freeze all backbone weights and attach a custom head
        specified by head_type.  Only head parameters will be trainable.
        If False, return the standard timm model with all params trainable.
    head_type:
        Classification head type — "linear", "mlp", or "mlp_deep".
        Only used when freeze_backbone=True.
    head_hidden_dim:
        Hidden dimension for MLP heads.
    head_dropout:
        Dropout rate for MLP heads.

    Returns
    -------
    nn.Module
        forward(x) returns logits of shape (B, num_classes).

    Examples
    --------
    >>> # Full fine-tuning (default)
    >>> model = create_model("resnet18", num_classes=2)

    >>> # Frozen backbone with MLP head
    >>> model = create_model("resnet18", freeze_backbone=True, head_type="mlp")
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list_models()}"
        )

    config = MODEL_REGISTRY[model_name]
    model_type = config["type"]

    if model_type == "timm":
        if freeze_backbone:
            backbone, embed_dim = _create_timm_backbone(config, pretrained=pretrained)
            head = _create_head(
                head_type,
                in_features=embed_dim,
                num_classes=num_classes,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout,
            )
            return BackboneWithHead(backbone, head, freeze_backbone=True)
        else:
            return _create_timm_classifier(config, num_classes=num_classes, pretrained=pretrained)

    elif model_type in ("dinov2", "dinov3", "clip"):
        if model_type == "dinov2":
            backbone, embed_dim = _create_dino_backbone(config, pretrained=pretrained)
        elif model_type == "dinov3":
            backbone, embed_dim = _create_dinov3_backbone(config, pretrained=pretrained)
        else:
            backbone, embed_dim = _create_clip_backbone(config, pretrained=pretrained)
        head = _create_head(
            head_type,
            in_features=embed_dim,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )
        return BackboneWithHead(backbone, head, freeze_backbone=freeze_backbone)
    else:
        raise NotImplementedError(
            f"Unknown model type '{model_type}' for model '{model_name}'"
        )


def create_backbone(
    model_name: str,
    pretrained: bool = True,
) -> tuple[nn.Module, int]:
    """Create a backbone (feature extractor) without a classification head.

    Use this for pre-computing and caching embeddings offline.

    Parameters
    ----------
    model_name:
        Key from MODEL_REGISTRY.
    pretrained:
        Load pretrained weights when True.

    Returns
    -------
    (backbone, embedding_dim)
        backbone: nn.Module where forward(x) returns (B, embedding_dim)
        embedding_dim: int, the feature vector size

    Examples
    --------
    >>> backbone, embed_dim = create_backbone("resnet18")
    >>> backbone.eval()
    >>> with torch.no_grad():
    ...     features = backbone(images)  # (B, 512)
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list_models()}"
        )

    config = MODEL_REGISTRY[model_name]
    model_type = config["type"]

    if model_type == "timm":
        return _create_timm_backbone(config, pretrained=pretrained)
    elif model_type == "dinov2":
        return _create_dino_backbone(config, pretrained=pretrained)
    elif model_type == "dinov3":
        return _create_dinov3_backbone(config, pretrained=pretrained)
    elif model_type == "clip":
        return _create_clip_backbone(config, pretrained=pretrained)
    else:
        raise NotImplementedError(
            f"Unknown model type '{model_type}' for model '{model_name}'"
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def list_models() -> list[str]:
    """Return sorted list of available model names."""
    return sorted(MODEL_REGISTRY.keys())


def get_model_config(model_name: str) -> dict:
    """Return the registry config dict for a model.

    Raises KeyError if model_name is not registered.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list_models()}"
        )
    return MODEL_REGISTRY[model_name]


def get_preprocess_key(model_name: str) -> str:
    """Return the preprocessing key that pairs with this model.

    Links models/factory.py to data/preprocessing.py:
        "resnet18"      → "imagenet_cnn"
        "dinov2_base"   → "dinov2"
        "clip_vit_base" → "clip"
    """
    return get_model_config(model_name)["preprocess_key"]


def get_embedding_dim(model_name: str) -> int:
    """Return the backbone embedding dimension for a model."""
    return get_model_config(model_name)["embedding_dim"]


def freeze_module(module: nn.Module) -> None:
    """Freeze all parameters in a module (in-place)."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters in a module (in-place)."""
    for param in module.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters, split by trainable vs frozen.

    Returns
    -------
    dict with keys "total", "trainable", "frozen".
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Available models:", list_models())
    print()

    x = torch.randn(2, 3, 224, 224)

    # Test 1: Full fine-tuning
    model = create_model("resnet18", num_classes=2, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Full model:        {count_parameters(model)}")

    # Test 2: Frozen backbone + linear head
    model = create_model("resnet18", num_classes=2, pretrained=False,
                         freeze_backbone=True, head_type="linear")
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Frozen + linear:   {count_parameters(model)}")

    # Test 3: Frozen backbone + MLP head
    model = create_model("resnet18", num_classes=2, pretrained=False,
                         freeze_backbone=True, head_type="mlp")
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Frozen + mlp:      {count_parameters(model)}")

    # Test 4: Backbone only
    backbone, embed_dim = create_backbone("resnet18", pretrained=False)
    backbone.eval()
    with torch.no_grad():
        features = backbone(x)
    print(f"Backbone only:     embed_dim={embed_dim}, features shape={tuple(features.shape)}")

    # Forward pass shape check
    print(f"\nForward pass: {tuple(x.shape)} → {tuple(out.shape)}")
    print(f"Preprocess key: {get_preprocess_key('resnet18')!r}")

    # Test DINOv2 (requires network access on first run)
    try:
        model = create_model("dinov2_base", num_classes=2, freeze_backbone=True)
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"dinov2_base:       {tuple(x.shape)} → {tuple(out.shape)}")
    except Exception as e:
        print(f"dinov2_base skipped: {e}")

    # Test DINOv3 (requires transformers with DINOv3 support)
    try:
        model = create_model("dinov3_base", num_classes=2, freeze_backbone=True)
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"dinov3_base:       {tuple(x.shape)} → {tuple(out.shape)}")
    except Exception as e:
        print(f"dinov3_base skipped: {e}")
        print("  Install dev transformers if needed: pip install git+https://github.com/huggingface/transformers")
