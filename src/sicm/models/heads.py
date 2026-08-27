"""Classification head architectures for transfer learning.

Usage
-----
    from sicm.models.heads import create_head, list_heads

    head = create_head("mlp", in_features=512, num_classes=2, dropout=0.3)
    logits = head(features)  # (B, 2)

Head types
----------
    "linear"   — Linear(in_features, num_classes)
    "mlp"      — One hidden layer: Linear → ReLU → Dropout → Linear
    "mlp_deep" — Two hidden layers: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Head architectures
# ---------------------------------------------------------------------------

_HEAD_INFO: dict[str, dict] = {
    "linear": {
        "description": "Single linear layer (baseline).",
        "num_layers": 1,
    },
    "mlp": {
        "description": "One hidden layer: Linear → ReLU → Dropout → Linear.",
        "num_layers": 2,
    },
    "mlp_deep": {
        "description": "Two hidden layers: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear.",
        "num_layers": 3,
    },
}


def create_head(
    head_type: str,
    in_features: int,
    num_classes: int = 2,
    hidden_dim: int = 256,  # Default hidden dimension is a common choice for MLP heads, but can be tuned based on validation performance.
    dropout: float = 0.3,   # Default dropout is a common choice for MLP heads, but can be tuned based on validation performance.
) -> nn.Module:
    """Create a classification head.

    Parameters
    ----------
    head_type:
        One of "linear", "mlp", "mlp_deep".
    in_features:
        Input dimension (embedding size from backbone).
    num_classes:
        Number of output classes.
    hidden_dim:
        Hidden layer dimension (for mlp variants).
    dropout:
        Dropout probability (for mlp variants).

    Returns
    -------
    nn.Module
        Head where forward(x) maps (B, in_features) → (B, num_classes).

    Raises
    ------
    ValueError
        If head_type is not recognised.
    """
    if head_type == "linear":
        return nn.Linear(in_features, num_classes)

    elif head_type == "mlp":
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    elif head_type == "mlp_deep":
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    else:
        raise ValueError(
            f"Unknown head type '{head_type}'. "
            f"Available heads: {list_heads()}"
        )


def list_heads() -> list[str]:
    """Return available head types."""
    return list(_HEAD_INFO.keys())


def get_head_info(head_type: str) -> dict:
    """Return info about a head type (description, num_layers).

    Raises ValueError if head_type is not recognised.
    """
    if head_type not in _HEAD_INFO:
        raise ValueError(
            f"Unknown head type '{head_type}'. "
            f"Available heads: {list_heads()}"
        )
    return _HEAD_INFO[head_type]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for head_type in list_heads():
        head = create_head(head_type, in_features=512, num_classes=2)
        x = torch.randn(4, 512)
        out = head(x)
        n_params = sum(p.numel() for p in head.parameters())
        print(f"{head_type}: input {tuple(x.shape)} → output {tuple(out.shape)}, params: {n_params:,}")
