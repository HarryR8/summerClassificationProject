#!/usr/bin/env python
"""Test checkpoint save/load round-trip and inference.

Adapted from the reference irc-classification-project's
test_checkpoint_loading.py. The original script loaded pre-existing
checkpoints from hardcoded /tmp/test_training/*/best.pt paths left over
from a prior manual training run — not reproducible on a fresh checkout.
This version creates its own throwaway checkpoints first (matching this
project's actual checkpoint format: model_state_dict/epoch/train_loss/
train_auc — no val_auc, since there's no val split here; see train.py's
module docstring), so it runs standalone with `python test_checkpoint_loading.py`.
"""
import tempfile
from pathlib import Path

import torch

from sicm.models.factory import create_model


def make_checkpoint(path: Path, freeze_backbone: bool, head_type: str):
    model = create_model("resnet18", num_classes=2, freeze_backbone=freeze_backbone, head_type=head_type)
    torch.save({
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "train_loss": 0.31,
        "train_auc": 0.91,
    }, path)
    return model


print("=" * 60)
print("CHECKPOINT SAVE/LOAD TEST")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # Test 1: Full fine-tuning checkpoint
    print("\n=== Test 1: Full Fine-tuning Checkpoint ===")
    ckpt_path = tmp / "full_finetune_best.pt"
    make_checkpoint(ckpt_path, freeze_backbone=False, head_type="linear")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Train AUC (final epoch — no val split, see train.py): {checkpoint['train_auc']:.4f}")

    model = create_model("resnet18", num_classes=2, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("\u2713 Model state loaded successfully")

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits sample): {output[0]}")
    assert output.shape == (2, 2), "Output shape should be (batch_size, num_classes)"
    print("\u2713 Forward pass successful")

    # Test 2: Frozen backbone checkpoint
    print("\n=== Test 2: Frozen Backbone Checkpoint ===")
    ckpt_path = tmp / "frozen_backbone_best.pt"
    make_checkpoint(ckpt_path, freeze_backbone=True, head_type="mlp")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Train AUC (final epoch): {checkpoint['train_auc']:.4f}")

    model = create_model("resnet18", num_classes=2, freeze_backbone=True, head_type="mlp")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("\u2713 Model state loaded successfully")

    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 2), "Output shape should be (batch_size, num_classes)"
    print("\u2713 Forward pass successful")

    # Test 3: Verify softmax probabilities
    print("\n=== Test 3: Verify Softmax Probabilities ===")
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.softmax(logits, dim=1)
    print(f"Logits: {logits[0]}")
    print(f"Probabilities: {probs[0]}")
    print(f"Sum of probabilities: {probs[0].sum():.4f}")
    assert torch.allclose(probs[0].sum(), torch.tensor(1.0)), "Probabilities should sum to 1"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities should be in [0, 1]"
    print("\u2713 Probabilities valid")

    # Test 4: Check cancerous-class probability extraction (label=1)
    print("\n=== Test 4: Extract Cancerous-Class Probabilities ===")
    cancerous_probs = probs[:, 1].cpu().numpy()
    print(f"P(cancerous) for batch: {cancerous_probs}")
    assert len(cancerous_probs) == 2, "Should have one probability per sample"
    assert (cancerous_probs >= 0).all() and (cancerous_probs <= 1).all(), "Probs in [0, 1]"
    print("\u2713 Cancerous-class probability extraction successful")

print("\n" + "=" * 60)
print("ALL CHECKPOINT LOADING TESTS PASSED \u2713")
print("=" * 60)
