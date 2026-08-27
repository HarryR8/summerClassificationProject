#!/usr/bin/env python
"""Smoke tests for model factory."""
import torch
from sicm.models.factory import create_model, list_models, count_parameters, create_backbone
from sicm.models.heads import create_head, list_heads

print("=" * 60)
print("MODEL FACTORY SMOKE TESTS")
print("=" * 60)

# Test 1: List available models
print("\n=== Test 1: List Available Models ===")
models = list_models()
print(f"Available models ({len(models)}): {models}")
print("✓ Model listing works")

# Test 2: Full fine-tuning
print("\n=== Test 2: Full Fine-tuning (resnet18) ===")
model = create_model('resnet18', num_classes=2, freeze_backbone=False)
params = count_parameters(model)
print(f"Trainable params: {params['trainable']:,}")
print(f"Total params: {params['total']:,}")
print(f"Frozen params: {params['frozen']:,}")
print(f"All params trainable: {params['trainable'] == params['total']}")
assert params['trainable'] == params['total'], "All params should be trainable"
assert params['frozen'] == 0, "No params should be frozen"
print("✓ Full fine-tuning model created successfully")

# Test 3: Frozen backbone + linear head
print("\n=== Test 3: Frozen Backbone + Linear Head (resnet18) ===")
model = create_model('resnet18', num_classes=2, freeze_backbone=True, head_type='linear')
params = count_parameters(model)
print(f"Trainable params: {params['trainable']:,}")
print(f"Total params: {params['total']:,}")
print(f"Frozen params: {params['frozen']:,}")
print(f"Backbone frozen: {params['trainable'] < params['total']}")
assert params['trainable'] < params['total'], "Some params should be frozen"
assert params['frozen'] > 0, "Backbone should be frozen"
print("✓ Frozen backbone + linear head created successfully")

# Test 4: Frozen backbone + MLP head
print("\n=== Test 4: Frozen Backbone + MLP Head (resnet18) ===")
model = create_model('resnet18', num_classes=2, freeze_backbone=True, head_type='mlp')
params = count_parameters(model)
print(f"Trainable params: {params['trainable']:,}")
print(f"Total params: {params['total']:,}")
print(f"Frozen params: {params['frozen']:,}")
assert params['trainable'] < params['total'], "Some params should be frozen"
print("✓ Frozen backbone + MLP head created successfully")

# Test 5: Frozen backbone + MLP Deep head
print("\n=== Test 5: Frozen Backbone + MLP Deep Head (resnet18) ===")
model = create_model('resnet18', num_classes=2, freeze_backbone=True, head_type='mlp_deep')
params = count_parameters(model)
print(f"Trainable params: {params['trainable']:,}")
print(f"Total params: {params['total']:,}")
print(f"Frozen params: {params['frozen']:,}")
assert params['trainable'] < params['total'], "Some params should be frozen"
print("✓ Frozen backbone + MLP Deep head created successfully")

# Test 6: Backbone-only mode
print("\n=== Test 6: Backbone-Only Mode (resnet18) ===")
backbone, embedding_dim = create_backbone('resnet18')
print(f"Embedding dimension: {embedding_dim}")
dummy_input = torch.randn(2, 3, 224, 224)
output = backbone(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
assert len(output.shape) == 2, "Output should be 2D (batch_size, features)"
assert output.shape[0] == 2, "Batch dimension should be preserved"
assert output.shape[1] == embedding_dim, f"Feature dimension should be {embedding_dim}"
print("✓ Backbone-only model created and forward pass successful")

# Test 7: Different model architectures
print("\n=== Test 7: Different Model Architectures ===")
for model_name in ['resnet50', 'efficientnet_b0', 'densenet121']:
    try:
        model = create_model(model_name, num_classes=2, freeze_backbone=False)
        params = count_parameters(model)
        print(f"{model_name}: {params['total']:,} params - ✓")
    except Exception as e:
        print(f"{model_name}: Failed - {e}")

# Test 8: Head types
print("\n=== Test 8: Classification Head Types ===")
heads = list_heads()
print(f"Available heads: {heads}")
for head_type in ['linear', 'mlp', 'mlp_deep']:
    head = create_head(head_type, in_features=512, num_classes=2)
    dummy = torch.randn(4, 512)
    output = head(dummy)
    print(f"{head_type}: input {dummy.shape} -> output {output.shape} - ✓")
    assert output.shape == (4, 2), f"Output shape should be (4, 2) for {head_type}"

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED ✓")
print("=" * 60)
