"""Tests for sicm.models.factory module."""
import pytest
import torch
from sicm.models.factory import (
    create_model,
    create_backbone,
    list_models,
    count_parameters,
    freeze_module,
)


class TestListModels:
    """Tests for list_models function."""

    def test_list_models_returns_list(self):
        """Test that list_models returns a list."""
        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_list_models_contains_expected_models(self):
        """Test that common models are in the registry."""
        models = list_models()
        expected = ["resnet18", "resnet50", "efficientnet_b0", "densenet121"]
        for model_name in expected:
            assert model_name in models, f"{model_name} should be in registry"


class TestCreateModel:
    """Tests for create_model function."""

    @pytest.mark.parametrize("model_name", ["resnet18", "resnet50", "efficientnet_b0"])
    def test_create_model_full_finetune(self, model_name, num_classes):
        """Test creating models in full fine-tuning mode."""
        model = create_model(model_name, num_classes, freeze_backbone=False)
        params = count_parameters(model)

        assert params["total"] > 0, "Model should have parameters"
        assert params["trainable"] == params["total"], "All params should be trainable"
        assert params["frozen"] == 0, "No params should be frozen"

    @pytest.mark.parametrize("head_type", ["linear", "mlp", "mlp_deep"])
    def test_create_model_frozen_backbone(self, num_classes, head_type):
        """Test creating models with frozen backbone."""
        model = create_model("resnet18", num_classes, freeze_backbone=True, head_type=head_type)
        params = count_parameters(model)

        assert params["trainable"] < params["total"], "Some params should be frozen"
        assert params["frozen"] > 0, "Backbone should be frozen"
        assert params["trainable"] + params["frozen"] == params["total"]

    def test_create_model_forward_pass(self, num_classes, dummy_batch):
        """Test forward pass through created model."""
        model = create_model("resnet18", num_classes, freeze_backbone=False)
        model.eval()

        with torch.no_grad():
            output = model(dummy_batch)

        assert output.shape == (dummy_batch.shape[0], num_classes)
        assert not torch.isnan(output).any(), "Output should not contain NaN"

    def test_create_model_invalid_name(self, num_classes):
        """Test that invalid model name raises error."""
        with pytest.raises(KeyError, match="Unknown model"):
            create_model("invalid_model_name", num_classes)


class TestCreateBackbone:
    """Tests for create_backbone function."""

    def test_create_backbone_returns_tuple(self):
        """Test that create_backbone returns (model, embedding_dim)."""
        result = create_backbone("resnet18")
        assert isinstance(result, tuple)
        assert len(result) == 2

        backbone, embedding_dim = result
        assert isinstance(embedding_dim, int)
        assert embedding_dim > 0

    def test_create_backbone_forward_pass(self, dummy_batch):
        """Test forward pass through backbone."""
        backbone, embedding_dim = create_backbone("resnet18")
        backbone.eval()

        with torch.no_grad():
            output = backbone(dummy_batch)

        assert output.shape == (dummy_batch.shape[0], embedding_dim)
        assert len(output.shape) == 2, "Output should be 2D (batch, features)"

    @pytest.mark.parametrize("model_name", ["resnet18", "efficientnet_b0"])
    def test_create_backbone_different_models(self, model_name, dummy_batch):
        """Test backbone creation for different model architectures."""
        backbone, embedding_dim = create_backbone(model_name)
        backbone.eval()

        with torch.no_grad():
            output = backbone(dummy_batch)

        assert output.shape[1] == embedding_dim


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_parameters_structure(self):
        """Test that count_parameters returns correct dict structure."""
        model = create_model("resnet18", num_classes=2, freeze_backbone=False)
        params = count_parameters(model)

        assert isinstance(params, dict)
        assert "total" in params
        assert "trainable" in params
        assert "frozen" in params

    def test_count_parameters_values(self):
        """Test that parameter counts are consistent."""
        model = create_model("resnet18", num_classes=2, freeze_backbone=False)
        params = count_parameters(model)

        assert params["total"] == params["trainable"] + params["frozen"]
        assert params["total"] > 0
        assert params["trainable"] >= 0
        assert params["frozen"] >= 0

    def test_count_parameters_frozen_backbone(self):
        """Test parameter counting for frozen backbone."""
        model = create_model("resnet18", num_classes=2, freeze_backbone=True)
        params = count_parameters(model)

        # With frozen backbone, most params should be frozen
        assert params["frozen"] > params["trainable"]


class TestFreezeModule:
    """Tests for freeze_module function."""

    def test_freeze_module_sets_requires_grad_false(self):
        """Test that freeze_module sets requires_grad to False."""
        model = create_model("resnet18", num_classes=2, freeze_backbone=False)

        # Initially all params are trainable
        params_before = count_parameters(model)
        assert params_before["frozen"] == 0

        # Freeze the model
        freeze_module(model)

        # Now all params should be frozen
        params_after = count_parameters(model)
        assert params_after["trainable"] == 0
        assert params_after["frozen"] == params_before["total"]

    def test_freeze_module_idempotent(self):
        """Test that freezing twice has same effect as once."""
        model = create_model("resnet18", num_classes=2, freeze_backbone=False)

        freeze_module(model)
        params_first = count_parameters(model)

        freeze_module(model)
        params_second = count_parameters(model)

        assert params_first == params_second
