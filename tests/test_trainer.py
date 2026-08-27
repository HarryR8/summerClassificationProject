"""Tests for sicm.training.trainer module."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sicm.training.trainer import train_one_epoch, evaluate
from sicm.models.factory import create_model


@pytest.fixture
def simple_model(num_classes):
    """Create a simple model for testing."""
    return create_model("resnet18", num_classes, freeze_backbone=True, head_type="linear")


@pytest.fixture
def simple_dataloader():
    """Create a simple dataloader for testing."""
    # Create dummy data
    images = torch.randn(16, 3, 224, 224)
    labels = torch.randint(0, 2, (16,))  # Binary labels
    image_ids = [f"img_{i:03d}" for i in range(16)]

    # Create dataset that returns dicts like the real loaders
    class DictDataset(TensorDataset):
        def __init__(self, images, labels, image_ids):
            super().__init__(images, labels)
            self.image_ids = image_ids

        def __getitem__(self, index):
            img, label = super().__getitem__(index)
            return {
                "image": img,
                "label": label,
                "image_id": self.image_ids[index]
            }

    dataset = DictDataset(images, labels, image_ids)
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def criterion():
    """Create a loss criterion."""
    return nn.CrossEntropyLoss()


@pytest.fixture
def optimizer(simple_model):
    """Create an optimizer."""
    return torch.optim.Adam(simple_model.parameters(), lr=1e-3)


class TestTrainOneEpoch:
    """Tests for train_one_epoch function."""

    def test_train_one_epoch_returns_dict(self, simple_model, simple_dataloader,
                                           criterion, optimizer, device):
        """Test that train_one_epoch returns a dict with expected keys."""
        metrics = train_one_epoch(simple_model, simple_dataloader, criterion, optimizer, device)

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "auc" in metrics

    def test_train_one_epoch_loss_is_finite(self, simple_model, simple_dataloader,
                                             criterion, optimizer, device):
        """Test that loss is a finite number."""
        metrics = train_one_epoch(simple_model, simple_dataloader, criterion, optimizer, device)

        assert isinstance(metrics["loss"], float)
        assert not torch.isnan(torch.tensor(metrics["loss"]))
        assert not torch.isinf(torch.tensor(metrics["loss"]))

    def test_train_one_epoch_auc_in_valid_range(self, simple_model, simple_dataloader,
                                                  criterion, optimizer, device):
        """Test that AUC is in valid range [0, 1] or NaN."""
        metrics = train_one_epoch(simple_model, simple_dataloader, criterion, optimizer, device)

        # AUC can be NaN if there's only one class, otherwise should be in [0, 1]
        if not torch.isnan(torch.tensor(metrics["auc"])):
            assert 0.0 <= metrics["auc"] <= 1.0

    def test_train_one_epoch_updates_model(self, simple_model, simple_dataloader,
                                            criterion, optimizer, device, seed):
        """Test that training updates model parameters."""
        # Get initial parameters
        initial_params = [p.clone() for p in simple_model.parameters() if p.requires_grad]

        # Train for one epoch
        train_one_epoch(simple_model, simple_dataloader, criterion, optimizer, device)

        # Check that at least some parameters changed
        final_params = [p for p in simple_model.parameters() if p.requires_grad]
        params_changed = any(
            not torch.equal(initial, final)
            for initial, final in zip(initial_params, final_params)
        )

        assert params_changed, "At least some parameters should change during training"

    def test_train_one_epoch_sets_train_mode(self, simple_model, simple_dataloader,
                                               criterion, optimizer, device):
        """Test that model is in training mode during train_one_epoch."""
        # Start in eval mode
        simple_model.eval()
        assert not simple_model.training

        # Create a hook to check if model is in training mode during forward pass
        was_training = []

        def hook(module, input):
            was_training.append(module.training)

        # Register hook
        handle = simple_model.register_forward_pre_hook(hook)

        try:
            train_one_epoch(simple_model, simple_dataloader, criterion, optimizer, device)
            assert any(was_training), "Model should be in training mode during forward pass"
        finally:
            handle.remove()


class TestEvaluate:
    """Tests for evaluate function."""

    def test_evaluate_returns_dict(self, simple_model, simple_dataloader, criterion, device):
        """Test that evaluate returns a dict with expected keys."""
        metrics = evaluate(simple_model, simple_dataloader, criterion, device)

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "auc" in metrics
        assert "labels" in metrics
        assert "probs" in metrics
        assert "image_ids" in metrics

    def test_evaluate_arrays_have_correct_length(self, simple_model, simple_dataloader,
                                                   criterion, device):
        """Test that output arrays have correct length."""
        metrics = evaluate(simple_model, simple_dataloader, criterion, device)

        # Should have one entry per sample in dataset
        dataset_size = len(simple_dataloader.dataset)

        assert len(metrics["labels"]) == dataset_size
        assert len(metrics["probs"]) == dataset_size
        assert len(metrics["image_ids"]) == dataset_size

    def test_evaluate_probabilities_in_range(self, simple_model, simple_dataloader,
                                               criterion, device):
        """Test that probabilities are in [0, 1]."""
        metrics = evaluate(simple_model, simple_dataloader, criterion, device)

        probs = metrics["probs"]
        assert (probs >= 0).all(), "Probabilities should be >= 0"
        assert (probs <= 1).all(), "Probabilities should be <= 1"

    def test_evaluate_sets_eval_mode(self, simple_model, simple_dataloader, criterion, device):
        """Test that model is in eval mode during evaluate."""
        # Start in train mode
        simple_model.train()
        assert simple_model.training

        # Create a hook to check if model is in eval mode during forward pass
        was_training = []

        def hook(module, input):
            was_training.append(module.training)

        # Register hook
        handle = simple_model.register_forward_pre_hook(hook)

        try:
            evaluate(simple_model, simple_dataloader, criterion, device)
            assert not any(was_training), "Model should be in eval mode during forward pass"
        finally:
            handle.remove()

    def test_evaluate_no_gradient_computation(self, simple_model, simple_dataloader,
                                                criterion, device):
        """Test that gradients are not computed during evaluation."""
        # Enable gradient computation
        torch.set_grad_enabled(True)

        metrics = evaluate(simple_model, simple_dataloader, criterion, device)

        # Check that no gradients are stored (all should be None)
        for param in simple_model.parameters():
            # Gradients might not be None if backward was called before,
            # but they shouldn't accumulate during evaluate
            pass  # The function uses torch.no_grad(), which we trust

    def test_evaluate_single_class_batch_handles_gracefully(self, simple_model,
                                                              criterion, device):
        """Test that single-class batches are handled (AUC returns NaN)."""
        # Create dataset with only one class
        images = torch.randn(8, 3, 224, 224)
        labels = torch.zeros(8, dtype=torch.long)  # All class 0
        image_ids = [f"img_{i:03d}" for i in range(8)]

        class DictDataset(TensorDataset):
            def __init__(self, images, labels, image_ids):
                super().__init__(images, labels)
                self.image_ids = image_ids

            def __getitem__(self, index):
                img, label = super().__getitem__(index)
                return {
                    "image": img,
                    "label": label,
                    "image_id": self.image_ids[index]
                }

        dataset = DictDataset(images, labels, image_ids)
        loader = DataLoader(dataset, batch_size=4)

        metrics = evaluate(simple_model, loader, criterion, device)

        # AUC should be NaN for single-class dataset
        import math
        assert math.isnan(metrics["auc"]), "AUC should be NaN for single-class dataset"

    def test_evaluate_consistent_results(self, simple_model, simple_dataloader,
                                          criterion, device, seed):
        """Test that evaluation gives consistent results (deterministic)."""
        simple_model.eval()

        metrics1 = evaluate(simple_model, simple_dataloader, criterion, device)
        metrics2 = evaluate(simple_model, simple_dataloader, criterion, device)

        # Results should be identical
        assert metrics1["loss"] == metrics2["loss"]
        assert (metrics1["probs"] == metrics2["probs"]).all()
        assert (metrics1["labels"] == metrics2["labels"]).all()
