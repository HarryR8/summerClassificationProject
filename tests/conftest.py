"""Pytest configuration and shared fixtures for sicm tests."""
import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def dummy_batch():
    """Create a dummy batch of images for testing."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def dummy_labels():
    """Create dummy labels for testing."""
    return torch.tensor([0, 1, 0, 1], dtype=torch.long)


@pytest.fixture
def num_classes():
    """Return number of classes for binary classification."""
    return 2


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed
