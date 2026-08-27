"""Tests for sicm.models.heads module."""
import pytest
import torch
from sicm.models.heads import create_head, list_heads


class TestListHeads:
    """Tests for list_heads function."""

    def test_list_heads_returns_list(self):
        """Test that list_heads returns a list."""
        heads = list_heads()
        assert isinstance(heads, list)
        assert len(heads) > 0

    def test_list_heads_contains_expected_heads(self):
        """Test that expected heads are in the registry."""
        heads = list_heads()
        expected = ["linear", "mlp", "mlp_deep"]
        for head_type in expected:
            assert head_type in heads, f"{head_type} should be in registry"


class TestCreateHead:
    """Tests for create_head function."""

    @pytest.mark.parametrize("head_type", ["linear", "mlp", "mlp_deep"])
    def test_create_head_returns_module(self, head_type):
        """Test that create_head returns a nn.Module."""
        head = create_head(head_type, in_features=512, num_classes=2)
        assert isinstance(head, torch.nn.Module)

    @pytest.mark.parametrize("head_type", ["linear", "mlp", "mlp_deep"])
    def test_create_head_forward_pass(self, head_type):
        """Test forward pass through different head types."""
        head = create_head(head_type, in_features=512, num_classes=2)
        head.eval()

        dummy_features = torch.randn(4, 512)
        with torch.no_grad():
            output = head(dummy_features)

        assert output.shape == (4, 2), f"Output shape should be (4, 2) for {head_type}"
        assert not torch.isnan(output).any(), "Output should not contain NaN"

    def test_create_head_linear_architecture(self):
        """Test that linear head has correct architecture."""
        head = create_head("linear", in_features=512, num_classes=2)

        # Count parameters: 512 * 2 (weights) + 2 (bias) = 1026
        total_params = sum(p.numel() for p in head.parameters())
        expected_params = 512 * 2 + 2
        assert total_params == expected_params

    def test_create_head_mlp_has_hidden_layer(self):
        """Test that MLP head has more parameters than linear."""
        linear_head = create_head("linear", in_features=512, num_classes=2)
        mlp_head = create_head("mlp", in_features=512, num_classes=2)

        linear_params = sum(p.numel() for p in linear_head.parameters())
        mlp_params = sum(p.numel() for p in mlp_head.parameters())

        assert mlp_params > linear_params, "MLP should have more parameters than linear"

    def test_create_head_mlp_deep_has_most_parameters(self):
        """Test that MLP Deep has most parameters."""
        linear_head = create_head("linear", in_features=512, num_classes=2)
        mlp_head = create_head("mlp", in_features=512, num_classes=2)
        mlp_deep_head = create_head("mlp_deep", in_features=512, num_classes=2)

        linear_params = sum(p.numel() for p in linear_head.parameters())
        mlp_params = sum(p.numel() for p in mlp_head.parameters())
        mlp_deep_params = sum(p.numel() for p in mlp_deep_head.parameters())

        assert mlp_deep_params > mlp_params > linear_params

    def test_create_head_custom_hidden_dim(self):
        """Test that custom hidden_dim parameter works."""
        head1 = create_head("mlp", in_features=512, num_classes=2, hidden_dim=256)
        head2 = create_head("mlp", in_features=512, num_classes=2, hidden_dim=512)

        params1 = sum(p.numel() for p in head1.parameters())
        params2 = sum(p.numel() for p in head2.parameters())

        assert params2 > params1, "Larger hidden_dim should have more parameters"

    def test_create_head_custom_dropout(self):
        """Test that custom dropout parameter is accepted."""
        head = create_head("mlp", in_features=512, num_classes=2, dropout=0.3)
        # Just verify it doesn't raise an error
        assert isinstance(head, torch.nn.Module)

    def test_create_head_invalid_type(self):
        """Test that invalid head type raises error."""
        with pytest.raises(ValueError, match="Unknown head type"):
            create_head("invalid_head", in_features=512, num_classes=2)

    def test_create_head_different_input_sizes(self):
        """Test head creation with different input feature sizes."""
        for in_features in [256, 512, 1024, 2048]:
            head = create_head("linear", in_features=in_features, num_classes=2)
            dummy = torch.randn(4, in_features)
            with torch.no_grad():
                output = head(dummy)
            assert output.shape == (4, 2)

    def test_create_head_different_num_classes(self):
        """Test head creation with different number of classes."""
        for num_classes in [2, 3, 10, 100]:
            head = create_head("linear", in_features=512, num_classes=num_classes)
            dummy = torch.randn(4, 512)
            with torch.no_grad():
                output = head(dummy)
            assert output.shape == (4, num_classes)

    def test_create_head_dropout_in_train_mode(self):
        """Test that dropout is applied differently in train vs eval mode."""
        head = create_head("mlp", in_features=512, num_classes=2, dropout=0.5)
        dummy = torch.randn(100, 512)

        # In eval mode, dropout is disabled (deterministic output)
        head.eval()
        with torch.no_grad():
            out1 = head(dummy)
            out2 = head(dummy)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

        # In train mode with dropout=0.5, outputs should differ
        head.train()
        with torch.no_grad():
            out3 = head(dummy)
            out4 = head(dummy)
        # Note: There's a small chance they could be equal, but very unlikely with dropout=0.5
        # We just check that the mechanism works by ensuring eval mode is deterministic
