"""Verify the MLP modules are working correctly."""

import pytest
import torch

from shredx.modules.mlp import MLPDecoder, MLPEncoder, MOEMLPEncoder, SINDyLossMLPEncoder


@pytest.mark.parametrize("hidden_size", [3, 10])
@pytest.mark.parametrize("num_layer", [1, 3])
def test_mlp_encoder_forward_success(hidden_size, num_layer):
    # Test that the MLP encoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    sequence_length = 10
    input_sizes = 6
    dropout = 0.1

    input_tensor = torch.randn(batch_size, sequence_length, input_sizes).float()

    mlp = MLPEncoder(input_size=input_sizes, hidden_size=hidden_size, num_layers=num_layer, dropout=dropout)
    output = mlp(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, 1, 1, hidden_size)


@pytest.mark.parametrize("input_size", [3, 10])
@pytest.mark.parametrize("num_layer", [1, 3])
def test_mlp_decoder_forward_success(input_size, num_layer):
    # Test that the MLP decoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    dropout = 0.1

    out_dim = 10

    input_tensor = torch.randn(batch_size, 1, 1, input_size).float()
    input_tuple = (input_tensor, None)
    mlp = MLPDecoder(in_dim=input_size, out_dim=out_dim, n_layers=num_layer, dropout=dropout)
    output = mlp(input_tuple)
    assert output is not None
    assert output[0].shape == (batch_size, 1, 1, out_dim)


@pytest.mark.parametrize("hidden_size", [3, 10])
@pytest.mark.parametrize("num_layer", [1, 3])
def test_mlp_moe_forward_success(hidden_size, num_layer):
    # Test that the MLP encoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    sequence_length = 10
    input_sizes = 6
    forecast_length = 5
    strict_symmetry = True
    n_experts = 2
    dropout = 0.1

    input_tensor = torch.randn(batch_size, sequence_length, input_sizes).float()

    mlp = MOEMLPEncoder(
        input_size=input_sizes,
        hidden_size=hidden_size,
        num_layers=num_layer,
        forecast_length=forecast_length,
        strict_symmetry=strict_symmetry,
        n_experts=n_experts,
        dropout=dropout,
    )
    output = mlp(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, forecast_length, 1, hidden_size)


@pytest.mark.parametrize("hidden_size", [3, 10])
@pytest.mark.parametrize("num_layer", [1, 3])
def test_mlp_sindy_loss_forward_success(hidden_size, num_layer):
    # Test that the MLP encoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    sequence_length = 10
    input_sizes = 6
    dropout = 0.1

    input_tensor = torch.randn(batch_size, sequence_length, input_sizes).float()

    mlp = SINDyLossMLPEncoder(
        input_size=input_sizes,
        hidden_size=hidden_size,
        num_layers=num_layer,
        dropout=dropout,
        poly_order=2,
        dt=0.1,
        sindy_loss_threshold=0.1,
    )
    output = mlp(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, 1, 1, hidden_size)
    assert output[1].shape == torch.Size([])
    mlp.thresholding(threshold=0.1)
    mlp.thresholding(threshold=None)
