"""Verify the RNN modules are working correctly."""

import pytest
import torch
from torch import nn

from shredx.modules.transformer import (
    MultiHeadSINDyAttention,
    SINDyAttentionSINDyLossTransformerEncoder,
    SINDyAttentionTransformerEncoder,
    SINDyLossTransformerEncoder,
    TransformerEncoder,
)


@pytest.mark.parametrize("input_size", [6, 7])
@pytest.mark.parametrize("num_layer", [1, 3])
@pytest.mark.parametrize("norm_first", [True, False])
def test_transformer_encoder_forward_success(input_size, num_layer, norm_first):
    # Test that the Transformer encoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    sequence_length = 10
    n_heads = 2 if input_size == 6 else 1

    input_tensor = torch.randn(batch_size, sequence_length, input_size).float()

    transformer = TransformerEncoder(
        d_model=input_size,
        n_heads=n_heads,
        num_layers=num_layer,
        dim_feedforward=input_size * 4,
        dropout=0.1,
        activation=nn.ReLU(),
        layer_norm_eps=1e-5,
        norm_first=norm_first,
        bias=True,
        input_length=sequence_length,
        hidden_size=input_size,
        device="cpu",
    )
    output = transformer(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, 1, sequence_length, input_size)


def test_sindy_attention_transformer_fail_n_heads():
    # Test that the SA Transformer encoder forward pass works correctly with varying input sizes and sequence lengths

    sequence_length = 10
    forecast_length = 3
    input_size = 7
    n_heads = 2
    num_layer = 1
    norm_first = False

    with pytest.raises(ValueError) as context:
        SINDyAttentionTransformerEncoder(
            d_model=input_size,
            n_heads=n_heads,
            forecast_length=forecast_length,
            num_layers=num_layer,
            dim_feedforward=input_size * 4,
            dropout=0.1,
            activation=nn.ReLU(),
            layer_norm_eps=1e-5,
            norm_first=norm_first,
            bias=True,
            strict_symmetry=True,
            input_length=sequence_length,
            hidden_size=input_size,
            device="cpu",
        )
    assert str(context.value) == "Embedding dim is not divisible by n_heads"


def test_multi_head_sindy_attention_fail_n_heads():
    """Covers ValueError in MultiHeadSINDyAttention when E_total is not divisible by n_heads."""
    with pytest.raises(ValueError) as context:
        MultiHeadSINDyAttention(
            E_q=7,
            E_k=7,
            E_v=7,
            E_total=7,
            n_heads=2,
            forecast_length=3,
            dropout=0.1,
            strict_symmetry=True,
            bias=True,
            dtype=None,
            device="cpu",
        )
    assert str(context.value) == "Embedding dim is not divisible by n_heads"


@pytest.mark.parametrize("input_size", [6, 7])
@pytest.mark.parametrize("num_layer", [1, 3])
@pytest.mark.parametrize("norm_first", [True, False])
def test_sindy_attention_transformer_encoder_forward_success(input_size, num_layer, norm_first):
    # Test that the SA Transformer encoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    sequence_length = 10
    forecast_length = 3
    n_heads = 2 if input_size == 6 else 1

    input_tensor = torch.randn(batch_size, sequence_length, input_size).float()

    transformer = SINDyAttentionTransformerEncoder(
        d_model=input_size,
        n_heads=n_heads,
        forecast_length=forecast_length,
        num_layers=num_layer,
        dim_feedforward=input_size * 4,
        dropout=0.1,
        activation=nn.ReLU(),
        layer_norm_eps=1e-5,
        norm_first=norm_first,
        bias=True,
        strict_symmetry=True,
        input_length=sequence_length,
        hidden_size=input_size,
        device="cpu",
    )
    output = transformer(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, forecast_length, sequence_length, input_size)

    forecast_length = 12
    transformer.set_forecast_length(forecast_length)

    output = transformer(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, forecast_length, sequence_length, input_size)


def test_sindy_attention_transformer_encoder_helpers_success():
    # Test that the SA Transformer encoder forward pass works correctly with varying input sizes and sequence lengths

    input_size = 6
    sequence_length = 10
    forecast_length = 3
    n_heads = 2
    num_layer = 1
    norm_first = False

    transformer = SINDyAttentionTransformerEncoder(
        d_model=input_size,
        n_heads=n_heads,
        forecast_length=forecast_length,
        num_layers=num_layer,
        dim_feedforward=input_size * 4,
        dropout=0.1,
        activation=nn.ReLU(),
        layer_norm_eps=1e-5,
        norm_first=norm_first,
        bias=True,
        strict_symmetry=True,
        input_length=sequence_length,
        hidden_size=input_size,
        device="cpu",
    )

    transformer.print_sindy_layer_coefficients()
    assert transformer.get_sindy_layer_coefficients_sum() != 0.0
    assert transformer.get_sindy_layer_coefficients_eigenvalues() is not None
    assert transformer.get_dense_sindy_coefficients() is not None

    transformer.threshold_sindy_layer_coefficients(1e32, verbose=True)
    assert transformer.get_sindy_layer_coefficients_sum() == 0.0


@pytest.mark.parametrize("input_size", [6, 7])
@pytest.mark.parametrize("num_layer", [1, 3])
@pytest.mark.parametrize("norm_first", [True, False])
def test_sindy_loss_transformer_encoder_helpers_success(input_size, num_layer, norm_first):
    # Test that the SASL Transformer encoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    sequence_length = 10
    n_heads = 2 if input_size == 6 else 1

    input_tensor = torch.randn(batch_size, sequence_length, input_size).float()

    transformer = SINDyLossTransformerEncoder(
        d_model=input_size,
        n_heads=n_heads,
        num_layers=num_layer,
        dim_feedforward=input_size * 4,
        dropout=0.1,
        activation=nn.ReLU(),
        layer_norm_eps=1e-5,
        norm_first=norm_first,
        bias=True,
        input_length=sequence_length,
        hidden_size=input_size,
        dt=0.1,
        sindy_loss_threshold=0.1,
        device="cpu",
    )
    output = transformer(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, 1, sequence_length, input_size)


@pytest.mark.parametrize("input_size", [6, 7])
@pytest.mark.parametrize("num_layer", [1, 3])
@pytest.mark.parametrize("norm_first", [True, False])
def test_sindy_attention_sindy_loss_transformer_encoder_forward_success(input_size, num_layer, norm_first):
    # Test that the SASL Transformer encoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    sequence_length = 10
    forecast_length = 3
    n_heads = 2 if input_size == 6 else 1

    input_tensor = torch.randn(batch_size, sequence_length, input_size).float()

    transformer = SINDyAttentionSINDyLossTransformerEncoder(
        d_model=input_size,
        n_heads=n_heads,
        forecast_length=forecast_length,
        num_layers=num_layer,
        dim_feedforward=input_size * 4,
        dropout=0.1,
        activation=nn.ReLU(),
        layer_norm_eps=1e-5,
        norm_first=norm_first,
        bias=True,
        strict_symmetry=True,
        input_length=sequence_length,
        hidden_size=input_size,
        dt=0.1,
        sindy_loss_threshold=0.1,
        device="cpu",
    )
    output = transformer(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, forecast_length, sequence_length, input_size)
