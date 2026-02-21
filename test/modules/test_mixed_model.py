"""Verify the MixedModel is working correctly."""

import pytest
import torch
from torch import nn

from shredx.models.mixed_model import MixedModel
from shredx.modules.cnn import CNNDecoder
from shredx.modules.mlp import MLPDecoder, MLPEncoder
from shredx.modules.transformer import SINDyAttentionSINDyLossTransformerEncoder


def test_mlp_mlp_mixed_model_forward_success():
    # Test that the MixedModel forward pass works correctly with varying input sizes and sequence lengths
    # Encoder and decoder are both MLPs

    batch_size = 2
    sequence_length = 10
    hidden_size = 3
    input_size = 6
    out_dim = 12
    dropout = 0.1
    num_layer = 1

    input_tensor = torch.randn(batch_size, sequence_length, input_size).float()

    mlp_encoder = MLPEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, dropout=dropout)
    mlp_decoder = MLPDecoder(in_dim=hidden_size, out_dim=out_dim, n_layers=num_layer, dropout=dropout)

    mixed_model = MixedModel(encoder=mlp_encoder, decoder=mlp_decoder)
    output = mixed_model(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, 1, 1, out_dim)
    print(output[1])
    assert output[1] is None


@pytest.mark.parametrize("norm_first", [True, False])
def test_transformer_cnn_mixed_model_forward_success(norm_first):
    # Test that the MixedModel forward pass works correctly with varying input sizes and sequence lengths
    # Encoder is a transformer, decoder is a CNN

    batch_size = 2
    sequence_length = 10
    input_size = 6
    out_dim = 60
    dropout = 0.1
    num_layer = 1
    n_heads = 1
    forecast_length = 3

    input_tensor = torch.randn(batch_size, sequence_length, input_size).float()

    sindy_attention_transformer_encoder = SINDyAttentionSINDyLossTransformerEncoder(
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

    cnn_decoder = CNNDecoder(in_dim=input_size, out_dim=out_dim, n_layers=num_layer, dropout=dropout)

    mixed_model = MixedModel(encoder=sindy_attention_transformer_encoder, decoder=cnn_decoder)
    output = mixed_model(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, forecast_length, sequence_length, out_dim)
    print(output[1])
    assert output[1]["sindy_loss"].shape == torch.Size([])
