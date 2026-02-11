"""Verify the MLP modules are working correctly."""

import torch
from shredx.modules.mlp import MLPEncoder, MLPDecoder


def test_mlp_encoder_forward():
    # Test that the MLP encoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    sequence_length = 10
    input_sizes = 6
    hidden_sizes = [3, 10]
    num_layers = [1, 3]
    dropout = 0.1

    input_tensor = torch.randn(batch_size, sequence_length, input_sizes).float()

    for hidden_size in hidden_sizes:
        for num_layer in num_layers:
            mlp = MLPEncoder(input_size=input_sizes, hidden_size=hidden_size, num_layers=num_layer, dropout=dropout)
            output = mlp(input_tensor)
            assert output is not None
            assert output[0].shape == (batch_size, 1, 1, hidden_size)


def test_mlp_decoder_forward():
    # Test that the MLP decoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    input_sizes = 6
    input_sizes = [3, 10]
    num_layers = [1, 3]
    dropout = 0.1

    out_dim = 10

    for input_size in input_sizes:
        input_tensor = torch.randn(batch_size, 1, 1, input_size).float()
        input_tuple = (input_tensor, None)
        for num_layer in num_layers:
            mlp = MLPDecoder(in_dim=input_size, out_dim=out_dim, n_layers=num_layer, dropout=dropout)
            output = mlp(input_tuple)
            assert output is not None
            assert output[0].shape == (batch_size, 1, 1, out_dim)
