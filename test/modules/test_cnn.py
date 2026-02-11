"""Verify the decoder modules are working correctly."""

import torch
from shredx.modules.cnn import CNNDecoder


def test_cnn_decoder_forward():
    # Test that the CNN decoder forward pass works correctly with varying input sizes and sequence lengths

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
            cnn = CNNDecoder(in_dim=input_size, out_dim=out_dim, n_layers=num_layer, dropout=dropout)
            output = cnn(input_tuple)
            assert output is not None
            assert output[0].shape == (batch_size, 1, 1, out_dim)
