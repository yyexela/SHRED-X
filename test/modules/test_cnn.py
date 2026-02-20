"""Verify the decoder modules are working correctly."""

import pytest
import torch

from shredx.modules.cnn import CNNDecoder


@pytest.mark.parametrize("input_size", [3, 10])
@pytest.mark.parametrize("num_layer", [1, 3])
def test_cnn_decoder_forward_success(input_size, num_layer):
    # Test that the CNN decoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    dropout = 0.1

    out_dim = 10

    input_tensor = torch.randn(batch_size, 1, 1, input_size).float()
    input_tuple = (input_tensor, None)
    cnn = CNNDecoder(in_dim=input_size, out_dim=out_dim, n_layers=num_layer, dropout=dropout)
    output = cnn(input_tuple)
    assert output is not None
    assert output[0].shape == (batch_size, 1, 1, out_dim)
