"""Verify the RNN modules are working correctly."""

import torch
import pytest
from shredx.modules.rnn import GRUEncoder, LSTMEncoder


@pytest.mark.parametrize("hidden_size", [3, 10])
@pytest.mark.parametrize("num_layer", [1, 3])
def test_gru_encoder_forward_success(hidden_size, num_layer):
    # Test that the GRU encoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    sequence_length = 10
    input_sizes = 6
    dropout = 0.1

    input_tensor = torch.randn(batch_size, sequence_length, input_sizes).float()

    gru = GRUEncoder(input_size=input_sizes, hidden_size=hidden_size, num_layers=num_layer, dropout=dropout)
    output = gru(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, 1, 1, hidden_size)


@pytest.mark.parametrize("hidden_size", [3, 10])
@pytest.mark.parametrize("num_layer", [1, 3])
def test_lstm_encoder_forward_success(hidden_size, num_layer):
    # Test that the LSTM encoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    sequence_length = 10
    input_sizes = 6
    dropout = 0.1

    input_tensor = torch.randn(batch_size, sequence_length, input_sizes).float()

    lstm = LSTMEncoder(input_size=input_sizes, hidden_size=hidden_size, num_layers=num_layer, dropout=dropout)
    output = lstm(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, 1, 1, hidden_size)
