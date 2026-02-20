"""Verify the RNN modules are working correctly."""

import torch
import pytest
from shredx.modules.rnn import (
    GRUEncoder,
    LSTMEncoder,
    MOEGRUEncoder,
    MOELSTMEncoder,
    SINDyLossGRUEncoder,
    SINDyLossLSTMEncoder,
)


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


@pytest.mark.parametrize("hidden_size", [3, 10])
@pytest.mark.parametrize("num_layer", [1, 3])
def test_gru_moe_forward_success(hidden_size, num_layer):
    # Test that the MLP encoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    sequence_length = 10
    input_sizes = 6
    forecast_length = 5
    strict_symmetry = True
    n_experts = 2
    dropout = 0.1

    input_tensor = torch.randn(batch_size, sequence_length, input_sizes).float()

    gru = MOEGRUEncoder(
        input_size=input_sizes,
        hidden_size=hidden_size,
        num_layers=num_layer,
        forecast_length=forecast_length,
        strict_symmetry=strict_symmetry,
        n_experts=n_experts,
        dropout=dropout,
    )
    output = gru(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, forecast_length, 1, hidden_size)


@pytest.mark.parametrize("hidden_size", [3, 10])
@pytest.mark.parametrize("num_layer", [1, 3])
def test_lstm_moe_forward_success(hidden_size, num_layer):
    # Test that the MLP encoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    sequence_length = 10
    input_sizes = 6
    forecast_length = 5
    strict_symmetry = True
    n_experts = 2
    dropout = 0.1

    input_tensor = torch.randn(batch_size, sequence_length, input_sizes).float()

    lstm = MOELSTMEncoder(
        input_size=input_sizes,
        hidden_size=hidden_size,
        num_layers=num_layer,
        forecast_length=forecast_length,
        strict_symmetry=strict_symmetry,
        n_experts=n_experts,
        dropout=dropout,
    )
    output = lstm(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, forecast_length, 1, hidden_size)


@pytest.mark.parametrize("hidden_size", [3, 10])
@pytest.mark.parametrize("num_layer", [1, 3])
def test_gru_sindy_loss_forward_success(hidden_size, num_layer):
    # Test that the MLP encoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 2
    sequence_length = 10
    input_sizes = 6
    dropout = 0.1

    input_tensor = torch.randn(batch_size, sequence_length, input_sizes).float()

    gru = SINDyLossGRUEncoder(
        input_size=input_sizes,
        hidden_size=hidden_size,
        num_layers=num_layer,
        dropout=dropout,
        poly_order=2,
        dt=0.1,
        sindy_loss_threshold=0.1,
    )
    output = gru(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, 1, 1, hidden_size)
    assert output[1].shape == torch.Size([])
    gru.thresholding(threshold=0.1)
    gru.thresholding(threshold=None)


@pytest.mark.parametrize("hidden_size", [3, 10])
@pytest.mark.parametrize("num_layer", [1, 3])
def test_lstm_sindy_loss_forward_success(hidden_size, num_layer):
    # Test that the MLP encoder forward pass works correctly with varying input sizes and sequence lengths

    batch_size = 5
    sequence_length = 10
    input_sizes = 6
    dropout = 0.1

    input_tensor = torch.randn(batch_size, sequence_length, input_sizes).float()

    lstm = SINDyLossLSTMEncoder(
        input_size=input_sizes,
        hidden_size=hidden_size,
        num_layers=num_layer,
        dropout=dropout,
        poly_order=2,
        dt=0.1,
        sindy_loss_threshold=0.1,
    )
    output = lstm(input_tensor)
    assert output is not None
    assert output[0].shape == (batch_size, 1, 1, hidden_size)
    assert output[1].shape == torch.Size([])
    lstm.thresholding(threshold=0.1)
    lstm.thresholding(threshold=None)
