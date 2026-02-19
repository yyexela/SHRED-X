import torch
import pytest
from shredx.modules.rnn import MOELSTMEncoder


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
    verbose = True

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

    # Run moe_mixin helper methods
    forecast_length = 10
    lstm.set_forecast_length(forecast_length)

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

    lstm.print_sindy_layer_coefficients()
    lstm.get_sindy_layer_coefficients_eigenvalues()
    lstm.get_sindy_layer_coefficients_sum()
    assert lstm.get_sindy_layer_coefficients_sum() != 0.0
    assert lstm.get_sindy_layer_coefficients_eigenvalues() is not None

    lstm.threshold_sindy_layer_coefficients(threshold=1e32, verbose=verbose)
    assert lstm.get_sindy_layer_coefficients_sum() == 0.0
