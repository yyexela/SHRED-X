"""Verify the SINDy layer module is working correctly."""

import torch
import pytest
from shredx.modules.sindy_layer import SindyLayer


@pytest.mark.parametrize("strict_symmetry", [True, False])
def test_sindy_layer_forward_success(strict_symmetry):
    # Test that the SINDy layer forward pass works correctly

    batch_size = 2
    hidden_size = 10
    forecast_length = 10

    input_tensor = torch.randn(batch_size, hidden_size).float()
    sindy_layer = SindyLayer(hidden_size=hidden_size, forecast_length=forecast_length, strict_symmetry=strict_symmetry)
    output = sindy_layer(input_tensor)
    assert output is not None
    assert output.shape == (batch_size, forecast_length, hidden_size)


@pytest.mark.parametrize("strict_symmetry", [True, False])
def test_get_raw_sindy_coefficients_success(strict_symmetry):
    # Test that the get_raw_sindy_coefficients method works correctly

    hidden_size = 10
    sindy_layer = SindyLayer(hidden_size=hidden_size, forecast_length=10, strict_symmetry=strict_symmetry)
    raw_sindy_coefficients = sindy_layer.get_raw_sindy_coefficients()
    assert raw_sindy_coefficients is not None
    if strict_symmetry:
        assert raw_sindy_coefficients.shape == torch.Size([(hidden_size * (hidden_size + 1)) // 2])
    else:
        assert raw_sindy_coefficients.shape == torch.Size([hidden_size, hidden_size])


@pytest.mark.parametrize("strict_symmetry", [True, False])
def test_set_raw_sindy_coefficients_success(strict_symmetry):
    # Test that the get_raw_sindy_coefficients method works correctly

    hidden_size = 10
    sindy_layer = SindyLayer(hidden_size=hidden_size, forecast_length=10, strict_symmetry=strict_symmetry)
    if strict_symmetry:
        new_sindy_coefficients = torch.randn((hidden_size * (hidden_size + 1)) // 2)
    else:
        new_sindy_coefficients = torch.randn(hidden_size, hidden_size)
    sindy_layer.set_raw_sindy_coefficients(new_sindy_coefficients)
    assert torch.allclose(sindy_layer.get_raw_sindy_coefficients(), new_sindy_coefficients)


@pytest.mark.parametrize("strict_symmetry", [True, False])
def test_get_eigenvalues_success(strict_symmetry):
    # Test that the get_eigenvalues method works correctly

    hidden_size = 10
    sindy_layer = SindyLayer(hidden_size=hidden_size, forecast_length=10, strict_symmetry=strict_symmetry)
    eigenvalues = sindy_layer.get_eigenvalues()
    assert eigenvalues is not None
    assert eigenvalues.shape == torch.Size([hidden_size])
