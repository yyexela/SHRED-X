"""Verify the scaling functions are working correctly."""

import torch
import pytest
from shredx.utils.scaling import min_max_scale, inverse_min_max_scale


def test_min_max_scale_success_1():
    # Test that min max scaling works without passing a scaler (scaled data to [0,1])
    input_tensor = torch.tensor([[0, 0, 1], [0, 2, 0], [4, 0, 0]], dtype=torch.float32)

    expected_output = torch.tensor([[0, 0, 0.25], [0, 0.5, 0], [1.0, 0, 0]], dtype=torch.float32)
    expected_scaler = (0.0, 4.0)

    output_tensor, scaler = min_max_scale(input_tensor, feature_range=(0, 1))

    assert torch.allclose(output_tensor, expected_output)
    assert scaler == expected_scaler


def test_min_max_scale_success_2():
    # Test that min max scaling works with passing a scaler (scales data to [0,2])
    input_tensor = torch.tensor([[0, 0, 1], [0, 2, 0], [4, 0, 0]], dtype=torch.float32)
    input_scaler = (0.0, 2.0)

    expected_output = torch.tensor([[0, 0, 0.5], [0, 1.0, 0], [2.0, 0, 0]], dtype=torch.float32)
    output_tensor, scaler = min_max_scale(input_tensor, feature_range=(0, 1), scaler=input_scaler)

    assert torch.allclose(output_tensor, expected_output)
    assert scaler == input_scaler


def test_min_max_scale_success_3():
    # Test that min max scaling works when data is all the same
    input_tensor = torch.tensor([[1, 1]], dtype=torch.float32)
    expected_scaler = (1.0, 1.0)

    expected_output = torch.tensor([[0, 0]], dtype=torch.float32)
    output_tensor, scaler = min_max_scale(input_tensor, feature_range=(0, 1))

    assert torch.allclose(output_tensor, expected_output)
    assert scaler == expected_scaler


def test_min_max_scale_fail_1():
    # Test that min max scaling raises an error if the input tensor has less than two elements
    input_tensor = torch.tensor([0], dtype=torch.float32)
    with pytest.raises(ValueError):
        min_max_scale(input_tensor, feature_range=(0, 1))


def test_min_max_scale_success_fail_2():
    # Test that inverse min max fails when feature_range is not a tuple of two elements
    input_tensor = torch.tensor([[0, 0, 0.25], [0, 0.5, 0], [1.0, 0, 0]], dtype=torch.float32)

    feature_range = (0.0, 4.0, 2.0)
    with pytest.raises(ValueError):
        min_max_scale(input_tensor, feature_range=feature_range)


def test_inverse_min_max_scale_success():
    # Test that inverse min max scaling works
    input_tensor = torch.tensor([[0, 0, 0.25], [0, 0.5, 0], [1.0, 0, 0]], dtype=torch.float32)
    input_scaler = (0.0, 4.0)
    expected_output = torch.tensor([[0, 0, 1], [0, 2, 0], [4, 0, 0]], dtype=torch.float32)

    output_tensor = inverse_min_max_scale(input_tensor, input_scaler, feature_range=(0, 1))
    assert torch.allclose(output_tensor, expected_output)


def test_inverse_min_max_scale_success_fail_1():
    # Test that inverse min max scaling works
    input_tensor = torch.tensor([[0, 0, 0.25], [0, 0.5, 0], [1.0, 0, 0]], dtype=torch.float32)
    input_scaler = (0.0, 4.0)
    expected_output = torch.tensor([[0, 0, 1], [0, 2, 0], [4, 0, 0]], dtype=torch.float32)

    output_tensor = inverse_min_max_scale(input_tensor, input_scaler, feature_range=(0, 1))
    assert torch.allclose(output_tensor, expected_output)


def test_inverse_min_max_scale_success_fail_2():
    # Test that inverse min max fails when input_scaler is not 2 elements
    input_tensor = torch.tensor([[0, 0, 0.25], [0, 0.5, 0], [1.0, 0, 0]], dtype=torch.float32)

    input_scaler = (0.0, 4.0, 2.0)
    with pytest.raises(ValueError):
        inverse_min_max_scale(input_tensor, input_scaler, feature_range=(0, 1))


def test_inverse_min_max_scale_success_fail_3():
    # Test that inverse min max fails when feature_range is not a tuple of two elements
    input_tensor = torch.tensor([[0, 0, 0.25], [0, 0.5, 0], [1.0, 0, 0]], dtype=torch.float32)
    input_scaler = (0.0, 4.0)

    feature_range = (0.0, 4.0, 2.0)
    with pytest.raises(ValueError):
        inverse_min_max_scale(input_tensor, input_scaler, feature_range=feature_range)
