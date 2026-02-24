"""Tests for sensor utilities."""

import pytest
import torch

from shredx.utils.sensors import (
    extract_static_sensors,
    generate_static_sensors_from_mask,
)


def test_generate_static_sensors_from_mask_dim_agnostic_success():
    """dim_agnostic=True returns (row, col) where all dims are valid."""
    # mask: shape (rows=2, cols=2, dim=2)
    # Valid spatial locations (all dims == 0) are (0, 0) and (1, 1)
    mask = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0]],
        ]
    )

    sensors = generate_static_sensors_from_mask(
        n_sensors=2,
        mask=mask,
        dim_agnostic=True,
    )

    assert set(sensors) == {(0, 0), (1, 1)}


def test_generate_static_sensors_from_mask_dim_specific_success():
    """dim_agnostic=False returns (row, col, dim) for each valid dim."""
    # Only (0, 0, 0) and (1, 0, 1) are valid.
    mask = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ]
    )

    sensors = generate_static_sensors_from_mask(
        n_sensors=2,
        mask=mask,
        dim_agnostic=False,
    )

    assert set(sensors) == {(0, 0, 0), (1, 0, 1)}


def test_generate_static_sensors_from_mask_dim_agnostic_raises_when_no_all_dim_valid():
    """dim_agnostic=True raises when no (row, col) has all dims valid."""
    mask = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ]
    )

    with pytest.raises(ValueError):
        generate_static_sensors_from_mask(
            n_sensors=1,
            mask=mask,
            dim_agnostic=True,
        )


def test_generate_static_sensors_from_mask_too_many_sensors_raises():
    """Requesting more sensors than valid positions raises ValueError."""
    mask = torch.zeros((2, 2, 1))  # 4 valid positions

    with pytest.raises(ValueError):
        generate_static_sensors_from_mask(
            n_sensors=5,
            mask=mask,
        )


def test_extract_static_sensors_success_dim_agnostic():
    """extract_static_sensors returns the expected values."""
    # tensor shape: (time=3, rows=2, cols=2, dim=2)
    base = torch.arange(3 * 2 * 2 * 2, dtype=torch.float32)
    tensor = base.view(3, 2, 2, 2)

    # Choose two sensor locations: (row, col)
    sensor_locations = [(0, 0), (1, 1)]

    extracted = extract_static_sensors(tensor, sensor_locations)

    # Manually build expected result
    expected = torch.stack(
        [
            tensor[:, 0, 0, :],  # first sensor
            tensor[:, 1, 1, :],  # second sensor
        ],
        dim=1,
    )

    assert extracted.shape == expected.shape
    assert torch.allclose(extracted, expected)


def test_extract_static_sensors_success_dim_specific():
    """extract_static_sensors returns the expected values."""
    # tensor shape: (time=3, rows=2, cols=2, dim=2)
    base = torch.arange(3 * 2 * 2 * 2, dtype=torch.float32)
    tensor = base.view(3, 2, 2, 2)

    # Choose two sensor locations: (row, col)
    sensor_locations = [(0, 0, 0), (1, 0, 1)]

    extracted = extract_static_sensors(tensor, sensor_locations)

    # Manually build expected result: shape (time, n_sensors)
    expected = torch.stack(
        [
            tensor[:, 0, 0, 0],  # first sensor
            tensor[:, 1, 0, 1],  # second sensor
        ],
        dim=1,
    )

    assert extracted.shape == expected.shape
    assert torch.allclose(extracted, expected)


def test_extract_static_sensors_fails_when_no_sensors():
    """extract_static_sensors raises when no sensors are provided."""
    tensor = torch.randn(3, 2, 2, 2)
    sensor_locations: list[tuple[int, int]] = []
    with pytest.raises(ValueError):
        extract_static_sensors(tensor, sensor_locations)
