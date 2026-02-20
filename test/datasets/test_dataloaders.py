"""Verify the scaling functions are working correctly."""

import pytest
import torch

from shredx.datasets.dataloaders import TimeSeriesDataset


def test_time_series_dataset_success():
    # Test that the dataset can be created and has the correct length
    window_length = 5
    input_tensors = [torch.randn(10, 3), torch.randn(10, 3)]
    output_tensors = [torch.randn(10, 2), torch.randn(10, 2)]
    dataset = TimeSeriesDataset(input_tensors, window_length, output_tensors)
    assert len(dataset) == 12

    for i in range(window_length * len(input_tensors)):
        j = i % len(input_tensors)
        assert torch.allclose(dataset[j][0], input_tensors[0][j : j + window_length])
        assert torch.allclose(dataset[j][1], output_tensors[0][j : j + window_length])


def test_time_series_dataset_fail():
    # Test that the dataset fails when the window length is greater than the input tensor length
    window_length = 51
    input_tensors = [torch.randn(10, 3)]
    with pytest.raises(ValueError):
        TimeSeriesDataset(input_tensors, window_length)
