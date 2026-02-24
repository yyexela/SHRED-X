"""Smoke tests for plotting utilities."""

import matplotlib
import pytest

matplotlib.use("Agg")  # use non-interactive backend for tests

import torch

from shredx.utils.plots import plot_sensor_timeseries, plot_timestep


@pytest.mark.parametrize("save_file", [True, False])
def test_plot_timestep(tmp_path, save_file):
    """plot_timestep should run and save a figure without error."""
    tensor = torch.randn(4, 5, 2)  # rows, cols, dim
    sensors = [(0, 0), (3, 4)]

    out_file = tmp_path / "timestep.pdf"
    plot_timestep(tensor, title="test timestep", save=save_file, fname=str(out_file), sensors=sensors)

    if save_file:
        assert out_file.exists()


@pytest.mark.parametrize("save_file", [True, False])
def test_plot_sensor_timeseries(tmp_path, save_file):
    """plot_sensor_timeseries should run and save a figure without error."""
    tensor = torch.randn(10, 3, 2)  # timesteps, n_sensors, dim

    out_file = tmp_path / "sensor_timeseries.pdf"
    plot_sensor_timeseries(tensor, title="test sensors", save=save_file, fname=str(out_file))

    if save_file:
        assert out_file.exists()
