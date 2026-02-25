"""This file contains all plotting utilities."""

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float


def plot_timestep(
    tensor: Float[torch.Tensor, "rows cols dim"],
    title: str = "",
    save: bool = False,
    fname: str = "plot.pdf",
    sensors: list[tuple[int, int]] | None = None,
) -> None:
    """Plot the input tensor.

    Parameters
    ----------
    tensor : Float[torch.Tensor, "rows cols dim"]
        The tensor to plot.
    title : str, optional
        The title of the plot.
    save : bool, optional
        Whether to save the plot. Default is False.
    fname : str, optional
        The name of the file to save the plot to. Default is "plot.pdf".
    sensors : list of (row, col) tuples, optional
        If provided, red dots are drawn at each (row, col) coordinate on every
        dimension subplot. Same convention as generate_static_sensors_from_mask.
    """
    tensor = tensor.detach().cpu()
    n_rows, n_cols, n_dims = tensor.shape

    fig, axes = plt.subplots(n_dims, 1, squeeze=False, figsize=(6, 3 * n_dims))
    axes = axes[:, 0]

    for d in range(n_dims):
        ax = axes[d]
        im = ax.imshow(
            tensor[:, :, d].numpy(),
            aspect="equal",
            origin="upper",
            cmap="viridis",
            extent=(0, n_cols - 1, n_rows - 1, 0),
        )
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlim(0, n_cols - 1)
        ax.set_ylim(n_rows - 1, 0)
        ax.set_ylabel("row")
        ax.set_title(f"Dimension {d}")
        if d == n_dims - 1:
            ax.set_xlabel("col")

        if sensors is not None and len(sensors) > 0:
            cols = [c for _r, c in sensors]
            rows = [r for r, _c in sensors]
            ax.scatter(cols, rows, c="red", s=30, zorder=5)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save:
        fig.savefig(fname)
    else:
        plt.show()
    plt.close(fig)


def plot_sensor_timeseries(
    tensor: Float[torch.Tensor, "timesteps n_sensors dim"],
    title: str = "",
    save: bool = False,
    fname: str = "plot.pdf",
) -> None:
    """Plot sensor values over time, one subplot per (dimension, sensor) pair.

    Subplots are ordered first by dimension, then by sensor, each on its own row.

    Parameters
    ----------
    tensor : Float[torch.Tensor, "timesteps n_sensors dim"]
        Sensor readings over time.
    title : str, optional
        The title of the plot.
    save : bool, optional
        Whether to save the plot. Default is False.
    fname : str, optional
        The name of the file to save the plot to. Default is "plot.pdf".
    """
    tensor = tensor.detach().cpu()
    n_timesteps, n_sensors, n_dims = tensor.shape

    n_subplots = n_dims * n_sensors
    height_per_subplot = 2.0
    fig, axes = plt.subplots(
        n_subplots,
        1,
        squeeze=False,
        figsize=(7, n_subplots * height_per_subplot),
    )
    axes = axes[:, 0]

    time = range(n_timesteps)
    for d in range(n_dims):
        for s in range(n_sensors):
            row = d * n_sensors + s
            ax = axes[row]
            ax.plot(time, tensor[:, s, d].numpy())
            ax.set_ylabel("value")
            ax.set_title(f"dim {d}, sensor {s}")
            if row == n_subplots - 1:
                ax.set_xlabel("t")

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save:
        fig.savefig(fname)
    else:
        plt.show()
    plt.close(fig)
