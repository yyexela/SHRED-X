"""This file contains utilities for extracting sensors from a tensor."""

from typing import cast

import torch
from jaxtyping import Float


def generate_static_sensors_from_mask(
    n_sensors: int, mask: Float[torch.Tensor, "rows cols dim"], dim_agnostic: bool = True
) -> list[tuple[int, int]] | list[tuple[int, int, int]]:
    """Generate random locations for static sensors from a tensor.

    Parameters
    ----------
    n_sensors : int
        The number of sensors to generate.
    mask : Float[torch.Tensor, "rows cols dim"]
        The mask to use to generate sensors. 0 means a sensor can be placed there, otherwise means it cannot.
    dim_agnostic: bool, optional
        Whether to generate sensors in a dimension-agnostic manner. Default is True.
        If True, sensor locations are only on rows, cols where all dims are valid.
        If false, sensor locations are on rows, cols, dims where at least one dim is valid.

    Returns
    -------
    list[tuple[int, int]] | list[tuple[int, int, int]]
        If ``dim_agnostic`` is True, returns (row, col) locations where **all**
        dimensions are valid. If False, returns (row, col, dim) locations for
        each valid dimension.

    Raises
    ------
    ValueError
        If there are fewer valid positions than n_sensors.
    """
    if dim_agnostic:
        # Dimension-agnostic over channels: a spatial location (row, col)
        # is valid only if it is valid in **all** dimensions.
        per_dim_valid = mask == 0
        valid = per_dim_valid.all(dim=-1)
        rows, cols = valid.nonzero(as_tuple=True)
        valid_locations = list(zip(rows.tolist(), cols.tolist(), strict=True))
    else:
        # Dimension-specific: each sensor corresponds to a particular (row, col, dim)
        # triple where that specific dimension is valid.
        rows, cols, dims = (mask == 0).nonzero(as_tuple=True)
        valid_locations = list(zip(rows.tolist(), cols.tolist(), dims.tolist(), strict=True))

    if len(valid_locations) < n_sensors:
        raise ValueError(f"Requested {n_sensors} sensors but only {len(valid_locations)} valid positions in mask.")

    indices = torch.randperm(len(valid_locations), device=mask.device)[:n_sensors]
    return [valid_locations[i] for i in indices.tolist()]


def extract_static_sensors(
    tensor: Float[torch.Tensor, "time rows cols dim"],
    sensor_locations: list[tuple[int, int]] | list[tuple[int, int, int]],
) -> Float[torch.Tensor, "time n_sensors dim"]:
    """Extract sensor readings from a tensor at the given locations for each time step.

    Parameters
    ----------
    tensor : Float[torch.Tensor, "time rows cols dim"]
        The tensor to extract sensors from (time, rows, cols, channels).
    sensor_locations : list of tuples of (row, col)
        The locations of the sensors (row, col coordinates).

    Returns
    -------
    Float[torch.Tensor, "time n_sensors dim"]
        The extracted sensor values at each time step.
    """
    if len(sensor_locations) == 0:
        raise ValueError("No sensor locations provided.")
    elif len(sensor_locations[0]) == 2:
        sensor_locations = cast(list[tuple[int, int]], sensor_locations)
        rows = torch.tensor([r for r, c in sensor_locations], device=tensor.device)
        cols = torch.tensor([c for r, c in sensor_locations], device=tensor.device)
        return tensor[:, rows, cols, :]
    elif len(sensor_locations[0]) == 3:
        sensor_locations = cast(list[tuple[int, int, int]], sensor_locations)
        rows = torch.tensor([r for r, c, d in sensor_locations], device=tensor.device)
        cols = torch.tensor([c for r, c, d in sensor_locations], device=tensor.device)
        dims = torch.tensor([d for r, c, d in sensor_locations], device=tensor.device)
        return tensor[:, rows, cols, dims]
    else:
        raise ValueError(f"Invalid sensor locations: {sensor_locations}")
