"""This file contains scaling utilities."""

import torch
from jaxtyping import Float


def min_max_scale(
    tensor: Float[torch.Tensor, "... dim"],
    feature_range: tuple[float, float] = (0, 1),
    scaler: tuple[float, float] | None = None,
) -> tuple[Float[torch.Tensor, "... dim"], tuple[float, float]]:
    """
    Scale a tensor to a given feature range using min-max normalization.

    The input tensor must contain at least two elements. By default, the
    tensor is scaled to the `(0, 1)` interval. Optionally, a precomputed
    ``(min, max)`` pair can be passed via ``scaler`` for consistent scaling
    across different tensors.

    Parameters
    ----------
    tensor : Float[torch.Tensor, "... dim"]
        Input tensor to be scaled.
    feature_range : tuple[float, float], optional
        Desired range of transformed data, by default ``(0, 1)``.
    scaler : tuple[float, float] or None, optional
        Tuple ``(min, max)`` used for scaling, typically obtained from a
        previous call to this function. If ``None``, the minimum and maximum
        are computed from ``tensor``.

    Returns
    -------
    scaled : Float[Tensor, "... dim"]
        Tensor scaled to the specified ``feature_range``.
    min_max : tuple[float, float]
        Tuple ``(min, max)`` values used for scaling, either computed from
        ``tensor`` or passed in via ``scaler``.

    Raises
    ------
    ValueError
        If ``feature_range`` is not a tuple of two elements.
    ValueError
        If ``tensor`` has fewer than two elements.
    """
    # Check that the feature range is a tuple of two elements
    if type(feature_range) is not tuple or len(feature_range) != 2:
        raise ValueError("Feature range must be a tuple of two elements")

    # Ensure the input has at least two elements
    if torch.numel(tensor) < 2:
        raise ValueError("Input tensor must have at least two elements")

    if scaler is None:
        # Calculate min and max
        t_min = float(tensor.min())
        t_max = float(tensor.max())
    else:
        t_min, t_max = scaler

    # Avoid division by zero
    t_range = t_max - t_min
    if t_range == 0:  # all values are the same
        t_range = 1

    # Scale to [0, 1] first
    scaled = (tensor - t_min) / t_range

    # Then scale to feature_range
    min_range, max_range = feature_range
    scaled = scaled * (max_range - min_range) + min_range

    return scaled, (t_min, t_max)


def inverse_min_max_scale(
    scaled_tensor: Float[torch.Tensor, "... dim"],
    original_min_max: tuple[float, float],
    feature_range: tuple[float, float] = (0, 1),
) -> Float[torch.Tensor, "... dim"]:
    """
    Invert min-max scaling and recover the original data scale.

    This performs the reverse transformation of :func:`min_max_scale`
    using the original ``(min, max)`` values and the ``feature_range``
    used during scaling.

    Parameters
    ----------
    scaled_tensor : Float[torch.Tensor, "... dim"]
        Tensor that was previously scaled with min-max normalization.
    original_min_max : tuple[float, float]
        Tuple ``(min, max)`` values from the original data before scaling.
    feature_range : tuple[float, float], optional
        Feature range that was used during the original scaling, by
        default ``(0, 1)``.

    Returns
    -------
    Float[torch.Tensor, "... dim"]
        Tensor mapped back to the original scale.

    Raises
    ------
    ValueError
        If ``feature_range`` is not a tuple of two elements.
    ValueError
        If ``original_min_max`` is not a tuple of two elements.
    """
    if type(feature_range) is not tuple or len(feature_range) != 2:
        raise ValueError("Feature range must be a tuple of two elements")
    if type(original_min_max) is not tuple or len(original_min_max) != 2:
        raise ValueError("Original min-max must be a tuple of two elements")

    t_min, t_max = original_min_max
    min_range, max_range = feature_range

    # First scale back to [0, 1] range
    normalized = (scaled_tensor - min_range) / (max_range - min_range)

    # Then scale back to original range
    original = normalized * (t_max - t_min) + t_min

    return original
