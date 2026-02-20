"""Time series dataset and dataloader utilities.

Provides a PyTorch ``Dataset`` for sliding-window time series built from
lists of input and output tensors.
"""

import bisect

import torch
from jaxtyping import Float
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    r"""Dataset of sliding windows over time series tensors.

    Builds a dataset from lists of input and output tensors, each of shape
    ``(time_steps, ...)``. Assumes input and output tensors have the same
    shape. Input and output windows are returned separately to support use
    with proper orthogonal decomposition; they can be split or concatenated
    as needed when used.

    Parameters
    ----------
    input_tensors : list of torch.Tensor
        List of input tensors, each a time series of shape ``(time_steps, ...)``.
    length : int
        Length of each sliding window.
    output_tensors : list of torch.Tensor, optional
        List of output tensors, each of shape ``(time_steps, ...)``.
        If ``None``, output windows are the same as input windows. Default is ``None``.
    device : str, optional
        Device to place tensors on. Default is ``"cpu"``.

    Raises
    ------
    ValueError
        If no valid windows can be created (e.g. tensor lengths too short for ``length``).

    Notes
    -----
    **Class Methods:**

    **__len__():**

    - Returns the total number of valid windows across all tensors.
    - Returns:
        - int. Total number of windows.

    **__getitem__(index):**

    - Returns the input and output windows for a given index.
    - Parameters:
        - index : int. Index of the window to return.
    - Returns:
        - tuple of ``(Float[torch.Tensor, "length ..."], Float[torch.Tensor, "length ..."])``.
          ``(input_window, output_window)``, each of shape ``(length, ...)``.

    **prepare_tensors(tensors):**

    - Converts a list of tensors to float32 and moves them to the configured device.
    - Parameters:
        - tensors : list of ``Float[torch.Tensor, "time_steps ..."]``. List of tensors to prepare.
    - Returns:
        - list of ``Float[torch.Tensor, "time_steps ..."]``. Tensors on the correct device in float32.
    """

    def __init__(
        self,
        input_tensors: list[Float[torch.Tensor, "time_steps ..."]],
        length: int,
        output_tensors: list[Float[torch.Tensor, "time_steps ..."]] | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize ``TimeSeriesDataset``."""
        super().__init__()
        self.length = length
        self.device = device

        # Preprocess
        self.input_tensors = self.prepare_tensors(input_tensors)
        self.output_tensors = None if output_tensors is None else self.prepare_tensors(output_tensors)

        # Calculate cumulative window counts for index mapping
        self.cumulative_offsets = [0]
        current = 0
        for tensor in self.input_tensors:
            L = tensor.size(0)
            # Need enough data for window
            n_windows = (L - length) + 1
            n_windows = max(n_windows, 0)  # Ensure non-negative
            current += n_windows
            self.cumulative_offsets.append(current)

        if self.cumulative_offsets[-1] == 0:
            raise ValueError("No valid windows created. Check input_length, output_length, and tensor lengths.")

    def prepare_tensors(
        self, tensors: list[Float[torch.Tensor, "time_steps ..."]]
    ) -> list[Float[torch.Tensor, "time_steps ..."]]:
        """Prepare a list of tensors for use in the dataset.

        Converts arrays to torch tensors, float32, and moves them to the
        configured device.

        Parameters
        ----------
        tensors : list of Float[torch.Tensor, "time_steps ..."]
            List of tensors or arrays to prepare.

        Returns
        -------
        list of Float[torch.Tensor, "time_steps ..."]
            List of tensors on the correct device in float32.
        """
        # To float32
        tensors = [tensor.float() for tensor in tensors]

        # Move to GPU
        tensors = [tensor.to(self.device) for tensor in tensors]

        return tensors

    def __len__(self) -> int:
        """Return the total number of windows in the dataset."""
        return self.cumulative_offsets[-1]

    def __getitem__(self, index: int) -> tuple[Float[torch.Tensor, "length ..."], Float[torch.Tensor, "length ..."]]:
        """Return the input and output windows for a given index.

        Parameters
        ----------
        index : Index of the window to return.

        Returns
        -------
        tuple of (Float[torch.Tensor, "length ..."], Float[torch.Tensor, "length ..."])
            ``(input_window, output_window)``, each of shape ``(length, ...)``.
        """
        # Find which tensor contains this index
        tensor_idx = bisect.bisect_right(self.cumulative_offsets, index) - 1

        # Calculate local index within the tensor
        start_idx = self.cumulative_offsets[tensor_idx]
        local_idx = index - start_idx

        # Get corresponding tensor and calculate input window
        start = local_idx
        end = start + self.length

        # Get corresponding input and output tensors
        input_tensor = self.input_tensors[tensor_idx]
        output_tensor = input_tensor if self.output_tensors is None else self.output_tensors[tensor_idx]

        # Get corresponding input and output windows
        input_window = input_tensor[start:end]
        output_window = output_tensor[start:end]

        return input_window, output_window
