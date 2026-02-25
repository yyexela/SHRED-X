import pickle
from pathlib import Path

import einops
import numpy as np
import scipy.io as sio
import torch

from shredx.datasets.dataloaders import TimeSeriesDataset
from shredx.utils.scaling import min_max_scale

data_dir = Path(__file__).parent.parent.parent / "data"


def load_sst_data(input_length: int, forecast_length: int, device: str = "cpu"):
    """
    Loads the SST dataset from the `datasets/sst/` directory.

    Parameters
    ----------
    input_length : int
        The length of the input sequence.
    forecast_length : int
        The length of the forecast sequence.
    device : str, optional
        The device to load the data on. Default is "cpu".

    Returns
    -------
    tuple: (train_ds, valid_ds, test_ds, metadata) where each is a TimeSeriesDataset and metadata is a dictionary containing the scalers.

    Notes
    -----
    The shape of each dataset is (input_length + forecast_length, rows, cols). Each dataloader will return a tuple of (input_window, output_window) where each is a Float[torch.Tensor, "length rows cols 1"].
    """
    # Load raw file
    sst_data_path = data_dir / "sst" / "SST_data.mat"
    sst_zeros_path = data_dir / "sst" / "SST_zeros.pkl"
    with open(sst_zeros_path, "rb") as f:
        sst_zeros = pickle.load(f)
        sst_zeros = torch.from_numpy(sst_zeros).float()
    sst_data = sio.loadmat(sst_data_path)["Z"]  # (64800, 1400)
    sst_data = einops.rearrange(sst_data, "(r c) t -> t r c 1", r=180, c=360, t=1400)
    sst_zeros = einops.rearrange(sst_zeros, "r c -> r c 1", r=180, c=360)

    # Create training, testing, and validation split
    train_size = int(sst_data.shape[0] * 0.8)
    val_size = int(sst_data.shape[0] * 0.1)
    train, val, test = np.split(sst_data, [train_size, train_size + val_size])

    # Convert sst_data to torch
    sst_data = torch.from_numpy(sst_data).float()

    # Convert data to pytorch (treat it like a row x col x 1 image)
    train = torch.from_numpy(train).float()
    val = torch.from_numpy(val).float()
    test = torch.from_numpy(test).float()

    # Min Max Scale data
    _, scaler = min_max_scale(sst_data)
    train, _ = min_max_scale(train, scaler=scaler)
    val, _ = min_max_scale(val, scaler=scaler)
    test, _ = min_max_scale(test, scaler=scaler)

    # Create torch datasets
    datasets = []
    for i, split in enumerate([train, val, test]):
        sst_ds = TimeSeriesDataset(
            input_tensors=[split],
            length=input_length + forecast_length,
            device=device,
        )
        datasets.append(sst_ds)

    train_ds = datasets[0]
    valid_ds = datasets[1]
    test_ds = datasets[2]

    return train_ds, valid_ds, test_ds, {"scalers": [scaler], "sst_zeros": sst_zeros}
