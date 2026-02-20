import math

import einops
import torch
from jaxtyping import Float
from torch import nn


class CNNDecoder(nn.Module):
    r"""1D convolutional neural network (CNN) decoder.

    Creates a convolutional network with logarithmically spaced channel sizes
    between the input and output dimensions. Uses 1D convolutions with kernel
    size 3 and padding 1, ReLU activations between intermediate layers, and
    applies dropout after the final layer.

    Parameters
    ----------
    in_dim : int
        Input channel dimension of the decoder.
    out_dim : int
        Output channel dimension of the decoder.
    n_layers : int
        Number of convolutional layers in the network.
    dropout : float
        Dropout probability applied after the final layer.
    device : str, optional
        Device on which to place the module. Default is ``"cpu"``.

    Notes
    -----
    **Class Methods:**

    **forward(x):**

    - Applies the CNN decoder to an input batch.
    - Parameters:
        - x : tuple. Tuple containing the input tensor of shape
          ``(batch, forecast_length, sequence_length, hidden_dim)`` and the auxiliary losses
          (``List[torch.FloatTensor] | None``).
    - Returns:
        - tuple. Tuple containing the decoded tensor of shape
          ``(batch, forecast_length, sequence_length, out_dim)`` and the auxiliary losses
          (``List[torch.FloatTensor] | None``).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_layers: int,
        dropout: float,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        # Class variables
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Model layer sizes
        start_exp = math.log2(in_dim)
        end_exp = math.log2(out_dim)
        sizes_tensor = torch.logspace(
            start_exp,
            end_exp,
            steps=n_layers + 1,
            base=2.0,
            dtype=torch.float32,
        )
        sizes = sizes_tensor.to(dtype=torch.int64).tolist()
        sizes[0] = self.in_dim
        sizes[-1] = self.out_dim

        # Define model layers
        self.layers = []
        for idx in range(len(sizes) - 1):
            self.layers.append(nn.Conv1d(sizes[idx], sizes[idx + 1], kernel_size=3, padding=1))
            if idx != (len(sizes) - 2):
                self.layers.append(nn.ReLU())

        model = nn.Sequential(*self.layers)
        model = model.to(device)
        self.model = model

    def forward(
        self,
        x: tuple[
            Float[torch.Tensor, "batch forecast_length sequence_length hidden_dim"], list[torch.FloatTensor] | None
        ],
    ) -> tuple[Float[torch.Tensor, "batch forecast_length sequence_length out_dim"], list[torch.FloatTensor] | None]:
        """Apply the CNN decoder to an input batch."""
        aux_losses = x[1]
        x_in = x[0]

        _batch_size, forecast_length, sequence_length, _hidden_dim = x_in.shape
        x_in = einops.rearrange(x_in, "b f s d -> b d (f s)", f=forecast_length, s=sequence_length)
        out = self.model(x_in)
        out = self.dropout(out)  # want: batch forecast seq_len (rows cols dim)
        out = einops.rearrange(out, "b o (f s) -> b f s o", f=forecast_length, s=sequence_length)
        return (out, aux_losses)
