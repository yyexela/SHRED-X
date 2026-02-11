"""Multi-layer perceptron (MLP) encoders for sequence modeling."""

import math
import einops
import torch
import torch.nn as nn
from jaxtyping import Float
from typing import Tuple
from typing import List


class MLPEncoder(nn.Module):
    """Multi-layer perceptron (MLP) encoder.

    Creates a feed-forward neural network with identical layer sizes.
    Uses ReLU activations between layers and applies dropout after the
    final layer.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    hidden_size : int
        Hidden state dimension.
    num_layers : int
        Number of stacked MLP layers.
    dropout : float
        Dropout probability applied to the outputs.
    device : str, optional
        Device on which to place the module, by default ``"cpu"``.

    Notes
    -----
    **Inputs**

    - ``x`` : torch.Tensor
      Input tensor of shape ``(batch_size, sequence_length, input_size)``.

    **Outputs**

    - ``output`` : Tuple
      Tuple containing the final
      output tensor of shape ``(batch_size, 1, 1, hidden_size)`` and None for no auxiliary losses.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        device: str = "cpu",
    ):
        """Initialize ``MLPEncoder``."""
        super(MLPEncoder, self).__init__()
        # Class variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Model layer sizes
        sizes = [self.input_size] + [self.hidden_size] * self.num_layers

        # Define model layers
        self.layers = []
        for idx in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[idx], sizes[idx + 1]))
            if idx != (len(sizes) - 2):
                self.layers.append(nn.ReLU())

        model = nn.Sequential(*self.layers)
        model = model.to(device)
        self.model = model

    def forward(
        self, x: Float[torch.Tensor, "batch sequence input_size"]
    ) -> Tuple[Float[torch.Tensor, "batch 1 1 hidden_size"], None]:
        """Apply the MLP encoder to an input batch."""
        out = self.model(x)
        out = self.dropout(out)
        out = einops.rearrange(out, "b s d -> b 1 s d")

        final_output = out[:, :, -1:, :]

        return (final_output, None)


class MLPDecoder(nn.Module):
    """Multi-Layer Perceptron (MLP) decoder.

    Creates a feedforward neural network with logarithmically spaced layer
    sizes between the input and output dimensions. Uses ReLU activations
    between intermediate layers and applies dropout after the final layer.

    Parameters
    ----------
    in_dim : int
        Input feature dimension of the decoder.
    out_dim : int
        Output feature dimension of the decoder.
    n_layers : int
        Number of linear layers in the network.
    dropout : float
        Dropout probability applied after the final layer.
    device : str, optional
        Device on which to place the model, by default ``"cpu"``.

    Notes
    -----
    **Inputs**

    - ``x`` : Tuple
      Tuple containing the input tensor of shape
      ``(batch, forecast_length, sequence_length, hidden_dim)`` and the auxiliary losses.

    **Outputs**

    - ``output`` : Tuple
      Tuple containing the decoded tensor of shape
      ``(batch, forecast_length, sequence_length, out_dim)`` and the auxiliary losses.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_layers: int,
        dropout: float,
        device: str = "cpu",
    ):
        """Initialize the MLP decoder."""
        super(MLPDecoder, self).__init__()
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
            self.layers.append(nn.Linear(sizes[idx], sizes[idx + 1]))
            if idx != (len(sizes) - 2):
                self.layers.append(nn.ReLU())

        model = nn.Sequential(*self.layers)
        model = model.to(device)
        self.model = model

    def forward(
        self,
        x: Tuple[
            Float[torch.Tensor, "batch forecast_length sequence_length hidden_dim"], List[torch.FloatTensor] | None
        ],
    ) -> Tuple[Float[torch.Tensor, "batch forecast_length sequence_length out_dim"], List[torch.FloatTensor] | None]:
        """Apply the MLP decoder to an input batch."""
        aux_losses = x[1]
        x_in = x[0]
        out = self.model(x_in)
        out = self.dropout(out)
        return (out, aux_losses)
