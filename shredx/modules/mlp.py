"""Multi-layer perceptron (MLP) encoders and decoders for sequence modeling. Also implements MOE-MLP encoder."""

import math
import einops
import torch
import torch.nn as nn
from jaxtyping import Float
from typing import Tuple
from typing import List
from shredx.modules.moe_mixin import MOE_SINDy_Layer_Helpers_Mixin
from shredx.modules.sindy_layer import SindyLayer


class MLPEncoder(nn.Module):
    r"""Multi-layer perceptron (MLP) encoder.

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
        Device on which to place the module. Default is ``"cpu"``.

    Notes
    -----
    **Class Methods:**

    **forward(x):**

    - Applies the MLP encoder to an input batch.
    - Parameters:
        - x : ``Float[torch.Tensor, "batch sequence input_size"]``. Input tensor.
    - Returns:
        - tuple. Tuple containing the final output tensor of shape
          ``(batch_size, 1, 1, hidden_size)`` and ``None`` for no auxiliary losses.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        device: str = "cpu",
    ) -> None:
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
    r"""Multi-Layer Perceptron (MLP) decoder.

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
        Device on which to place the model. Default is ``"cpu"``.

    Notes
    -----
    **Class Methods:**

    **forward(x):**

    - Applies the MLP decoder to an input batch.
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


class MOEMLPEncoder(nn.Module, MOE_SINDy_Layer_Helpers_Mixin):
    r"""Multi-Layer Perceptron (MLP) with SINDy layer forecasting.

    Creates a feedforward neural network with identical layer sizes, ReLU
    activations between layers, and multiple SINDy expert layers. Expert
    outputs are combined via learned weighted averaging.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    hidden_size : int
        Hidden state dimension for MLP and experts.
    n_experts : int
        Number of SINDy expert layers.
    forecast_length : int
        Number of timesteps to forecast.
    strict_symmetry : bool
        If True, enforce symmetric SINDy coefficients.
    num_layers : int
        Number of MLP layers.
    dropout : float
        Dropout probability for expert weighting.
    device : str, optional
        Device on which to place the module. Default is ``"cpu"``.

    Notes
    -----
    **Class Methods:**

    **forward(x):**

    - Processes input through the MLP, then passes the final hidden state through all SINDy experts and combines their outputs.
    - Parameters:
        - x : torch.Tensor. Input tensor of shape ``(batch_size, sequence_length, input_size)``.
    - Returns:
        - dict. Keys: ``"sequence_output"`` (MLP output, shape ``(batch_size, sequence_length, hidden_size)``),
          ``"output"`` (combined expert forecasts, shape ``(batch_size, forecast_length, 1, hidden_size)``).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        forecast_length: int,
        strict_symmetry: bool,
        num_layers: int,
        dropout: float,
        device: str = "cpu",
    ) -> None:
        """Initialize the MOE-MLP model."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = n_experts
        self.forecast_length = forecast_length
        self.strict_symmetry = strict_symmetry
        self.num_layers = num_layers
        self.output_size = hidden_size
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

        mlp = nn.Sequential(*self.layers)
        mlp = mlp.to(device)
        self.mlp = mlp

        self.softmax = nn.Softmax(dim=-1)
        self.linear_combination = nn.Parameter(torch.ones(self.n_experts) / self.n_experts)
        self.experts = nn.ModuleList(
            [
                SindyLayer(
                    hidden_size=self.hidden_size,
                    forecast_length=self.forecast_length,
                    device=self.device,
                    strict_symmetry=self.strict_symmetry,
                )
                for _ in range(self.n_experts)
            ]
        )

    def forward(
        self, x: Float[torch.Tensor, "batch sequence input_size"]
    ) -> Tuple[Float[torch.Tensor, "batch forecast_length 1 hidden_size"], None]:
        """Forward pass through the MOE-MLP model."""
        out = self.mlp(x)

        # SINDy forward all experts
        sindy_outputs = [expert(out[:, -1, :]) for expert in self.experts]
        sindy_outputs = torch.stack(sindy_outputs)
        sindy_outputs = sindy_outputs.unsqueeze(3)  # Adds sequence length dimension

        # Combine experts: weighted sum across expert dimension
        # NOTE: Dropout drops random experts
        weights = self.dropout(self.linear_combination)
        weights = self.softmax(weights)
        combined = torch.einsum("ebfsd,e->bfsd", sindy_outputs, weights)

        return (combined, None)
