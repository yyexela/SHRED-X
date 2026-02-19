"""Recurrent neural network encoders for sequence modeling.

Implements GRU, LSTM, and MLP encoders compatible with an
encoder–decoder architecture. Also implements MOE-GRU and MOE-LSTM encoders.
"""

import einops
import torch
import torch.nn as nn
from jaxtyping import Float
from typing import Tuple
from shredx.modules.moe_mixin import MOE_SINDy_Layer_Helpers_Mixin
from shredx.modules.sindy_layer import SindyLayer


class GRUEncoder(nn.Module):
    r"""GRU encoder for sequence-to-sequence modeling.

    Wraps PyTorch's ``GRU`` with dropout and output reshaping for
    compatibility with encoder–decoder architectures.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    hidden_size : int
        Hidden state dimension.
    num_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout probability applied to the outputs.
    device : str, optional
        Device on which to place the module. Default is ``"cpu"``.
    **kwargs
        Additional keyword arguments passed for compatibility but ignored.

    Notes
    -----
    **Class Methods:**

    **forward(x):**

    - Applies the GRU encoder to an input batch.
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
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(
        self, x: Float[torch.Tensor, "batch sequence input_size"]
    ) -> Tuple[Float[torch.Tensor, "batch 1 1 hidden_size"], None]:
        """Apply the GRU encoder to an input batch."""
        out, h_out = self.gru(x)

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        out = einops.rearrange(out, "b s d -> b 1 s d")
        h_out = einops.rearrange(h_out, "h b d -> b 1 h d")

        final_output = h_out[:, :, -1:, :]

        return (final_output, None)


class LSTMEncoder(nn.Module):
    r"""LSTM encoder for sequence-to-sequence modeling.

    Wraps PyTorch's ``LSTM`` with dropout and output reshaping for
    compatibility with encoder–decoder architectures.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    hidden_size : int
        Hidden state dimension.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability applied to the outputs.
    device : str, optional
        Device on which to place the module. Default is ``"cpu"``.
    **kwargs
        Additional keyword arguments passed for compatibility but ignored.

    Notes
    -----
    **Class Methods:**

    **forward(x):**

    - Applies the LSTM encoder to an input batch.
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
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(
        self, x: Float[torch.Tensor, "batch sequence input_size"]
    ) -> Tuple[Float[torch.Tensor, "batch 1 1 hidden_size"], None]:
        """Apply the LSTM encoder to an input batch."""
        out, (h_out, c_out) = self.lstm(x)

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        out = einops.rearrange(out, "b s d -> b 1 s d")
        h_out = einops.rearrange(h_out, "h b d -> b 1 h d")

        final_output = h_out[:, :, -1:, :]

        return (final_output, None)


class MOEGRUEncoder(nn.Module, MOE_SINDy_Layer_Helpers_Mixin):
    r"""Mixture of Experts GRU with SINDy layer forecasting.

    Combines a GRU encoder with multiple SINDy expert layers for long-horizon
    forecasting. Expert outputs are combined via learned weighted averaging.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    hidden_size : int
        Hidden state dimension for GRU and experts.
    n_experts : int
        Number of SINDy expert layers.
    forecast_length : int
        Number of timesteps to forecast.
    strict_symmetry : bool
        If True, enforce symmetric SINDy coefficients.
    num_layers : int
        Number of GRU layers.
    dropout : float
        Dropout probability for expert weighting.
    device : str, optional
        Device on which to place the module. Default is ``"cpu"``.
    **kwargs
        Additional keyword arguments (ignored).

    Notes
    -----
    **Class Methods:**

    **forward(x):**

    - Processes input through the GRU, then passes the final hidden state through all SINDy experts and combines their outputs.
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
        n_experts: int,
        forecast_length: int,
        strict_symmetry: bool,
        num_layers: int,
        dropout: float,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the MOE-GRU model."""
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

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

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
        """Forward pass through the MOE-GRU model."""
        # Normal GRU forward
        out, h_out = self.gru(x)

        # SINDy forward all experts
        sindy_outputs = [expert(h_out[-1]) for expert in self.experts]
        sindy_outputs = torch.stack(sindy_outputs)
        sindy_outputs = sindy_outputs.unsqueeze(3)  # Adds sequence length dimension

        # Combine experts: weighted sum across expert dimension
        # NOTE: Dropout drops random experts
        weights = self.dropout(self.linear_combination)
        weights = self.softmax(weights)
        combined = torch.einsum("ebfsd,e->bfsd", sindy_outputs, weights)

        return (combined, None)


class MOELSTMEncoder(nn.Module, MOE_SINDy_Layer_Helpers_Mixin):
    r"""Mixture of Experts LSTM with SINDy layer forecasting.

    Combines an LSTM encoder with multiple SINDy expert layers for long-horizon
    forecasting. Expert outputs are combined via learned weighted averaging.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    hidden_size : int
        Hidden state dimension for LSTM and experts.
    n_experts : int
        Number of SINDy expert layers.
    forecast_length : int
        Number of timesteps to forecast.
    strict_symmetry : bool
        If True, enforce symmetric SINDy coefficients.
    num_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout probability for expert weighting.
    device : str, optional
        Device on which to place the module. Default is ``"cpu"``.
    **kwargs
        Additional keyword arguments (ignored).

    Notes
    -----
    **Class Methods:**

    **initialize():**

    - Initializes the LSTM, expert combination weights, and SINDy expert layers (called from ``__init__``).
    - Returns:
        - None.

    **forward(x):**

    - Processes input through the LSTM, then passes the final hidden state through all SINDy experts and combines their outputs.
    - Parameters:
        - x : torch.Tensor. Input tensor of shape ``(batch_size, sequence_length, input_size)``.
    - Returns:
        - dict. Keys: ``"sequence_output"`` (LSTM output, shape ``(batch_size, sequence_length, hidden_size)``),
          ``"final_hidden_state"`` (shape ``(num_layers, batch_size, hidden_size)``),
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
        **kwargs,
    ) -> None:
        """Initialize the MOE-LSTM model."""
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

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

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
        """Forward pass through the MOE-LSTM model."""
        # Normal LSTM forward
        out, (h_out, c_out) = self.lstm(x)

        # SINDy forward all experts
        sindy_outputs = [expert(h_out[-1]) for expert in self.experts]
        sindy_outputs = torch.stack(sindy_outputs)
        sindy_outputs = sindy_outputs.unsqueeze(3)  # Adds sequence length dimension

        # Combine experts: weighted sum across expert dimension
        # NOTE: Dropout drops random experts
        weights = self.dropout(self.linear_combination)
        weights = self.softmax(weights)
        combined = torch.einsum("ebfsd,e->bfsd", sindy_outputs, weights)

        return (combined, None)
