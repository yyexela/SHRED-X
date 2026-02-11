"""Recurrent neural network encoders for sequence modeling.

Implements GRU, LSTM, and MLP encoders compatible with an
encoder–decoder architecture.
"""

import einops
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


class GRU(nn.Module):
    """GRU encoder for sequence-to-sequence modeling.

    Wraps PyTorch's ``GRU`` with dropout and output reshaping for
    compatibility with encoder–decoder architectures.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize the GRU encoder.

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
            Device on which to place the module, by default ``"cpu"``.
        **kwargs
            Additional keyword arguments passed for compatibility but ignored.
        """
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

    def forward(self, x: Float[Tensor, "batch sequence input_size"]) -> Float[Tensor, "batch 1 1 hidden_size"]:
        """Compute a forward pass through the GRU encoder.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape
            ``(batch_size, sequence_length, input_size)``.

        Returns
        -------
        final_output : Tensor
            Final output tensor of shape
            ``(batch_size, 1, 1, hidden_size)``.
            Note: second dimension and third dimension are 1, corresponding to
                  one forecast step and the final hidden state respectively.
        """
        out, h_out = self.gru(x)

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        out = einops.rearrange(out, "b s d -> b 1 s d")
        h_out = einops.rearrange(h_out, "h b d -> b 1 h d")

        final_output = h_out[:, :, -1:, :]

        return final_output


class LSTM(nn.Module):
    """LSTM encoder for sequence-to-sequence modeling.

    Wraps PyTorch's ``LSTM`` with dropout and output reshaping for
    compatibility with encoder–decoder architectures.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize the LSTM encoder.

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
            Device on which to place the module, by default ``"cpu"``.
        **kwargs
            Additional keyword arguments passed for compatibility but ignored.
        """
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

    def forward(self, x: Float[Tensor, "batch sequence input_size"]) -> Float[Tensor, "batch 1 1 hidden_size"]:
        """Compute a forward pass through the LSTM encoder.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape
            ``(batch_size, sequence_length, input_size)``.

        Returns
        -------
        final_output : Tensor
            Final output tensor of shape
            ``(batch_size, 1, 1, hidden_size)``.
            Note: second dimension and third dimension are 1, corresponding to
                  one forecast step and the final hidden state respectively.
        """
        out, (h_out, c_out) = self.lstm(x)

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        out = einops.rearrange(out, "b s d -> b 1 s d")
        h_out = einops.rearrange(h_out, "h b d -> b 1 h d")

        final_output = h_out[:, :, -1:, :]

        return final_output
