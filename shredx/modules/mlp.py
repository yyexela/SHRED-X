"""Multi-layer perceptron (MLP) encoders for sequence modeling."""

import einops
import torch.nn as nn


class MLPEncoder(nn.Module):
    """Multi-layer perceptron (MLP) encoder.

    Creates a feed-forward neural network with identical layer sizes.
    Uses ReLU activations between layers and applies dropout after the
    final layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        device: str = "cpu",
    ):
        """Initialize the MLP encoder.

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
        """
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

    def forward(self, x):
        """Compute a forward pass through the MLP encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape
            ``(batch_size, sequence_length, input_size)``.

        Returns
        -------
        outputs : dict[str, torch.Tensor]
            Dictionary containing:

            - ``"sequence_output"``: all hidden states of shape
              ``(batch_size, 1, sequence_length, hidden_size)``.
            - ``"final_hidden_state"``: same as ``"sequence_output"``.
            - ``"output"``: last state in the sequence of shape
              ``(batch_size, 1, 1, hidden_size)``.
        """
        out = self.model(x)
        out = self.dropout(out)
        out = einops.rearrange(out, "b s d -> b 1 s d")
        return {
            "sequence_output": out,
            "final_hidden_state": out,
            "output": out[:, :, -1:, :],
        }
