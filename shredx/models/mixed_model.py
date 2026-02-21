import torch
from jaxtyping import Float
from torch import nn


class MixedModel(nn.Module):
    """
    A flexible encoder-decoder model supporting multiple encoder and decoder types.

    Combines various encoder architectures (RNNs, Transformers)
    with decoder architectures (MLP, CNN) based on configuration arguments.

    Parameters
    ----------
    encoder : nn.Module
        The encoder module to use.
    decoder : nn.Module
        The decoder module to use.

    Returns
    -------
    nn.Module: A mixed model with the specified encoder and decoder.

    Notes
    -----
    **Class Methods:**

    **forward(x):**

    - Applies the encoder then decoder to an input batch.
    - Parameters:
        - x : ``Float[torch.Tensor, "batch sequence input_size"]``. Input tensor.
    - Returns:
        - tuple. Tuple containing the final output tensor of shape
          ``(batch_size, forecast_length, sequence_length, out_dim)`` and a dictionary of auxiliary losses.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        """Initialize the MixedModel with specified encoder and decoder."""
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.add_module("encoder", self.encoder)
        self.add_module("decoder", self.decoder)

    def forward(
        self, src: Float[torch.Tensor, "batch sequence input_size"]
    ) -> tuple[
        Float[torch.Tensor, "batch forecast_length sequence_length out_dim"],
        dict[str, Float[torch.Tensor, ""]] | None,
    ]:
        """Apply the encoder then decoder to an input batch."""
        out, encoder_aux_losses = self.encoder(src)
        out, _ = self.decoder(out)

        aux_losses = {}

        if encoder_aux_losses is not None:
            aux_losses.update(encoder_aux_losses)

        if len(aux_losses) == 0:
            aux_losses = None

        return out, aux_losses
