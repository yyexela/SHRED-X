"""Transformer encoders for sequence modeling.

Implements standard and SINDy-augmented transformer encoders compatible
with encoder–decoder architectures.
"""

import copy
import torch
import einops
import torch.nn as nn
from jaxtyping import Float
from typing import Optional, Tuple, List, cast
import torch.nn.functional as F
from shredx.modules.sindy_layer import SINDyLayer
from shredx.modules.positional_encoding import PositionalEncoding
from shredx.modules.sindy_loss_mixin import SINDyLossMixin


class MultiHeadAttention(nn.Module):
    r"""Standard multi-head attention mechanism.

    Implements scaled dot-product attention with multiple heads,
    supporting both same and different query/key/value dimensions.

    Parameters
    ----------
    E_q : int
        Size of embedding dimension for query.
    E_k : int
        Size of embedding dimension for key.
    E_v : int
        Size of embedding dimension for value.
    E_total : int
        Total embedding dimension of combined heads post input projection.
        Each head has dimension ``E_total // n_heads``.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability for attention weights.
    bias : bool
        Whether to add bias to input/output projections.
    dtype : torch.dtype, optional
        Data type for parameters.
    device : str, optional
        Device to place the model on. Default is ``"cpu"``.

    Raises
    ------
    ValueError
        If ``E_total`` is not divisible by ``n_heads``.
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        n_heads: int,
        dropout: float,
        bias: bool,
        dtype: Optional[torch.dtype],
        device: str = "cpu",
    ) -> None:
        """Initialize ``MultiHeadAttention``."""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        if E_total % n_heads != 0:
            raise ValueError("Embedding dim is not divisible by n_heads")
        self.E_head = E_total // n_heads
        self.bias = bias

    def forward(
        self,
        query: Float[torch.Tensor, "N L_q E_qk"],
        key: Float[torch.Tensor, "N L_kv E_qk"],
        value: Float[torch.Tensor, "N L_kv E_v"],
        is_causal=True,
    ) -> Float[torch.Tensor, "N L_t E_q"]:
        """Apply input projection, split heads, run SDPA, and project output."""
        # Step 1. Apply input projection
        result = self.packed_proj(query)
        query, key, value = torch.chunk(result, 3, dim=-1)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, n_heads, E_head) -> (N, n_heads, L_t, E_head)
        query = query.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, n_heads, E_head) -> (N, n_heads, L_s, E_head)
        key = key.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, n_heads, E_head) -> (N, n_heads, L_s, E_head)
        value = value.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, n_heads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout, is_causal=is_causal)
        # (N, n_heads, L_t, E_head) -> (N, L_t, n_heads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


class MultiHeadSINDyAttention(nn.Module):
    r"""Multi-head attention with SINDy-based latent space rollout.

    Replaces standard scaled dot-product attention output with ODE-based
    rollouts using learned SINDy dynamics. Each attention head has its
    own SINDy layer for independent dynamics learning.

    Parameters
    ----------
    E_q : int
        Size of embedding dimension for query.
    E_k : int
        Size of embedding dimension for key.
    E_v : int
        Size of embedding dimension for value.
    E_total : int
        Total embedding dimension of combined heads post input projection.
        Each head has dimension ``E_total // n_heads``.
    n_heads : int
        Number of attention heads.
    forecast_length : int
        Number of future timesteps to predict via ODE rollout.
    dropout : float
        Dropout probability for attention weights.
    strict_symmetry : bool
        If True, enforce strict symmetry in SINDy coefficients.
    bias : bool
        Whether to add bias to input/output projections.
    dtype : torch.dtype, optional
        Data type for parameters.
    device : str, optional
        Device to place the model on. Default is ``"cpu"``.

    Raises
    ------
    ValueError
        If ``E_total`` is not divisible by ``n_heads``.
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        n_heads: int,
        forecast_length: int,
        dropout: float,
        strict_symmetry: bool,
        bias: bool,
        dtype: Optional[torch.dtype],
        device="cpu",
    ) -> None:
        """Initialize ``MultiHeadSINDyAttention``."""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Class variables
        self.n_heads = n_heads
        self.dropout = dropout
        self.forecast_length = forecast_length
        self.device = device
        self.strict_symmetry = strict_symmetry

        # Create projection matrices (Q K V)
        self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)

        # Create output projection matrix
        self.out_proj = nn.Linear(E_total, E_q, bias=bias, **factory_kwargs)

        # Check if embedding dim is divisible by n_heads
        if E_total % n_heads != 0:
            raise ValueError("Embedding dim is not divisible by n_heads")
        self.E_head = E_total // n_heads

        # Initialize SINDy Attention layers
        self.sindy_layers = nn.ModuleList(
            [
                SINDyLayer(
                    hidden_size=self.E_head,
                    forecast_length=self.forecast_length,
                    device=self.device,
                    strict_symmetry=self.strict_symmetry,
                )
                for _ in range(self.n_heads)
            ]
        )

    def forward(
        self,
        query: Float[torch.Tensor, "N L_q E_qk"],
        key: Float[torch.Tensor, "N L_kv E_qk"],
        value: Float[torch.Tensor, "N L_kv E_v"],
        is_causal=True,
    ) -> Float[torch.Tensor, "N forecast_length L_t E_q"]:
        """Apply input projection, split heads, run SDPA, SINDy rollout, and project output."""
        # Step 1. Apply input projection
        result = self.packed_proj(query)
        query, key, value = torch.chunk(result, 3, dim=-1)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, n_heads, E_head) -> (N, n_heads, L_t, E_head)
        query = query.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, n_heads, E_head) -> (N, n_heads, L_s, E_head)
        key = key.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, n_heads, E_head) -> (N, n_heads, L_s, E_head)
        value = value.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, n_heads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        )  # 2 x 6 x 20 x 2
        # (N, n_heads, L_t, E_head) -> (N, L_t, n_heads, E_head) -> (N, L_t, E_total)

        batch_size, _, seq_len, _ = attn_output.shape

        # Step 4. Per-head pysindy
        # coeffs: n_terms x hidden_dim
        # library_Theta: (batch x window len) x n_terms
        sindy_attn_output = []
        for i in range(self.n_heads):
            # Extract head values
            head = attn_output[:, i, :, :]
            # Reshape for input to sindy layer
            head = einops.rearrange(head, "b s h -> (b s) h")
            # Pass through sindy layer
            rollout = self.sindy_layers[i](head)

            # Reshape update back to (batch_size, forecast_length, sequence_length, hidden_size)
            rollout = einops.rearrange(
                rollout,
                "(b s) n h -> b n s h",
                n=self.forecast_length,
                b=batch_size,
                s=seq_len,
                h=self.E_head,
            )
            sindy_attn_output.append(rollout)
        sindy_attn_output = torch.stack(sindy_attn_output, dim=2)

        attn_output = sindy_attn_output.transpose(2, 3).flatten(-2)

        # Step 5. Apply output projection (ff network)
        attn_output = self.out_proj(attn_output)

        return attn_output


class TransformerEncoderLayer(nn.Module):
    r"""Single transformer encoder layer.

    Consists of multi-head self-attention followed by a position-wise
    feedforward network, with residual connections and layer normalization.

    Parameters
    ----------
    d_model : int
        Model dimension (input/output size).
    n_heads : int
        Number of attention heads.
    dim_feedforward : int
        Dimension of feedforward network hidden layer.
    dropout : float
        Dropout probability.
    activation : nn.Module
        Activation function for feedforward network.
    layer_norm_eps : float
        Epsilon for layer normalization.
    norm_first : bool
        If True, apply layer norm before attention/feedforward.
    bias : bool
        Whether to use bias in linear layers.
    dtype : torch.dtype, optional
        Data type for parameters.
    device : str, optional
        Device to place the model on. Default is ``"cpu"``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: nn.Module,
        layer_norm_eps: float,
        norm_first: bool,
        bias: bool,
        dtype: Optional[torch.dtype],
        device: str = "cpu",
    ) -> None:
        """Initialize ``TransformerEncoderLayer``."""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            d_model,
            d_model,
            d_model,
            n_heads,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def _sa_block(
        self, x: Float[torch.Tensor, "batch seq_len d_model"], is_causal: bool
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """Self-attention block with dropout."""
        x = self.self_attn(x, x, x, is_causal=is_causal)
        return self.dropout1(x)

    def _ff_block(
        self, x: Float[torch.Tensor, "batch seq_len d_model"]
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """Feedforward block with dropout."""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(
        self, src: Float[torch.Tensor, "batch seq_len d_model"], is_causal: bool = True
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """Forward pass through the encoder layer."""
        x = src
        if self.norm_first:
            sa_out = self._sa_block(self.norm1(x), is_causal)
            if sa_out.dim() == 4:
                # SINDy/rollout attention returns (batch, forecast_length, seq_len, d_model)
                x = x.unsqueeze(1).expand_as(sa_out) + sa_out
            else:
                x = x + sa_out
            x = x + self._ff_block(self.norm2(x))
        else:
            out_1 = self._sa_block(x, is_causal)
            if out_1.dim() == 4:
                # Required for rollout transformer
                out_2 = out_1 + x.unsqueeze(1).expand_as(out_1)
            else:
                # Standard transformers
                out_2 = out_1 + x
            x = self.norm1(out_2)
            x = self.norm2(x + self._ff_block(x))
        return x


class TransformerEncoderModule(nn.Module):
    r"""Stack of transformer encoder layers.

    Applies multiple encoder layers sequentially with optional final normalization.

    Parameters
    ----------
    encoder_layer : nn.Module
        Single encoder layer to clone.
    num_layers : int
        Number of encoder layers.
    norm : nn.Module, optional
        Final layer normalization.
    dtype : torch.dtype, optional
        Data type for parameters.
    device : str, optional
        Device to place the model on. Default is ``"cpu"``.
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module],
        dtype: Optional[torch.dtype],
        device: str = "cpu",
    ) -> None:
        """Initialize ``TransformerEncoderModule``."""
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self, src: Float[torch.Tensor, "batch seq_len d_model"], is_causal: bool = True
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """Forward pass through all encoder layers."""
        output = src
        for mod in self.layers:
            output = mod(output, is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoder(nn.Module):
    r"""Standard transformer encoder for sequence modeling.

    Implements input embedding, positional encoding, and stacked encoder layers
    for sequence-to-sequence transformation.

    Parameters
    ----------
    d_model : int
        Input dimension.
    n_heads : int
        Number of attention heads.
    num_layers : int
        Number of encoder layers.
    dim_feedforward : int
        Dimension of feedforward network.
    dropout : float
        Dropout probability.
    activation : nn.Module
        Activation function for feedforward layers.
    layer_norm_eps : float
        Epsilon for layer normalization.
    norm_first : bool
        Whether to apply layer norm before attention.
    bias : bool
        Whether to use bias in linear layers.
    input_length : int
        Maximum input sequence length.
    hidden_size : int
        Hidden dimension size.
    device : str, optional
        Device to place the model on. Default is ``"cpu"``.
    **kwargs
        Additional keyword arguments (ignored).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: nn.Module,
        layer_norm_eps: float,
        norm_first: bool,
        bias: bool,
        input_length: int,
        hidden_size: int,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initialize ``TransformerEncoder``."""
        super().__init__()

        self.input_embedding = nn.Linear(d_model, hidden_size, bias=bias, device=device)

        encoder_layer = TransformerEncoderLayer(
            hidden_size,  # Fix: Use d_model instead of hidden_size
            n_heads,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
            dtype=None,
            device=device,
        )

        encoder_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, bias=bias, device=device)
        self.encoder = TransformerEncoderModule(
            encoder_layer,
            num_layers,
            encoder_norm,
            dtype=None,
            device=device,
        )

        self.pos_encoder = PositionalEncoding(
            d_model=hidden_size,
            sequence_length=input_length + 10,  # Provide some buffer
            dropout=dropout,
            device=device,
        )

    def forward(
        self,
        src: Float[torch.Tensor, "batch seq_len d_model"],
        is_causal: bool = True,
    ) -> Tuple[Float[torch.Tensor, "batch 1 seq_len hidden_size"], None]:
        """Forward pass through the transformer encoder."""
        # Embed input
        x_embedded = self.input_embedding(src)

        # Apply positional encoding
        x_pos_encoded = self.pos_encoder(x_embedded)

        transformer_output = self.encoder(
            x_pos_encoded,
            is_causal=is_causal,
        )

        transformer_output = einops.rearrange(transformer_output, "b s d -> b 1 s d")

        return (transformer_output, None)


class SINDyLossTransformerEncoder(SINDyLossMixin, TransformerEncoder):
    r"""Transformer encoder with SINDy loss regularization.

    Combines a standard transformer encoder with SINDy-based regularization
    that encourages the learned representations to follow sparse polynomial ODEs.

    Parameters
    ----------
    d_model : int
        Input dimension of the model.
    n_heads : int
        Number of attention heads.
    dim_feedforward : int
        Dimension of feedforward network.
    dropout : float
        Dropout probability.
    hidden_size : int
        Hidden dimension size.
    input_length : int
        Length of input sequences.
    num_layers : int
        Number of transformer encoder layers.
    dt : float
        Time step for SINDy derivatives.
    sindy_loss_threshold : float
        Threshold for coefficient sparsification.
    activation : nn.Module
        Activation function for feedforward layers.
    bias : bool
        Whether to use bias in linear layers.
    layer_norm_eps : float
        Epsilon for layer normalization.
    norm_first : bool
        Whether to apply layer norm before attention.
    device : str, optional
        Device to place the model on. Default is ``"cpu"``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        hidden_size: int,
        input_length: int,
        num_layers: int,
        dt: float,
        sindy_loss_threshold: float,
        activation: nn.Module,
        bias: bool,
        layer_norm_eps: float,
        norm_first: bool,
        device: str = "cpu",
    ) -> None:
        """Initialize ``SINDyLossTransformerEncoder``."""
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            hidden_size=hidden_size,
            input_length=input_length,
            num_layers=num_layers,
            dt=dt,
            sindy_loss_threshold=sindy_loss_threshold,
            activation=activation,
            bias=bias,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            device=device,
        )

    def forward(  # pyrefly: ignore[bad-override]
        self,
        src: Float[torch.Tensor, "batch seq_len d_model"],
        is_causal: bool = True,
    ) -> Tuple[Float[torch.Tensor, "batch 1 seq_len hidden_size"], Float[torch.Tensor, ""]]:
        """Forward pass through the transformer encoder with SINDy loss."""
        # Embed input
        x_embedded = self.input_embedding(src)

        # Apply positional encoding
        x_pos_encoded = self.pos_encoder(x_embedded)

        transformer_output = self.encoder(
            x_pos_encoded,
            is_causal=is_causal,
        )

        sindy_loss = self.compute_sindy_loss(transformer_output)

        transformer_output = einops.rearrange(transformer_output, "b s d -> b 1 s d")

        return (transformer_output, sindy_loss)


class SINDyAttentionTransformerEncoder(TransformerEncoder):
    r"""Transformer encoder with SINDy-based attention in the final layer.

    Extends the standard Transformer by replacing the attention mechanism
    in the last encoder layer with ``MultiHeadSINDyAttention``, enabling
    ODE-based latent space rollouts for multi-step forecasting.

    Parameters
    ----------
    d_model : int
        Input dimension of the model.
    n_heads : int
        Number of attention heads.
    forecast_length : int
        Number of future timesteps to predict.
    num_layers : int
        Number of transformer encoder layers.
    dim_feedforward : int
        Dimension of feedforward network.
    dropout : float
        Dropout probability.
    activation : nn.Module
        Activation function for feedforward layers.
    layer_norm_eps : float
        Epsilon for layer normalization.
    norm_first : bool
        Whether to apply layer norm before attention.
    bias : bool
        Whether to use bias in linear layers.
    strict_symmetry : bool
        If True, enforce strict symmetry in SINDy coefficients.
    input_length : int
        Length of input sequences.
    hidden_size : int
        Hidden dimension size.
    device : str, optional
        Device to place the model on. Default is ``"cpu"``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        forecast_length: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: nn.Module,
        layer_norm_eps: float,
        norm_first: bool,
        bias: bool,
        strict_symmetry: bool,
        input_length: int,
        hidden_size: int,
        device: str = "cpu",
    ) -> None:
        """Initialize ``SINDyAttentionTransformerEncoder``."""
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
            input_length=input_length,
            hidden_size=hidden_size,
            device=device,
        )

        self.encoder.layers[-1].self_attn = MultiHeadSINDyAttention(
            hidden_size,
            hidden_size,
            hidden_size,
            hidden_size,
            n_heads,
            forecast_length=forecast_length,
            dropout=dropout,
            bias=bias,
            strict_symmetry=strict_symmetry,
            device=device,
            dtype=None,
        )

        self.n_heads = n_heads

    def print_sindy_layer_coefficients(self) -> None:
        """Print the SINDy layer coefficients for all attention heads in human-readable format."""
        # coefficients: n_heads x ((library terms + 1 (for linear) terms) x library_terms equations)
        sindy_attn = cast(MultiHeadSINDyAttention, self.encoder.layers[-1].self_attn)
        for j in range(self.n_heads):
            print(f"Head {j}:")
            layer_j = cast(SINDyLayer, sindy_attn.sindy_layers[j])
            coefficients = layer_j.get_dense_sindy_coefficients()
            library = layer_j.pf.get_feature_names_out()
            for k in range(coefficients.shape[1]):
                print(f"Hidden layer {k}:")
                output_str = ""
                for lib_term_idx in range(coefficients.shape[0]):
                    output_str += f"{coefficients[lib_term_idx][k].item():.3f} \\cdot {library[lib_term_idx]} + "
                print(output_str[:-3])
            print()

    def get_sindy_layer_coefficients_eigenvalues(self) -> List[Float[torch.Tensor, "hidden_size"]]:  # noqa: F821
        """Get eigenvalues of SINDy coefficient matrices for all attention heads."""
        with torch.no_grad():
            eigvs_l = []
            sindy_attn = cast(MultiHeadSINDyAttention, self.encoder.layers[-1].self_attn)
            for i in range(self.n_heads):
                layer_i = cast(SINDyLayer, sindy_attn.sindy_layers[i])
                eigvs_l.append(layer_i.get_eigenvalues())
            return eigvs_l

    def get_sindy_layer_coefficients_sum(self) -> Float[torch.Tensor, ""]:
        """Sum of squared SINDy coefficients in all heads of the last layer."""
        with torch.no_grad():
            sindy_sum = torch.tensor(0.0)
            layer: TransformerEncoderLayer = cast(TransformerEncoderLayer, self.encoder.layers[-1])
            sindy_attn = cast(MultiHeadSINDyAttention, layer.self_attn)
            for i in range(sindy_attn.n_heads):
                layer_i = cast(SINDyLayer, sindy_attn.sindy_layers[i])
                sindy_sum += torch.sqrt((layer_i.get_dense_sindy_coefficients() ** 2).sum())
        return sindy_sum

    def set_forecast_length(self, forecast_length):
        """Set the forecast length for all SINDy attention layers."""
        # Set forecast length to expected plot length
        sindy_attn = cast(MultiHeadSINDyAttention, self.encoder.layers[-1].self_attn)
        sindy_attn.forecast_length = forecast_length
        for i in range(sindy_attn.n_heads):
            layer_i = cast(SINDyLayer, sindy_attn.sindy_layers[i])
            layer_i.forecast_length = forecast_length

    def threshold_sindy_layer_coefficients(self, threshold, verbose=False):
        """Threshold all SINDy coefficients in all heads of the last layer."""
        layer = self.encoder.layers[-1]
        with torch.no_grad():
            sindy_attn = cast(MultiHeadSINDyAttention, layer.self_attn)
            for i in range(sindy_attn.n_heads):
                layer_i = cast(SINDyLayer, sindy_attn.sindy_layers[i])
                mask = torch.abs(layer_i.get_raw_sindy_coefficients()) > threshold
                layer_i.set_raw_sindy_coefficients(layer_i.get_raw_sindy_coefficients() * mask)
                if verbose:
                    print(
                        f"SINDyAttentionTransformer: Applied threshold {threshold} to head {i}. Non-zero coeffs: {mask.sum().item()}/{mask.numel()}"
                    )
        if verbose:
            print()

    def get_dense_sindy_coefficients(self):
        """Return a list of dense SINDy coefficient matrices, one per attention head."""
        odes = []
        sindy_attn = cast(MultiHeadSINDyAttention, self.encoder.layers[-1].self_attn)
        for i in range(self.n_heads):
            layer_i = cast(SINDyLayer, sindy_attn.sindy_layers[i])
            odes.append(layer_i.get_dense_sindy_coefficients())
        return odes

    def forward(
        self,
        src: Float[torch.Tensor, "batch seq_len d_model"],
        is_causal=True,
    ) -> Tuple[Float[torch.Tensor, "batch forecast_length seq_len d_model"], None]:
        """Forward pass through the SINDy attention transformer."""
        # Embed input
        x_embedded = self.input_embedding(src)

        # Apply positional encoding
        x_pos_encoded = self.pos_encoder(x_embedded)

        transformer_output = self.encoder(
            x_pos_encoded,
            is_causal=is_causal,
        )

        return (transformer_output, None)


class SINDyAttentionSINDyLossTransformerEncoder(SINDyLossMixin, SINDyAttentionTransformerEncoder):
    r"""Transformer encoder with SINDy attention and SINDy loss regularization.

    Combines SINDy-based attention in the final layer with SINDy loss
    regularization for ODE-based latent rollouts and sparse dynamics.

    Parameters
    ----------
    d_model : int
        Input dimension of the model.
    n_heads : int
        Number of attention heads.
    forecast_length : int
        Number of future timesteps to predict.
    num_layers : int
        Number of transformer encoder layers.
    dim_feedforward : int
        Dimension of feedforward network.
    dropout : float
        Dropout probability.
    activation : nn.Module
        Activation function for feedforward layers.
    layer_norm_eps : float
        Epsilon for layer normalization.
    norm_first : bool
        Whether to apply layer norm before attention.
    bias : bool
        Whether to use bias in linear layers.
    strict_symmetry : bool
        If True, enforce strict symmetry in SINDy coefficients.
    input_length : int
        Length of input sequences.
    hidden_size : int
        Hidden dimension size.
    sindy_loss_threshold : float
        Threshold for coefficient sparsification.
    dt : float
        Time step for SINDy derivatives.
    device : str, optional
        Device to place the model on. Default is ``"cpu"``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        forecast_length: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: nn.Module,
        layer_norm_eps: float,
        norm_first: bool,
        bias: bool,
        strict_symmetry: bool,
        input_length: int,
        hidden_size: int,
        sindy_loss_threshold: float,
        dt: float,
        device: str = "cpu",
    ) -> None:
        """Initialize ``SINDyAttentionSINDyLossTransformerEncoder``."""
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            forecast_length=forecast_length,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
            input_length=input_length,
            strict_symmetry=strict_symmetry,
            hidden_size=hidden_size,
            sindy_loss_threshold=sindy_loss_threshold,
            dt=dt,
            device=device,
        )

    def forward(  # pyrefly: ignore[bad-override]
        self,
        src: Float[torch.Tensor, "batch seq_len d_model"],
        is_causal=True,
    ) -> Tuple[Float[torch.Tensor, "batch forecast_length seq_len d_model"], Float[torch.Tensor, ""]]:
        """Forward pass through the SINDy attention transformer with SINDy loss."""
        # Embed input
        x_embedded = self.input_embedding(src)

        # Apply positional encoding
        x_pos_encoded = self.pos_encoder(x_embedded)

        transformer_output = self.encoder(
            x_pos_encoded,
            is_causal=is_causal,
        )

        # Compute SINDy Loss, put forecast dimension into batch dimension
        transformer_output_3d = einops.rearrange(transformer_output, "b n s d -> (b n) s d")
        sindy_loss = self.compute_sindy_loss(transformer_output_3d)

        return (transformer_output, sindy_loss)


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Create N deep copies of a PyTorch module.

    Parameters
    ----------
    module : nn.Module
        PyTorch module to clone.
    N : int
        Number of clones to create.

    Returns
    -------
    nn.ModuleList
        List of N independent copies of the module.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
