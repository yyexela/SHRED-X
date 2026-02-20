"""
SINDy Loss mixin module for neural network regularization.
"""

from typing import cast

import einops
import torch
from jaxtyping import Float
from torch import nn
from torchdiffeq import odeint

from shredx.utils.pytorch_polynomial_features import PolynomialFeatures


class SINDyLossMixin(nn.Module):
    r"""Mixin providing SINDy loss regularization for neural networks.

    Adds learnable SINDy coefficients and methods to compute regularization
    loss based on how well the latent dynamics follow a sparse polynomial ODE.
    Designed to be used with multiple inheritance alongside encoder models.

    Parameters
    ----------
    poly_order : int
        Polynomial order for SINDy library features.
    dt : float
        Time step for computing derivatives.
    hidden_size : int
        Dimension of the hidden state.
    sindy_loss_threshold : float
        Threshold for coefficient sparsification.
    *args
        Additional positional arguments passed to parent class.
    **kwargs
        Additional keyword arguments passed to parent class.

    Notes
    -----
    **Class Methods:**

    **compute_sindy_loss(x):**

    - Calculates SINDy loss based on derivatives with torchdiffeq. Propagates forward all hidden states. Note: batch size and forecast length are combined into the batch dimension.
    - Parameters:
        - x : ``torch.Tensor``. Transformed sequence of shape ``(batch_size, sequence_length, hidden_size)``.
    - Returns:
        - ``torch.Tensor``. SINDy regularization loss.

    **compute_sindy_loss_original(x):**

    - Calculates SINDy loss based on derivatives with a midpoint integration method. For each time step (t0 to t1), integrates in two steps (t0 to t0.5, then t0.5 to t1).
    - Parameters:
        - x : ``torch.Tensor``. Transformed sequence of shape ``(batch_size, sequence_length, hidden_size)``.
    - Returns:
        - ``torch.Tensor``. SINDy regularization loss.

    **thresholding(threshold):**

    - Applies thresholding to SINDy coefficients to enforce sparsity.
    - Parameters:
        - threshold : float, optional. Threshold value. If None, uses the default threshold.
    - Returns:
        - None.
    """

    def __init__(
        self,
        dt: float,
        hidden_size: int,
        sindy_loss_threshold: float,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the SINDyLoss module."""
        # Stupid? Maybe. Works? Yes. Forwards kwargs even though this class uses them.
        kwargs["hidden_size"] = hidden_size
        super().__init__(*args, **kwargs)

        self.poly_order = 1  # force ODE to be symmetric
        self.dt = dt
        self.hidden_size = hidden_size
        self.sindy_loss_threshold = sindy_loss_threshold

        pf = PolynomialFeatures(degree=self.poly_order, interaction_only=False, include_bias=False)
        self.pf = pf
        self.pf.fit(torch.randn(1, self.hidden_size))  # Necessary for output features
        self.library_dim = self.pf.n_output_features_

        # SINDy coefficients (learnable parameters)
        self.coefficients = nn.Parameter(torch.Tensor(self.library_dim, self.hidden_size))
        nn.init.xavier_uniform_(self.coefficients, gain=0.0000000)  # Initialize with small values

        # Coefficient mask for thresholding (not learnable, used for sparsification)
        self.register_buffer("coefficient_mask", torch.ones(self.library_dim, self.hidden_size))

    def compute_sindy_loss(self, x: Float[torch.Tensor, "batch sequence hidden_size"]) -> Float[torch.Tensor, ""]:
        """Calculate SINDy loss based on derivatives with torchdiffeq."""
        batch_size, seq_len, hidden_size = x.shape

        if x.shape[0] < 3:
            return torch.tensor(0.0)

        x_0 = x[:, 0:-1, :]
        x_1 = x[:, 1:, :]

        x_0_poly = einops.rearrange(x_0, "batch seq_len hidden_size -> (batch seq_len) hidden_size")
        library_theta = self.pf.fit_transform(x_0_poly)

        def f(t, y):
            y = y.reshape(library_theta.shape[0], library_theta.shape[1])
            y = y.T
            terms = self.coefficients.to(y.device)
            dy = terms @ y
            dy = dy.T
            return dy.flatten()

        t_eval = torch.linspace(0, 1, 2, device=library_theta.device).float()
        library_theta_flat = library_theta.flatten()
        rollout = odeint(f, library_theta_flat, t_eval, method="rk4")

        # Reshape update back to (forecast, batch_size, seq_len, hidden_size)
        rollout = einops.rearrange(
            rollout,
            "n (b s h) -> n b s h",
            n=t_eval.shape[0],
            b=batch_size,
            s=seq_len - 1,
            h=hidden_size,
        )

        self.coefficient_mask = cast(torch.Tensor, self.coefficients)
        effective_coefficients = self.coefficients * self.coefficient_mask

        rollout = cast(torch.Tensor, rollout)
        step_loss = torch.mean(torch.square(rollout[1] - x_1))
        l2_loss = torch.mean(torch.square(effective_coefficients))
        total_loss = step_loss + 0.001 * l2_loss

        return total_loss

    def thresholding(self, threshold: float | None = None) -> None:
        """Apply thresholding to SINDy coefficients to enforce sparsity."""
        if threshold is None:
            threshold = self.sindy_loss_threshold

        with torch.no_grad():
            mask = torch.abs(self.coefficients.data) > threshold
            self.coefficients.data *= mask
            self.coefficient_mask.copy_(mask.float())
