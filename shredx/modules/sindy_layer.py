"""SINDy Layer module.

Implements a differentiable SINDy (Sparse Identification of Nonlinear Dynamics)
layer for learning interpretable ODEs and performing arbitrary-length forecasting.
"""

import torch
import einops
import torch.nn as nn
from jaxtyping import Float
from torchdiffeq import odeint
from shredx.utils.pytorch_polynomial_features import PolynomialFeatures


class SindyLayer(nn.Module):
    r"""Differentiable SINDy layer for ODE-based forecasting.

    Learns sparse polynomial dynamics from data and uses ODE integration
    for arbitrary-length forecasting. Supports both strict symmetry
    (parameterized via lower triangle) and general coefficient matrices.

    Parameters
    ----------
    hidden_size : int
        Input/output dimension of the layer.
    forecast_length : int
        Number of future timesteps to predict.
    device : str, optional
        Device to place the model on. Default is ``"cpu"``.
    strict_symmetry : bool, optional
        If True, enforces symmetric coefficient matrix via lower triangle
        parameterization. Default is ``True``.
    std_init : float, optional
        Standard deviation for initial coefficients. Default is ``0.1``.
    **kwargs
        Additional keyword arguments (ignored).

    Notes
    -----
    **Class Methods:**

    **get_dense_sindy_coefficients():**

    - Converts symmetric parameters (1D) to a dense matrix when
      ``strict_symmetry`` is True; otherwise returns the stored full matrix.
    - Returns:
        - ``Float[torch.Tensor, "hidden_size hidden_size"]``. Dense SINDy coefficient matrix.

    **get_raw_sindy_coefficients():**

    - Returns raw SINDy coefficients (1D lower triangle or full matrix).
    - Returns:
        - ``Float[torch.Tensor, "library_dim library_dim"]`` when ``strict_symmetry`` is False, else
          ``Float[torch.Tensor, "num_params"]`` (lower triangle 1D).

    **set_raw_sindy_coefficients(coefficients):**

    - Updates the layer parameters in-place from raw SINDy coefficients.
    - Parameters:
        - coefficients : ``Float[torch.Tensor, "library_dim library_dim"]`` | ``Float[torch.Tensor, "num_params"]``.
          Raw SINDy coefficients (same shape as ``get_raw_sindy_coefficients``).
    - Returns:
        - None.

    **dense_matrix_from_symmetric_params(params):**

    - Builds a symmetric matrix by filling the lower triangle and reflecting.
    - Parameters:
        - params : ``Float[torch.Tensor, "num_params"]``. 1D lower-triangle coefficients.
    - Returns:
        - ``Float[torch.Tensor, "library_dim library_dim"]``. Dense symmetric matrix.

    **get_eigenvalues():**

    - Returns eigenvalues of the SINDy coefficient matrix (1j * matrix).
    - Returns:
        - ``Float[torch.Tensor, "hidden_size"]``. Eigenvalues.

    **forward(x):**

    - Integrates the learned ODE (dopri5) over ``forecast_length`` steps to produce multi-step forecasts.
    - Parameters:
        - x : ``Float[torch.Tensor, "batch_size hidden_size"]``. Input state.
    - Returns:
        - ``Float[torch.Tensor, "batch_size forecast_length hidden_size"]``. Rollout of predicted states.
    """

    def __init__(
        self,
        hidden_size: int,
        forecast_length: int,
        device: str = "cpu",
        strict_symmetry: bool = True,
        std_init: float = 0.1,
        **kwargs,
    ) -> None:
        """Initialize the SINDy layer."""
        # Initialize parent class
        super().__init__()

        # Class variables
        self.hidden_size = hidden_size
        self.forecast_length = forecast_length
        self.device = device
        self.strict_symmetry = strict_symmetry

        # Polynomial features
        self.pf = PolynomialFeatures(degree=1, include_bias=False)
        self.pf.fit(torch.randn(1, self.hidden_size))
        self.library_dim = self.pf.n_output_features_

        # Initialize SINDy coefficients (SINDy library)
        # TODO: Initialization? Should be larger I think, or different per-layer for MOE?
        if self.strict_symmetry:
            # Symmetric parameters, builds a 1D list of parameters (lower triangle) that get converted to a dense symmetric matrix
            self.tril_indices = torch.tril_indices(self.library_dim, self.library_dim)
            num_params = (self.library_dim * (self.library_dim + 1)) // 2
            self.triangle_coefficients = nn.Parameter(torch.Tensor(num_params))
            nn.init.normal_(self.triangle_coefficients, mean=0.0, std=std_init)
        else:
            # Random symmetric matrix of shape library_dim x library_dim
            random_tensor = torch.randn(self.library_dim, self.library_dim, device=self.device) * std_init
            sindy_coefficients = (random_tensor + random_tensor.T) / 2.0
            self.sindy_coefficients = nn.Parameter(sindy_coefficients)

    def get_dense_sindy_coefficients(self) -> Float[torch.Tensor, "hidden_size hidden_size"]:
        """Return dense SINDy coefficient matrix."""
        if self.strict_symmetry:
            sindy_coefficients = self.dense_matrix_from_symmetric_params(self.triangle_coefficients)
            return sindy_coefficients
        else:
            return self.sindy_coefficients

    # Set noqa F821 because Ruff is throwing an error for the type hint "num_params"
    def get_raw_sindy_coefficients(
        self,
    ) -> Float[torch.Tensor, "library_dim library_dim"] | Float[torch.Tensor, "num_params"]:  # noqa: F821
        """Return raw SINDy coefficients (1D or full matrix)."""
        if self.strict_symmetry:
            return self.triangle_coefficients
        else:
            return self.sindy_coefficients

    # Set noqa F821 because Ruff is throwing an error for the type hint "num_params"
    def set_raw_sindy_coefficients(
        self,
        coefficients: Float[torch.Tensor, "library_dim library_dim"] | Float[torch.Tensor, "num_params"],  # noqa: F821
    ) -> None:
        """Set raw SINDy coefficients from a tensor."""
        if self.strict_symmetry:
            self.triangle_coefficients.data.copy_(coefficients)
        else:
            self.sindy_coefficients.data.copy_(coefficients)

    # Set noqa F821 because Ruff is throwing an error for the type hint "num_params"
    def dense_matrix_from_symmetric_params(
        self,
        params: Float[torch.Tensor, "num_params"],  # noqa: F821
    ) -> Float[torch.Tensor, "library_dim library_dim"]:
        """Build symmetric dense matrix from lower-triangle parameters."""
        sindy_coefficients = torch.zeros(self.library_dim, self.library_dim, device=params.device)
        self.tril_indices = self.tril_indices.to(sindy_coefficients.device)
        sindy_coefficients[self.tril_indices[0], self.tril_indices[1]] = params
        sindy_coefficients = sindy_coefficients + sindy_coefficients.t() - torch.diag(sindy_coefficients.diag())
        return sindy_coefficients

    # Set noqa F821 because Ruff is throwing an eror for the type hint "hidden_size"
    def get_eigenvalues(self) -> Float[torch.Tensor, "hidden_size"]:  # noqa: F821
        """Return eigenvalues of the SINDy coefficient matrix (1j * matrix)."""
        sindy_coefficients = self.get_dense_sindy_coefficients()
        eigenvalues = torch.linalg.eigvals(sindy_coefficients.to(torch.cfloat) * torch.tensor(1j))
        return eigenvalues

    def forward(
        self, x: Float[torch.Tensor, "batch_size hidden_size"]
    ) -> Float[torch.Tensor, "batch_size forecast_length hidden_size"]:
        """Integrate learned ODE dynamics to produce multi-step forecasts."""
        batch_size, hidden_size = x.shape
        sindy_coefficients = self.get_dense_sindy_coefficients()
        library_Theta = self.pf.fit_transform(x)

        def f(t, y):
            y = y.reshape(library_Theta.shape[0], library_Theta.shape[1])
            y = y.T
            terms = sindy_coefficients.to(y.device)
            terms = terms.to(torch.cfloat) * torch.tensor(1j)
            dy = terms @ y
            dy = dy.T
            return dy.flatten()

        t_eval = torch.arange(1, self.forecast_length + 1, 1, device=library_Theta.device).float()
        library_Theta_flat = library_Theta.flatten()
        library_Theta_flat = library_Theta_flat.to(torch.cfloat)
        rollout = odeint(f, library_Theta_flat, t_eval, method="dopri5", rtol=1e-3, atol=1e-3)
        # For some reason the rollout returning a complex tensor is not caught by pyrefly, so we need to ignore the missing attribute error
        # pyrefly: ignore[missing-attribute]
        rollout = rollout.real
        rollout = rollout.reshape(self.forecast_length, library_Theta.shape[0], library_Theta.shape[1])

        rollout = einops.rearrange(
            rollout,
            "n b h -> b n h",
            n=self.forecast_length,
            b=batch_size,
            h=hidden_size,
        )

        return rollout
