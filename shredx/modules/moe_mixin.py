"""
Mixture of Experts mixin module for SINDy layer forecasting.
"""

import torch
from jaxtyping import Float
import torch.nn as nn
from typing import cast

from shredx.modules.sindy_layer import SindyLayer


class MOE_SINDy_Layer_Helpers_Mixin:
    r"""Mixin providing helper methods for Mixture of Experts models with SINDy layers.

    Provides common functionality for printing, modifying, and analyzing SINDy
    coefficients across multiple expert networks.

    Notes
    -----
    **Class Methods:**

    **print_sindy_layer_coefficients():**

    - Prints the SINDy coefficients for all experts in a human-readable format (polynomial expression per hidden dimension).
    - Returns:
        - None.

    **set_forecast_length(forecast_length):**

    - Sets the forecast length for the model and all expert SINDy layers.
    - Parameters:
        - forecast_length : int. Number of timesteps to forecast.
    - Returns:
        - None.

    **get_sindy_layer_coefficients_eigenvalues():**

    - Returns the eigenvalues of SINDy coefficient matrices for all experts.
    - Returns:
        - list. List of eigenvalue tensors, one per expert.

    **get_sindy_layer_coefficients_sum():**

    - Computes the sum of absolute SINDy coefficients across all experts (used as a sparsity regularization term).
    - Returns:
        - float. Sum of square roots of absolute coefficient sums.

    **threshold_sindy_layer_coefficients(threshold, verbose):**

    - Applies sparsity thresholding: sets coefficients with absolute value below the threshold to zero for all experts.
    - Parameters:
        - threshold : float. Coefficients below this are zeroed.
        - verbose : bool, optional. If True, print information about thresholding. Default is ``False``.
    - Returns:
        - None.
    """

    # Declared for type checkers; actual values are set by the subclass (e.g. MOEGRUEncoder, MOEMLPEncoder).
    n_experts: int
    experts: nn.ModuleList
    forecast_length: int

    def print_sindy_layer_coefficients(self) -> None:
        """Print the SINDy coefficients for all experts in a human-readable format."""
        for j in range(self.n_experts):
            expert = self.experts[j]
            expert = cast(SindyLayer, expert)
            print(f"Expert {j}:")
            coefficients = expert.get_dense_sindy_coefficients()
            library = expert.pf.get_feature_names_out()
            for k in range(coefficients.shape[1]):
                print(f"Hidden layer {k}:")
                output_str = ""
                for idx in range(coefficients.shape[0]):
                    output_str += f"{coefficients[idx][k].item():.3f} \\cdot {library[idx]} + "
                print(output_str[:-3])
            print()

    def set_forecast_length(self, forecast_length: int) -> None:
        """Set the forecast length for the model and all expert SINDy layers."""
        self.forecast_length = forecast_length
        for expert in self.experts:
            expert = cast(SindyLayer, expert)
            expert.forecast_length = forecast_length

    def get_sindy_layer_coefficients_eigenvalues(self) -> list[Float[torch.Tensor, "hidden_size"]]:  # noqa: F821
        """Get the eigenvalues of SINDy coefficient matrices for all SINDy experts."""
        with torch.no_grad():
            eigvs_l = []
            for expert in self.experts:
                expert = cast(SindyLayer, expert)
                eigvs_l.append(expert.get_eigenvalues())
            return eigvs_l

    def get_sindy_layer_coefficients_sum(self) -> float:
        """Compute the sum of absolute SINDy coefficients across all experts (sparsity regularization)."""
        with torch.no_grad():
            sindy_sum: float = 0.0
            for expert in self.experts:
                expert = cast(SindyLayer, expert)
                sindy_sum += float(torch.sqrt(torch.abs(expert.get_raw_sindy_coefficients()).sum()))
            return sindy_sum

    def threshold_sindy_layer_coefficients(self, threshold: float, verbose: bool = False) -> None:
        """Apply sparsity thresholding to SINDy coefficients for all experts."""
        with torch.no_grad():
            for i in range(self.n_experts):
                expert = self.experts[i]
                expert = cast(SindyLayer, expert)
                mask = torch.abs(expert.get_raw_sindy_coefficients()) > threshold
                expert.set_raw_sindy_coefficients(expert.get_raw_sindy_coefficients() * mask)
                if verbose:
                    print(
                        f"MOE_SINDy_Layer_Helpers_Mixin: Applied threshold {threshold} to expert {i}. Non-zero coeffs: {mask.sum().item()}/{mask.numel()}"
                    )
        if verbose:
            print()
