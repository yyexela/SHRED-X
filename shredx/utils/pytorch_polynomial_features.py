"""PyTorch implementation of polynomial feature generation.
Drop-in replacement for sklearn.preprocessing.PolynomialFeatures with gradient support.
"""

import torch
import numpy as np
from scipy.special import comb
from itertools import combinations_with_replacement, combinations


class PolynomialFeatures(torch.nn.Module):
    """
    PyTorch implementation of polynomial features that supports backpropagation.

    Drop-in replacement for sklearn.preprocessing.PolynomialFeatures with
    gradient computation support for use in neural network training.
    """

    def __init__(self, degree: int, interaction_only: bool = False, include_bias: bool = True):
        """
        Initialize polynomial features.

        Args:
            degree (int): Maximum degree of polynomial features (must be >= 1)
            interaction_only (bool): If True, only include interaction features
                (products of distinct features), not powers (default: False)
            include_bias (bool): If True, include a bias column of ones (default: True)

        Raises:
            ValueError: If degree is less than 1
        """
        super(PolynomialFeatures, self).__init__()

        if degree < 1:
            raise ValueError("Degree must be at least 1")

        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

    def transform(self, x):
        """
        Transform data to polynomial features.

        Args:
            x (torch.Tensor): Input tensor of shape (n_samples, n_features_in)

        Returns:
            torch.Tensor: Transformed tensor of shape (n_samples, n_output_features_)
        """
        comb_f = combinations if self.interaction_only else combinations_with_replacement

        # Create combinations
        output = []
        for d in range(1, self.degree + 1):
            for combo in comb_f(range(self.n_feature_in), d):
                output.append(torch.prod(x[:, combo], dim=1))
        output = torch.stack(output, dim=1)

        # Add bias
        if self.include_bias:
            bias = torch.ones(x.shape[0], 1, device=x.device)
            output = torch.cat([bias, output], dim=1)

        return output

    def _combo_to_dict(self, combo):
        """
        Convert a combination of feature indices to a dictionary of powers.

        Helper function for get_feature_names_out(). Counts occurrences of each
        feature index to determine its power in the polynomial term.

        Example: combo (0, 0, 1, 3) -> {0: 2, 1: 1, 3: 1} (x0^2 * x1 * x3)

        Args:
            combo (tuple): Tuple of feature indices representing a polynomial term

        Returns:
            dict: Dictionary mapping feature indices to their powers
        """
        # Initialize an empty dictionary to store the counts
        count_dict = {}

        # Iterate over each element in the tuple
        for num in combo:
            count_dict[num] = count_dict.get(num, 0) + 1

        return count_dict

    def get_feature_names_out(self):
        """
        Get the feature names of the polynomial features.

        Must be called after fit() to access the number of input features.

        Returns:
            np.ndarray: Array of feature name strings (dtype=object),
                e.g., ["1", "x0", "x1", "x0^2", "x0 x1", "x1^2"]
        """
        comb_f = combinations if self.interaction_only else combinations_with_replacement
        feature_names = [f"x{i}" for i in range(self.n_feature_in)]

        # Create combinations
        output = []

        if self.include_bias:
            output.append("1")

        for d in range(1, self.degree + 1):
            for combo in comb_f(range(self.n_feature_in), d):
                combo_dict = self._combo_to_dict(combo)
                output.append(
                    " ".join(
                        [
                            (f"{feature_names[idx]}^{power}" if power > 1 else feature_names[idx])
                            for idx, power in combo_dict.items()
                        ]
                    )
                )

        output = np.asarray(output, dtype=object)

        return output

    def fit(self, X):
        """
        Fit the polynomial features transformer to the data.

        Computes and stores the number of input features and output features
        based on the polynomial degree and settings.

        Args:
            X (torch.Tensor): Input tensor of shape (n_samples, n_features_in)

        Returns:
            PolynomialFeatures: Returns self for method chaining
        """
        self.n_feature_in = X.shape[-1]

        self.n_output_features_ = int(
            sum([comb(self.n_feature_in, d, repetition=(not self.interaction_only)) for d in range(1, self.degree + 1)])
        )

        if self.include_bias:
            self.n_output_features_ += 1

        return self

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Convenience method that calls fit() followed by transform().

        Args:
            X (torch.Tensor): Input tensor of shape (n_samples, n_features_in)

        Returns:
            torch.Tensor: Transformed tensor of shape (n_samples, n_output_features_)
        """
        self.fit(X)
        return self.transform(X)
