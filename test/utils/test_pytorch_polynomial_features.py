import pytest
import torch
from shredx.utils.pytorch_polynomial_features import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures as SklearnPolyFeatures


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
def test_polynomial_features_matches_sklearn_success(degree, interaction_only, include_bias):
    """Output matches sklearn.preprocessing.PolynomialFeatures for given config."""
    X = torch.randn(10, 3, requires_grad=True)

    pf = PolynomialFeatures(
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only,
    )
    X_poly = pf.fit_transform(X)

    sklearn_pf = SklearnPolyFeatures(
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only,
    )
    X_sklearn = sklearn_pf.fit_transform(X.detach().numpy())

    assert pf.n_output_features_ == sklearn_pf.n_output_features_, (
        "Custom and sklearn polynomial feature counts do not match"
    )
    diff = torch.abs(X_poly.detach() - torch.tensor(X_sklearn)).max()
    assert diff.item() <= 1e-6, f"Max difference with sklearn: {diff.item():.2e}"


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
def test_polynomial_features_gradients_success(degree, interaction_only, include_bias):
    """Gradients can be computed through the transformation."""
    X = torch.randn(10, 3, requires_grad=True)

    pf = PolynomialFeatures(
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only,
    )
    X_poly = pf.fit_transform(X)
    loss = X_poly.sum()
    loss.backward()

    assert X.grad is not None
    assert not torch.isnan(X.grad).any()
    assert not torch.isinf(X.grad).any()


def test_polynomial_features_degree_fail():
    with pytest.raises(ValueError):
        PolynomialFeatures(degree=0)


def test_polynomial_features_get_feature_names_out_success():
    pf = PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)
    pf.fit(torch.randn(10, 3))
    feature_names = pf.get_feature_names_out()
    assert len(feature_names) == 10
    assert feature_names[0] == "1"
    assert feature_names[1] == "x0"
    assert feature_names[2] == "x1"
    assert feature_names[3] == "x2"
    assert feature_names[4] == "x0^2"
