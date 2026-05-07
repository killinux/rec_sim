import pytest
import numpy as np
from rec_sim.fidelity.metrics import kl_divergence, js_divergence, wasserstein_1d, correlation_matrix_distance, composite_fidelity


def test_kl_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-10)


def test_kl_different_distributions():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.5])
    assert kl_divergence(p, q) > 0


def test_js_symmetric():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.5])
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p), abs=1e-10)


def test_js_bounded():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.5])
    assert 0 <= js_divergence(p, q) <= 1


def test_wasserstein_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert wasserstein_1d(a, a) == pytest.approx(0.0, abs=1e-10)


def test_wasserstein_shifted():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    assert wasserstein_1d(a, b) == pytest.approx(1.0, abs=0.1)


def test_correlation_distance_identical():
    m = np.array([[1, 0.5], [0.5, 1]])
    assert correlation_matrix_distance(m, m) == pytest.approx(0.0, abs=1e-10)


def test_composite_fidelity_perfect():
    metrics = {"kl": 0.0, "js": 0.0, "wass": 0.0}
    max_vals = {"kl": 1.0, "js": 1.0, "wass": 10.0}
    f = composite_fidelity(metrics, max_vals)
    assert f == pytest.approx(1.0)


def test_composite_fidelity_half():
    metrics = {"kl": 0.5, "js": 0.5}
    max_vals = {"kl": 1.0, "js": 1.0}
    f = composite_fidelity(metrics, max_vals)
    assert f == pytest.approx(0.5)
