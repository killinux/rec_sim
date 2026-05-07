from __future__ import annotations
import pytest
import numpy as np
from rec_sim.fidelity.multidim import (
    category_distribution_fidelity, conditional_fidelity,
    activity_distribution_fidelity, correlation_fidelity,
    compute_multidim_fidelity,
)


def test_category_distribution_identical():
    counts = {"food": 100, "tech": 50, "beauty": 30}
    result = category_distribution_fidelity(counts, counts)
    assert result["js"] < 0.01


def test_category_distribution_different():
    real = {"food": 100, "tech": 50}
    sim = {"food": 20, "tech": 80}
    result = category_distribution_fidelity(real, sim)
    assert result["js"] > 0.01


def test_conditional_fidelity_similar():
    real = {"food": [0.6, 0.7, 0.5, 0.8, 0.65], "tech": [0.4, 0.3, 0.5, 0.35, 0.45]}
    sim = {"food": [0.62, 0.68, 0.52, 0.78, 0.63], "tech": [0.38, 0.32, 0.48, 0.37, 0.43]}
    result = conditional_fidelity(real, sim)
    assert "food" in result
    assert result["food"]["delta"] < 0.05


def test_activity_distribution():
    real = np.array([10, 20, 30, 15, 25, 18, 22, 35])
    sim = np.array([12, 18, 28, 17, 23, 20, 24, 33])
    result = activity_distribution_fidelity(real, sim)
    assert result["wasserstein"] >= 0
    assert abs(result["real_mean"] - result["sim_mean"]) < 5


def test_correlation_fidelity_identical():
    features = np.random.randn(100, 5)
    result = correlation_fidelity(features, features)
    assert result["frobenius_distance"] < 0.01


def test_compute_multidim_overall():
    marginal = {"watch_ratio_js": 0.05}
    cat = {"js": 0.03}
    cond = {"food": {"delta": 0.02}, "tech": {"delta": 0.04}}
    activity = {"wasserstein": 5.0}
    corr = {"normalized_distance": 0.3}
    result = compute_multidim_fidelity(marginal, cat, cond, activity, corr)
    assert 0 < result["F_multidim"] <= 1
    assert "per_dimension" in result
