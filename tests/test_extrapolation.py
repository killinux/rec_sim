"""Tests for extrapolation layer."""
from __future__ import annotations
import pytest
import numpy as np
from rec_sim.baseline.distribution import ArchetypeDistribution
from rec_sim.runner import run_simulation, SimulationConfig
from rec_sim.extrapolation.scaler import (
    extract_agent_features, fit_and_scale, generate_traffic_report, ScaledPopulation,
)


def _make_dist():
    return ArchetypeDistribution(
        archetype_id=0, n_users=100, proportion=1.0,
        watch_ratio_mean=0.5, watch_ratio_std=0.15,
        watch_ratio_beta_a=2.0, watch_ratio_beta_b=3.0,
        duration_mean=10000, duration_std=3000,
        duration_log_mu=9.2, duration_log_sigma=0.5,
    )


@pytest.fixture(scope="module")
def sim_result():
    config = SimulationConfig(n_agents=50, videos_per_session=10, seed=42)
    return run_simulation(config, [_make_dist()])


def test_extract_agent_features(sim_result):
    features, names = extract_agent_features(sim_result)
    assert features.shape[0] == 50  # n_agents
    assert features.shape[1] == 6
    assert len(names) == 6
    assert "avg_watch_pct" in names


def test_fit_and_scale_basic(sim_result):
    scaled = fit_and_scale(sim_result, target_population=1_000_000, n_representatives=500, seed=42)
    assert isinstance(scaled, ScaledPopulation)
    assert scaled.n_representatives == 500
    assert scaled.target_population == 1_000_000
    assert len(scaled.weights) == 500
    assert abs(scaled.weights.sum() - 1_000_000) < 1  # weights sum to target


def test_fit_and_scale_preserves_distribution(sim_result):
    scaled = fit_and_scale(sim_result, target_population=10_000, n_representatives=200, seed=42)
    # Sampled avg_watch_pct should be in similar range as original
    orig_features, _ = extract_agent_features(sim_result)
    orig_mean = orig_features[:, 0].mean()
    scaled_mean = np.average(scaled.features[:, 0], weights=scaled.weights)
    assert abs(orig_mean - scaled_mean) < 0.15  # reasonable tolerance


def test_fit_and_scale_quality_metrics(sim_result):
    scaled = fit_and_scale(sim_result, target_population=100_000, n_representatives=300, seed=42)
    assert "watch_pct_js" in scaled.quality_metrics
    assert scaled.quality_metrics["watch_pct_js"] >= 0


def test_generate_traffic_report(sim_result):
    scaled = fit_and_scale(sim_result, target_population=1_000_000_000, n_representatives=1000, seed=42)
    report = generate_traffic_report(scaled)
    assert "population" in report
    assert report["population"]["target"] == 1_000_000_000
    assert "segments" in report
    assert "quality" in report
    assert "overall" in report
    assert "avg_watch_pct" in report["overall"]


def test_scale_to_billion(sim_result):
    scaled = fit_and_scale(sim_result, target_population=1_000_000_000, n_representatives=5000, seed=42)
    report = generate_traffic_report(scaled)
    total_users = sum(s["user_count"] for s in report["segments"].values())
    # All users should be accounted for in segments
    assert total_users > 0
    assert total_users <= 1_000_000_000 * 1.01  # within 1% due to rounding
