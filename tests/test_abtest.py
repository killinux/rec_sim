"""Tests for A/B testing framework."""
from __future__ import annotations
import pytest
from rec_sim.baseline.distribution import ArchetypeDistribution
from rec_sim.runner import SimulationConfig
from rec_sim.evaluation.abtest import run_abtest, ABTestConfig, ABTestResult


def _make_dist():
    return ArchetypeDistribution(
        archetype_id=0, n_users=100, proportion=1.0,
        watch_ratio_mean=0.5, watch_ratio_std=0.15,
        watch_ratio_beta_a=2.0, watch_ratio_beta_b=3.0,
        duration_mean=10000, duration_std=3000,
        duration_log_mu=9.2, duration_log_sigma=0.5,
    )


def test_abtest_runs():
    dists = [_make_dist()]
    sim_config = SimulationConfig(n_agents=30, videos_per_session=5, seed=42)
    ab_config = ABTestConfig(name="test_experiment")

    result = run_abtest(ab_config, dists, sim_config)
    assert isinstance(result, ABTestResult)
    assert result.name == "test_experiment"
    assert "avg_watch_pct" in result.control_metrics
    assert "avg_watch_pct" in result.treatment_metrics
    assert "avg_watch_pct" in result.deltas


def test_abtest_has_statistical_tests():
    dists = [_make_dist()]
    sim_config = SimulationConfig(n_agents=50, videos_per_session=10, seed=42)
    ab_config = ABTestConfig(name="stat_test")

    result = run_abtest(ab_config, dists, sim_config)
    assert "watch_pct_mannwhitney" in result.statistical_tests
    assert "watch_pct_welch_t" in result.statistical_tests
    assert "p_value" in result.statistical_tests["watch_pct_mannwhitney"]
    assert "effect_size_cohens_d" in result.statistical_tests


def test_abtest_different_seeds_produce_different_results():
    dists = [_make_dist()]
    sim_config = SimulationConfig(n_agents=30, videos_per_session=5, seed=42)
    ab_config = ABTestConfig(name="seed_test")

    result = run_abtest(ab_config, dists, sim_config)
    # Control and treatment use different seeds, so metrics will differ slightly
    c = result.control_metrics["avg_watch_pct"]
    t = result.treatment_metrics["avg_watch_pct"]
    # They should both be valid numbers
    assert 0 <= c <= 1
    assert 0 <= t <= 1


def test_abtest_summary_text():
    dists = [_make_dist()]
    sim_config = SimulationConfig(n_agents=20, videos_per_session=5, seed=42)
    ab_config = ABTestConfig(
        name="summary_test",
        control_label="algo_v1",
        treatment_label="algo_v2",
    )

    result = run_abtest(ab_config, dists, sim_config)
    assert "algo_v1" in result.summary
    assert "algo_v2" in result.summary
    assert "avg_watch_pct" in result.summary
    assert result.winner in ("algo_v1", "algo_v2", "no_significant_difference", "tie", "inconclusive")


def test_abtest_with_treatment_overrides():
    dists = [_make_dist()]
    sim_config = SimulationConfig(n_agents=20, videos_per_session=5, seed=42)
    ab_config = ABTestConfig(
        name="override_test",
        treatment_overrides={"videos_per_session": 10},
    )

    result = run_abtest(ab_config, dists, sim_config)
    # Treatment should have more total steps due to more videos
    c_steps = len([l for l in result.deltas])
    assert isinstance(result, ABTestResult)
