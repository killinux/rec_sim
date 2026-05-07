"""Tests for calibration loop."""
from __future__ import annotations
import pytest
from rec_sim.baseline.distribution import ArchetypeDistribution
from rec_sim.runner import SimulationConfig, RealDataContext
from rec_sim.calibration.loop import calibrate, CalibrationConfig, CalibrationResult


def _make_dist(arch_id, proportion, wr_mean, beta_a=2.0, beta_b=3.0):
    return ArchetypeDistribution(
        archetype_id=arch_id, n_users=100, proportion=proportion,
        watch_ratio_mean=wr_mean, watch_ratio_std=0.15,
        watch_ratio_beta_a=beta_a, watch_ratio_beta_b=beta_b,
        duration_mean=10000, duration_std=3000,
        duration_log_mu=9.2, duration_log_sigma=0.5,
    )


def test_calibration_runs():
    dists = [_make_dist(0, 0.6, 0.5), _make_dist(1, 0.4, 0.7)]
    sim_config = SimulationConfig(n_agents=20, videos_per_session=5, seed=42)
    real_data = RealDataContext()
    cal_config = CalibrationConfig(max_outer_iterations=1, max_mid_iterations=2)

    result = calibrate(dists, sim_config, real_data, cal_config)
    assert isinstance(result, CalibrationResult)
    assert result.iterations_used > 0
    assert len(result.history) > 0
    assert result.initial_f >= 0


def test_calibration_improves_or_stable():
    dists = [_make_dist(0, 0.6, 0.4, 2.0, 5.0), _make_dist(1, 0.4, 0.3, 1.5, 4.0)]
    sim_config = SimulationConfig(n_agents=30, videos_per_session=5, seed=42)
    real_data = RealDataContext()
    cal_config = CalibrationConfig(
        max_outer_iterations=1, max_mid_iterations=5,
        f_multidim_target=0.99,  # unreachable, forces iterations
    )

    result = calibrate(dists, sim_config, real_data, cal_config)
    assert result.iterations_used >= 2
    assert len(result.history) >= 2
    # F should not degrade catastrophically
    assert result.final_f >= result.initial_f * 0.5


def test_calibration_history_has_metrics():
    dists = [_make_dist(0, 1.0, 0.5)]
    sim_config = SimulationConfig(n_agents=10, videos_per_session=3, seed=42)
    real_data = RealDataContext()
    cal_config = CalibrationConfig(max_outer_iterations=1, max_mid_iterations=3)

    result = calibrate(dists, sim_config, real_data, cal_config)
    for step in result.history:
        assert "iteration" in step
        assert "F_multidim" in step
        assert "per_dimension" in step
        assert "status" in step


def test_calibration_converges_with_easy_target():
    dists = [_make_dist(0, 1.0, 0.5)]
    sim_config = SimulationConfig(n_agents=10, videos_per_session=5, seed=42)
    real_data = RealDataContext()
    cal_config = CalibrationConfig(
        max_outer_iterations=1, max_mid_iterations=5,
        f_multidim_target=0.01,  # very easy target
    )

    result = calibrate(dists, sim_config, real_data, cal_config)
    assert result.converged
    assert result.iterations_used == 1
