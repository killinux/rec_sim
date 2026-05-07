import pytest
from rec_sim.runner import run_simulation, SimulationConfig, SimulationResult
from rec_sim.baseline.distribution import ArchetypeDistribution


def _make_dist():
    return ArchetypeDistribution(
        archetype_id=0, n_users=100, proportion=1.0,
        watch_ratio_mean=0.5, watch_ratio_std=0.15,
        watch_ratio_beta_a=2.0, watch_ratio_beta_b=3.0,
        duration_mean=10000, duration_std=3000,
        duration_log_mu=9.2, duration_log_sigma=0.5,
    )


def test_run_simulation_basic():
    config = SimulationConfig(n_agents=10, videos_per_session=5, seed=42)
    result = run_simulation(config, distributions=[_make_dist()])
    assert isinstance(result, SimulationResult)
    assert len(result.logs) <= 10 * 5
    assert len(result.logs) > 0
    assert result.summary["n_agents"] == 10
    assert 0 < result.summary["avg_watch_pct"] < 1


def test_run_simulation_logs_have_required_fields():
    config = SimulationConfig(n_agents=5, videos_per_session=3, seed=42)
    result = run_simulation(config, distributions=[_make_dist()])
    log = result.logs[0]
    assert "session_id" in log
    assert "agent_id" in log
    assert "action" in log
    assert "fidelity_tag" in log


def test_run_simulation_fidelity():
    config = SimulationConfig(n_agents=50, videos_per_session=10, seed=42)
    result = run_simulation(config, distributions=[_make_dist()])
    assert "fidelity" in result.summary
    assert "F_overall" in result.summary["fidelity"]
