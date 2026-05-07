"""End-to-end: load KuaiRec -> cluster -> generate personas -> simulate -> check fidelity."""
import pytest
from rec_sim.baseline.loader import load_kuairec, load_kuairec_items
from rec_sim.baseline.clustering import extract_user_features, cluster_users
from rec_sim.baseline.distribution import extract_archetype_distributions
from rec_sim.runner import run_simulation, SimulationConfig


@pytest.fixture(scope="module")
def archetype_distributions():
    interactions = load_kuairec(use_small=True)
    items = load_kuairec_items()
    features, _ = extract_user_features(interactions, items)
    labels, _ = cluster_users(features, n_clusters=20)
    user_ids = interactions.groupby("user_id").first().index.values
    dists = extract_archetype_distributions(interactions, labels, user_ids)
    return dists


def test_e2e_pipeline(archetype_distributions):
    config = SimulationConfig(n_agents=100, videos_per_session=10, seed=42)
    result = run_simulation(config, archetype_distributions)

    assert result.summary["n_agents"] == 100
    assert result.summary["total_steps"] > 0
    assert 0 < result.summary["avg_watch_pct"] < 1
    print(f"\n=== E2E Results ===")
    print(f"Agents: {result.summary['n_agents']}")
    print(f"Total steps: {result.summary['total_steps']}")
    print(f"Avg watch %: {result.summary['avg_watch_pct']:.3f}")
    print(f"Exit rate: {result.summary['exit_rate']:.3f}")
    print(f"Skip rate: {result.summary['skip_rate']:.3f}")
    print(f"Fidelity F: {result.summary['fidelity']['F_overall']:.3f}")
    print(f"  target WR: {result.summary['fidelity']['target_watch_ratio']:.3f}")
    print(f"  actual WR: {result.summary['fidelity']['actual_watch_ratio']:.3f}")
