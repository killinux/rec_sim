import pytest
import numpy as np
from rec_sim.baseline.distribution import ArchetypeDistribution
from rec_sim.persona.skeleton import generate_skeletons, PersonaSkeleton


def _make_dist(arch_id, proportion, wr_mean):
    return ArchetypeDistribution(
        archetype_id=arch_id, n_users=100, proportion=proportion,
        watch_ratio_mean=wr_mean, watch_ratio_std=0.15,
        watch_ratio_beta_a=2.0, watch_ratio_beta_b=3.0,
        duration_mean=10000, duration_std=3000,
        duration_log_mu=9.2, duration_log_sigma=0.5,
    )


def test_generate_skeletons_count():
    dists = [_make_dist(0, 0.6, 0.5), _make_dist(1, 0.4, 0.7)]
    skeletons = generate_skeletons(dists, n_agents=100, seed=42)
    assert len(skeletons) == 100


def test_generate_skeletons_proportional():
    dists = [_make_dist(0, 0.7, 0.5), _make_dist(1, 0.3, 0.7)]
    skeletons = generate_skeletons(dists, n_agents=1000, seed=42)
    arch0_count = sum(1 for s in skeletons if s.archetype_id == 0)
    assert 650 < arch0_count < 750


def test_skeleton_has_required_fields():
    dists = [_make_dist(0, 1.0, 0.5)]
    skeletons = generate_skeletons(dists, n_agents=10, seed=42)
    s = skeletons[0]
    assert isinstance(s, PersonaSkeleton)
    assert 0 <= s.watch_ratio_baseline <= 1
    assert s.duration_baseline > 0
    assert s.stall_tolerance > 0
    assert 0 <= s.quality_sensitivity <= 1


def test_skeletons_have_variance():
    dists = [_make_dist(0, 1.0, 0.5)]
    skeletons = generate_skeletons(dists, n_agents=50, seed=42)
    baselines = [s.watch_ratio_baseline for s in skeletons]
    assert np.std(baselines) > 0.01
