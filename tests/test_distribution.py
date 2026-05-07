import pytest
import json
import numpy as np
from rec_sim.baseline.distribution import ArchetypeDistribution


def test_archetype_distribution_from_data():
    watch_ratios = np.array([0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.2, 0.9, 0.5, 0.6])
    durations = np.array([5000, 10000, 15000, 8000, 12000, 20000, 3000, 25000, 11000, 14000])
    dist = ArchetypeDistribution.from_data(archetype_id=0, watch_ratios=watch_ratios, durations=durations, n_users=10)
    assert dist.archetype_id == 0
    assert dist.n_users == 10
    assert 0 < dist.watch_ratio_mean < 1
    assert dist.duration_mean > 0


def test_archetype_distribution_sample():
    watch_ratios = np.random.beta(2, 3, size=100)
    durations = np.random.lognormal(9, 0.5, size=100)
    dist = ArchetypeDistribution.from_data(0, watch_ratios, durations, 100)
    samples = dist.sample_watch_ratios(50, seed=42)
    assert len(samples) == 50
    assert all(0 <= s <= 1 for s in samples)


def test_archetype_distribution_serializable():
    watch_ratios = np.random.beta(2, 3, size=50)
    durations = np.random.lognormal(9, 0.5, size=50)
    dist = ArchetypeDistribution.from_data(0, watch_ratios, durations, 50)
    d = dist.to_dict()
    text = json.dumps(d)
    restored = ArchetypeDistribution.from_dict(json.loads(text))
    assert abs(restored.watch_ratio_mean - dist.watch_ratio_mean) < 1e-6
