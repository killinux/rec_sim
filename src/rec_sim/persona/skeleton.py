"""Generate persona skeletons using Latin Hypercube Sampling."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import qmc
from rec_sim.baseline.distribution import ArchetypeDistribution


@dataclass
class PersonaSkeleton:
    agent_id: int
    archetype_id: int
    watch_ratio_baseline: float
    duration_baseline: float
    stall_tolerance: float
    quality_sensitivity: float
    fatigue_rate: float
    first_session_patience: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def generate_skeletons(
    distributions: list[ArchetypeDistribution],
    n_agents: int = 1000,
    seed: int = 42,
) -> list[PersonaSkeleton]:
    rng = np.random.default_rng(seed)
    total_prop = sum(d.proportion for d in distributions) or 1.0
    allocations = _allocate_agents(distributions, n_agents, total_prop)

    sampler = qmc.LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n=n_agents)

    skeletons = []
    agent_id = 0
    for dist, count in zip(distributions, allocations):
        for i in range(count):
            if agent_id >= n_agents:
                break
            s = samples[agent_id]
            wr = dist.sample_watch_ratios(1, seed=seed + agent_id)[0]
            dur = dist.sample_durations(1, seed=seed + agent_id)[0]
            skeleton = PersonaSkeleton(
                agent_id=agent_id,
                archetype_id=dist.archetype_id,
                watch_ratio_baseline=float(wr),
                duration_baseline=float(dur),
                stall_tolerance=float(qmc.scale(s[0:1].reshape(1, -1), [500], [5000])[0, 0]),
                quality_sensitivity=float(s[1]),
                fatigue_rate=float(qmc.scale(s[2:3].reshape(1, -1), [0.01], [0.15])[0, 0]),
                first_session_patience=int(qmc.scale(s[3:4].reshape(1, -1), [2], [8])[0, 0]),
            )
            skeletons.append(skeleton)
            agent_id += 1
    return skeletons


def _allocate_agents(distributions, n_agents, total_prop):
    raw = [d.proportion / total_prop * n_agents for d in distributions]
    floors = [int(r) for r in raw]
    remainders = [r - f for r, f in zip(raw, floors)]
    deficit = n_agents - sum(floors)
    top_indices = sorted(range(len(remainders)), key=lambda i: -remainders[i])
    for i in range(deficit):
        floors[top_indices[i]] += 1
    return floors
