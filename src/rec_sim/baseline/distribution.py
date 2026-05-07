"""Per-archetype distribution extraction and serialization."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats


@dataclass
class ArchetypeDistribution:
    archetype_id: int
    n_users: int
    proportion: float
    watch_ratio_mean: float
    watch_ratio_std: float
    watch_ratio_beta_a: float
    watch_ratio_beta_b: float
    duration_mean: float
    duration_std: float
    duration_log_mu: float
    duration_log_sigma: float

    @classmethod
    def from_data(cls, archetype_id: int, watch_ratios: np.ndarray,
                  durations: np.ndarray, n_users: int, total_users: int = 0):
        wr = np.clip(watch_ratios, 1e-6, 1 - 1e-6)
        wr_mean, wr_std = float(wr.mean()), float(wr.std()) or 0.01
        a, b, _, _ = stats.beta.fit(wr, floc=0, fscale=1)
        dur = durations[durations > 0]
        d_mean, d_std = float(dur.mean()), float(dur.std()) or 1.0
        log_dur = np.log(dur + 1)
        log_mu, log_sigma = float(log_dur.mean()), float(log_dur.std()) or 0.1
        return cls(
            archetype_id=archetype_id, n_users=n_users,
            proportion=n_users / total_users if total_users > 0 else 0.0,
            watch_ratio_mean=wr_mean, watch_ratio_std=wr_std,
            watch_ratio_beta_a=float(a), watch_ratio_beta_b=float(b),
            duration_mean=d_mean, duration_std=d_std,
            duration_log_mu=log_mu, duration_log_sigma=log_sigma,
        )

    def sample_watch_ratios(self, n: int, seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.clip(rng.beta(self.watch_ratio_beta_a, self.watch_ratio_beta_b, size=n), 0, 1)

    def sample_durations(self, n: int, seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.lognormal(self.duration_log_mu, self.duration_log_sigma, size=n)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArchetypeDistribution":
        return cls(**d)


def extract_archetype_distributions(interactions, labels: np.ndarray, user_ids: np.ndarray) -> list[ArchetypeDistribution]:
    user_label_map = dict(zip(user_ids, labels))
    interactions = interactions.copy()
    interactions["archetype"] = interactions["user_id"].map(user_label_map)
    interactions = interactions.dropna(subset=["archetype"])
    interactions["archetype"] = interactions["archetype"].astype(int)
    total_users = len(user_ids)
    distributions = []
    for arch_id in sorted(interactions["archetype"].unique()):
        group = interactions[interactions["archetype"] == arch_id]
        n_users_in_arch = int((labels == arch_id).sum())
        dist = ArchetypeDistribution.from_data(
            archetype_id=arch_id, watch_ratios=group["watch_ratio"].values,
            durations=group["duration_ms"].values, n_users=n_users_in_arch, total_users=total_users,
        )
        distributions.append(dist)
    return distributions
