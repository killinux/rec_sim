"""Scale simulation from N agents to target population via generative modeling."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from sklearn.mixture import GaussianMixture
from rec_sim.runner import SimulationResult
from rec_sim.fidelity.metrics import wasserstein_1d, js_divergence


@dataclass
class ScaledPopulation:
    """Represents a scaled-up population with weighted representative users."""
    n_representatives: int
    target_population: int
    weights: np.ndarray
    features: np.ndarray
    feature_names: list[str]
    gmm_n_components: int
    quality_metrics: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Generate aggregated metrics for the scaled population."""
        weighted_means = {}
        for i, name in enumerate(self.feature_names):
            weighted_means[name] = float(np.average(self.features[:, i], weights=self.weights))

        return {
            "target_population": self.target_population,
            "n_representatives": self.n_representatives,
            "weighted_means": weighted_means,
            "quality": self.quality_metrics,
            "weight_stats": {
                "min": float(self.weights.min()),
                "max": float(self.weights.max()),
                "median": float(np.median(self.weights)),
                "std": float(self.weights.std()),
            },
        }


def extract_agent_features(result: SimulationResult) -> tuple[np.ndarray, list[str]]:
    """Extract per-agent behavioral features from simulation logs."""
    agent_data = {}
    for log in result.logs:
        aid = log.get("agent_id", 0)
        agent_data.setdefault(aid, {"watch_pcts": [], "actions": [], "l0_factors": []})
        if log["action"] == "watch":
            agent_data[aid]["watch_pcts"].append(log["watch_pct"])
        agent_data[aid]["actions"].append(log["action"])
        agent_data[aid]["l0_factors"].append(log.get("l0_factor", 1.0))

    feature_names = [
        "avg_watch_pct", "watch_pct_std", "completion_rate",
        "skip_rate", "avg_l0_factor", "n_videos",
    ]

    rows = []
    for aid in sorted(agent_data.keys()):
        d = agent_data[aid]
        wps = d["watch_pcts"] or [0.0]
        total = len(d["actions"])
        rows.append([
            float(np.mean(wps)),
            float(np.std(wps)) if len(wps) > 1 else 0.0,
            sum(1 for a in d["actions"] if a == "watch") / max(total, 1),
            sum(1 for a in d["actions"] if a == "skip") / max(total, 1),
            float(np.mean(d["l0_factors"])),
            float(total),
        ])

    return np.array(rows), feature_names


def fit_and_scale(
    result: SimulationResult,
    target_population: int = 1_000_000_000,
    n_representatives: int = 10_000,
    n_components: int = 10,
    seed: int = 42,
) -> ScaledPopulation:
    """Fit a GMM to agent features and generate a scaled population.

    Steps:
    1. Extract per-agent behavioral features from simulation logs
    2. Fit a Gaussian Mixture Model to the feature space
    3. Sample n_representatives new individuals from the GMM
    4. Assign weights so they sum to target_population
    5. Validate: check that sampled distribution matches original
    """
    rng = np.random.default_rng(seed)

    # Step 1: Extract features
    features, feature_names = extract_agent_features(result)
    n_agents = features.shape[0]

    if n_agents < n_components:
        n_components = max(1, n_agents // 2)

    # Step 2: Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=seed, covariance_type="full")
    gmm.fit(features)

    # Step 3: Sample representatives
    sampled, component_labels = gmm.sample(n_representatives)
    # Clip to valid ranges
    sampled[:, 0] = np.clip(sampled[:, 0], 0, 1)  # avg_watch_pct
    sampled[:, 1] = np.clip(sampled[:, 1], 0, 1)  # watch_pct_std
    sampled[:, 2] = np.clip(sampled[:, 2], 0, 1)  # completion_rate
    sampled[:, 3] = np.clip(sampled[:, 3], 0, 1)  # skip_rate
    sampled[:, 4] = np.clip(sampled[:, 4], 0, 1)  # avg_l0_factor
    sampled[:, 5] = np.clip(sampled[:, 5], 1, 1000)  # n_videos

    # Step 4: Assign weights via GMM component proportions
    # Each representative's weight = target_population * P(component) / n_per_component
    component_counts = np.bincount(component_labels, minlength=n_components)
    component_weights = gmm.weights_
    weights = np.zeros(n_representatives)
    for i in range(n_representatives):
        comp = component_labels[i]
        n_in_comp = max(component_counts[comp], 1)
        weights[i] = target_population * component_weights[comp] / n_in_comp

    # Normalize to exact target
    weights = weights * (target_population / weights.sum())

    # Step 5: Quality check
    quality = {}
    for i, name in enumerate(feature_names):
        orig = features[:, i]
        samp = sampled[:, i]
        quality[f"{name}_wasserstein"] = float(wasserstein_1d(orig, samp))

    # JS divergence on avg_watch_pct distribution
    orig_hist, _ = np.histogram(features[:, 0], bins=20, range=(0, 1), density=True)
    samp_hist, _ = np.histogram(sampled[:, 0], bins=20, range=(0, 1), density=True)
    orig_safe = orig_hist / orig_hist.sum() if orig_hist.sum() > 0 else orig_hist + 1e-10
    samp_safe = samp_hist / samp_hist.sum() if samp_hist.sum() > 0 else samp_hist + 1e-10
    quality["watch_pct_js"] = float(js_divergence(orig_safe, samp_safe))

    # Self-consistency: subsample and compare
    subsample_idx = rng.choice(n_representatives, size=min(n_agents, n_representatives), replace=False)
    subsample = sampled[subsample_idx]
    for i, name in enumerate(feature_names):
        quality[f"{name}_subsample_delta"] = float(abs(
            np.mean(features[:, i]) - np.mean(subsample[:, i])
        ))

    return ScaledPopulation(
        n_representatives=n_representatives,
        target_population=target_population,
        weights=weights,
        features=sampled,
        feature_names=feature_names,
        gmm_n_components=n_components,
        quality_metrics=quality,
    )


def generate_traffic_report(scaled: ScaledPopulation) -> dict:
    """Generate a traffic report from the scaled population."""
    summary = scaled.summary()

    # Segment by engagement level
    avg_wr = scaled.features[:, 0]
    segments = {
        "heavy_users": {"mask": avg_wr > 0.7, "label": "Heavy (WR>70%)"},
        "medium_users": {"mask": (avg_wr > 0.3) & (avg_wr <= 0.7), "label": "Medium (30-70%)"},
        "light_users": {"mask": avg_wr <= 0.3, "label": "Light (WR<30%)"},
    }

    segment_report = {}
    for seg_name, seg in segments.items():
        mask = seg["mask"]
        if mask.sum() == 0:
            continue
        seg_weights = scaled.weights[mask]
        seg_features = scaled.features[mask]
        segment_report[seg_name] = {
            "label": seg["label"],
            "user_count": int(seg_weights.sum()),
            "proportion": float(seg_weights.sum() / scaled.target_population),
            "avg_watch_pct": float(np.average(seg_features[:, 0], weights=seg_weights)),
            "avg_completion_rate": float(np.average(seg_features[:, 2], weights=seg_weights)),
            "avg_skip_rate": float(np.average(seg_features[:, 3], weights=seg_weights)),
        }

    return {
        "population": {
            "target": scaled.target_population,
            "representatives": scaled.n_representatives,
            "gmm_components": scaled.gmm_n_components,
        },
        "overall": summary["weighted_means"],
        "segments": segment_report,
        "quality": summary["quality"],
        "weight_distribution": summary["weight_stats"],
    }
