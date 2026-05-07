"""Multi-dimensional fidelity: category distribution, conditional, correlation."""
from __future__ import annotations
import numpy as np
from rec_sim.fidelity.metrics import kl_divergence, js_divergence, wasserstein_1d, composite_fidelity


def category_distribution_fidelity(
    real_cat_counts: dict[str, int],
    sim_cat_counts: dict[str, int],
) -> dict:
    """Compare category consumption distributions."""
    all_cats = sorted(set(list(real_cat_counts.keys()) + list(sim_cat_counts.keys())))
    real_total = sum(real_cat_counts.values()) or 1
    sim_total = sum(sim_cat_counts.values()) or 1

    real_dist = np.array([real_cat_counts.get(c, 0) / real_total for c in all_cats])
    sim_dist = np.array([sim_cat_counts.get(c, 0) / sim_total for c in all_cats])

    return {
        "kl": float(kl_divergence(real_dist, sim_dist)),
        "js": float(js_divergence(real_dist, sim_dist)),
        "categories": all_cats,
        "real_distribution": [float(x) for x in real_dist],
        "sim_distribution": [float(x) for x in sim_dist],
    }


def conditional_fidelity(
    real_wr_by_cat: dict[str, list[float]],
    sim_wr_by_cat: dict[str, list[float]],
) -> dict:
    """Compare P(watch_ratio | category) between real and simulated."""
    results = {}
    for cat in sorted(set(list(real_wr_by_cat.keys()) + list(sim_wr_by_cat.keys()))):
        real_vals = real_wr_by_cat.get(cat, [])
        sim_vals = sim_wr_by_cat.get(cat, [])
        if len(real_vals) < 5 or len(sim_vals) < 5:
            continue
        real_arr = np.array(real_vals)
        sim_arr = np.array(sim_vals)
        results[cat] = {
            "real_mean": float(real_arr.mean()),
            "sim_mean": float(sim_arr.mean()),
            "delta": float(abs(real_arr.mean() - sim_arr.mean())),
            "wasserstein": float(wasserstein_1d(real_arr, sim_arr)),
        }
    return results


def activity_distribution_fidelity(
    real_interactions_per_user: np.ndarray,
    sim_interactions_per_agent: np.ndarray,
) -> dict:
    """Compare activity level (interactions per user) distributions."""
    return {
        "wasserstein": float(wasserstein_1d(real_interactions_per_user, sim_interactions_per_agent)),
        "real_mean": float(real_interactions_per_user.mean()),
        "sim_mean": float(sim_interactions_per_agent.mean()),
        "real_std": float(real_interactions_per_user.std()),
        "sim_std": float(sim_interactions_per_agent.std()),
    }


def correlation_fidelity(
    real_features: np.ndarray,
    sim_features: np.ndarray,
) -> dict:
    """Compare correlation matrices between real and simulated feature sets."""
    if real_features.shape[1] < 2 or sim_features.shape[1] < 2:
        return {"frobenius_distance": 0.0}
    real_corr = np.corrcoef(real_features.T)
    sim_corr = np.corrcoef(sim_features.T)
    real_corr = np.nan_to_num(real_corr)
    sim_corr = np.nan_to_num(sim_corr)
    dist = float(np.linalg.norm(real_corr - sim_corr, "fro"))
    return {
        "frobenius_distance": dist,
        "normalized_distance": dist / max(real_corr.shape[0], 1),
    }


def compute_multidim_fidelity(
    marginal_metrics: dict[str, float],
    cat_fidelity: dict,
    conditional_results: dict,
    activity_fidelity: dict,
    correlation_result: dict,
) -> dict:
    """Compute the overall multi-dimensional fidelity score."""
    metrics = {
        "watch_ratio_js": marginal_metrics.get("watch_ratio_js", 0),
        "category_js": cat_fidelity.get("js", 0),
        "activity_wasserstein": activity_fidelity.get("wasserstein", 0),
        "correlation_distance": correlation_result.get("normalized_distance", 0),
    }

    cond_deltas = [v["delta"] for v in conditional_results.values()]
    if cond_deltas:
        metrics["conditional_avg_delta"] = float(np.mean(cond_deltas))

    max_acceptable = {
        "watch_ratio_js": 0.3,
        "category_js": 0.3,
        "activity_wasserstein": 50.0,
        "correlation_distance": 2.0,
        "conditional_avg_delta": 0.2,
    }
    max_acceptable = {k: v for k, v in max_acceptable.items() if k in metrics}

    f_overall = float(composite_fidelity(metrics, max_acceptable))

    return {
        "F_multidim": f_overall,
        "per_dimension": {k: float(1.0 - min(v / max_acceptable[k], 1.0)) for k, v in metrics.items()},
        "raw_metrics": {k: float(v) for k, v in metrics.items()},
    }
