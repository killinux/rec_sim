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
    """Compare P(watch_ratio | category) between real and simulated.

    Uses both absolute delta AND Spearman rank correlation.
    Rank correlation is more robust when real/sim have different absolute
    scales (e.g. fully-observed dataset vs simulated app behavior).
    """
    from scipy.stats import spearmanr

    results = {}
    real_means = {}
    sim_means = {}

    for cat in sorted(set(list(real_wr_by_cat.keys()) + list(sim_wr_by_cat.keys()))):
        real_vals = real_wr_by_cat.get(cat, [])
        sim_vals = sim_wr_by_cat.get(cat, [])
        if len(real_vals) < 5 or len(sim_vals) < 5:
            continue
        real_arr = np.array(real_vals)
        sim_arr = np.array(sim_vals)
        r_mean = float(real_arr.mean())
        s_mean = float(sim_arr.mean())
        real_means[cat] = r_mean
        sim_means[cat] = s_mean
        results[cat] = {
            "real_mean": r_mean,
            "sim_mean": s_mean,
            "delta": float(abs(r_mean - s_mean)),
            "wasserstein": float(wasserstein_1d(real_arr, sim_arr)),
        }

    # Compute Spearman rank correlation across categories
    common_cats = sorted(set(real_means.keys()) & set(sim_means.keys()))
    if len(common_cats) >= 3:
        real_ranks = [real_means[c] for c in common_cats]
        sim_ranks = [sim_means[c] for c in common_cats]
        rho, p_value = spearmanr(real_ranks, sim_ranks)
        results["_rank_correlation"] = {
            "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
            "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
            "n_categories": len(common_cats),
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

    # Conditional: use Spearman rank correlation (robust to scale differences)
    # rho ∈ [-1, 1], convert to distance: 1 - rho ∈ [0, 2], lower = better
    rank_info = conditional_results.get("_rank_correlation", {})
    if rank_info:
        rho = rank_info.get("spearman_rho", 0)
        metrics["conditional_rank_dist"] = float(1.0 - rho)  # 0=perfect, 2=inverse
    else:
        cond_deltas = [v["delta"] for v in conditional_results.values() if isinstance(v, dict) and "delta" in v]
        if cond_deltas:
            metrics["conditional_rank_dist"] = float(np.mean(cond_deltas))

    max_acceptable = {
        "watch_ratio_js": 0.3,
        "category_js": 0.3,
        "activity_wasserstein": 0.3,
        "correlation_distance": 2.0,
        "conditional_rank_dist": 1.5,
    }
    # Weights: conditional gets lower weight because fully-observed datasets
    # have fundamentally different conditional WR patterns than simulated app behavior.
    # This is a known limitation — proper weighting requires real app data.
    weights = {
        "watch_ratio_js": 1.0,
        "category_js": 1.0,
        "activity_wasserstein": 1.0,
        "correlation_distance": 1.0,
        "conditional_rank_dist": 0.5,  # down-weighted: fully-observed vs sim mismatch
    }
    max_acceptable = {k: v for k, v in max_acceptable.items() if k in metrics}
    weights = {k: v for k, v in weights.items() if k in metrics}

    f_overall = float(composite_fidelity(metrics, max_acceptable, weights))

    return {
        "F_multidim": f_overall,
        "per_dimension": {k: float(1.0 - min(v / max_acceptable[k], 1.0)) for k, v in metrics.items()},
        "raw_metrics": {k: float(v) for k, v in metrics.items()},
    }
