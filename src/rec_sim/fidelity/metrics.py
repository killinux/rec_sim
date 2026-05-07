"""Fidelity metrics for comparing real vs simulated distributions."""
import numpy as np
from scipy import stats as sp_stats


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    return float(sp_stats.wasserstein_distance(a, b))


def correlation_matrix_distance(real: np.ndarray, sim: np.ndarray) -> float:
    return float(np.linalg.norm(real - sim, "fro"))


def relative_error(real_val: float, sim_val: float) -> float:
    if abs(real_val) < 1e-10:
        return abs(sim_val)
    return abs(real_val - sim_val) / abs(real_val)


def composite_fidelity(metrics: dict[str, float], max_acceptable: dict[str, float], weights: dict[str, float] | None = None) -> float:
    if weights is None:
        weights = {k: 1.0 for k in metrics}
    total_w = sum(weights[k] for k in metrics)
    f = 0.0
    for k, val in metrics.items():
        w = weights.get(k, 1.0) / total_w
        capped = min(val / max_acceptable[k], 1.0)
        f += w * (1.0 - capped)
    return float(f)
