"""Generate standardized JSON reports from simulation results."""
from __future__ import annotations
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from rec_sim.runner import SimulationResult, SimulationConfig
from rec_sim.baseline.distribution import ArchetypeDistribution
from rec_sim.fidelity.metrics import kl_divergence, js_divergence, wasserstein_1d, composite_fidelity


def generate_report(
    result: SimulationResult,
    distributions: list[ArchetypeDistribution],
    config: SimulationConfig,
    output_path: str | Path | None = None,
) -> dict:
    logs = result.logs
    watch_logs = [l for l in logs if l["action"] == "watch"]
    skip_logs = [l for l in logs if l["action"] == "skip"]
    exit_logs = [l for l in logs if l["action"] == "exit_app"]

    sim_watch_pcts = [l["watch_pct"] for l in watch_logs]
    real_wr_mean = float(np.mean([d.watch_ratio_mean for d in distributions]))

    sim_wr_hist, bin_edges = np.histogram(sim_watch_pcts, bins=20, range=(0, 1), density=True)
    real_samples = np.concatenate([d.sample_watch_ratios(200, seed=42) for d in distributions])
    real_wr_hist, _ = np.histogram(real_samples, bins=20, range=(0, 1), density=True)

    sim_wr_hist_safe = sim_wr_hist / sim_wr_hist.sum() if sim_wr_hist.sum() > 0 else sim_wr_hist
    real_wr_hist_safe = real_wr_hist / real_wr_hist.sum() if real_wr_hist.sum() > 0 else real_wr_hist

    kl = float(kl_divergence(real_wr_hist_safe, sim_wr_hist_safe))
    js = float(js_divergence(real_wr_hist_safe, sim_wr_hist_safe))
    wass = float(wasserstein_1d(real_samples, np.array(sim_watch_pcts)))

    cat_counts = {}
    cat_watch_pcts = {}
    for l in logs:
        cat = str(l.get("category", "unknown"))
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        if l["action"] == "watch":
            cat_watch_pcts.setdefault(cat, []).append(l["watch_pct"])

    cat_avg_wr = {c: float(np.mean(v)) for c, v in cat_watch_pcts.items()}

    layer_counts = {"rule": 0, "parametric": 0, "llm": 0, "cached": 0}
    for l in logs:
        tag = l.get("fidelity_tag", "unknown")
        layer_counts[tag] = layer_counts.get(tag, 0) + 1

    l0_factors = [l.get("l0_factor", 1.0) for l in logs]
    l0_bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for f in l0_factors:
        if f < 0.2: l0_bins["0.0-0.2"] += 1
        elif f < 0.4: l0_bins["0.2-0.4"] += 1
        elif f < 0.6: l0_bins["0.4-0.6"] += 1
        elif f < 0.8: l0_bins["0.6-0.8"] += 1
        else: l0_bins["0.8-1.0"] += 1

    context_data = {}
    for l in logs:
        ctx = l.get("context", {})
        net = str(ctx.get("network", "unknown"))
        sess = str(ctx.get("session_type", "unknown"))
        context_data.setdefault("network", {}).setdefault(net, []).append(l.get("watch_pct", 0))
        context_data.setdefault("session_type", {}).setdefault(sess, []).append(l.get("watch_pct", 0))

    network_avg = {k: float(np.mean(v)) for k, v in context_data.get("network", {}).items()}
    session_type_avg = {k: float(np.mean(v)) for k, v in context_data.get("session_type", {}).items()}

    archetype_counts = {}
    for l in logs:
        aid = l.get("agent_id", 0)
        archetype_counts[aid] = archetype_counts.get(aid, 0) + 1

    f_metrics = {
        "watch_ratio_kl": kl,
        "watch_ratio_js": js,
        "watch_ratio_wasserstein": wass,
    }
    f_max = {"watch_ratio_kl": 2.0, "watch_ratio_js": 0.5, "watch_ratio_wasserstein": 0.5}
    f_overall = float(composite_fidelity(f_metrics, f_max))

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_agents": config.n_agents,
            "videos_per_session": config.videos_per_session,
            "seed": config.seed,
            "n_archetypes": len(distributions),
            "total_steps": len(logs),
        },
        "summary": {
            "avg_watch_pct": float(np.mean(sim_watch_pcts)) if sim_watch_pcts else 0,
            "median_watch_pct": float(np.median(sim_watch_pcts)) if sim_watch_pcts else 0,
            "watch_rate": len(watch_logs) / max(len(logs), 1),
            "skip_rate": len(skip_logs) / max(len(logs), 1),
            "exit_rate": len(exit_logs) / max(len(logs), 1),
            "like_rate": sum(1 for l in logs if l.get("liked")) / max(len(logs), 1),
            "comment_rate": sum(1 for l in logs if l.get("commented")) / max(len(logs), 1),
            "share_rate": sum(1 for l in logs if l.get("shared")) / max(len(logs), 1),
        },
        "fidelity": {
            "F_overall": f_overall,
            "watch_ratio_kl": kl,
            "watch_ratio_js": js,
            "watch_ratio_wasserstein": wass,
            "target_avg_wr": real_wr_mean,
            "actual_avg_wr": float(np.mean(sim_watch_pcts)) if sim_watch_pcts else 0,
            "target_wr_distribution": [float(x) for x in real_wr_hist_safe],
            "simulated_wr_distribution": [float(x) for x in sim_wr_hist_safe],
            "distribution_bin_edges": [float(x) for x in bin_edges],
        },
        "category_breakdown": {
            "counts": cat_counts,
            "avg_watch_pct": cat_avg_wr,
        },
        "decision_layer": {
            "fidelity_tag_counts": layer_counts,
            "l0_factor_distribution": l0_bins,
            "avg_l0_factor": float(np.mean(l0_factors)),
        },
        "context_analysis": {
            "by_network": network_avg,
            "by_session_type": session_type_avg,
        },
        "archetype_distribution": {
            "archetype_proportions": {str(d.archetype_id): d.proportion for d in distributions},
            "archetype_wr_means": {str(d.archetype_id): d.watch_ratio_mean for d in distributions},
        },
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report
