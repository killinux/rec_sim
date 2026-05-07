"""End-to-end: load KuaiRec -> cluster -> generate personas -> simulate -> report -> check fidelity."""
from __future__ import annotations
import pytest
import json
from pathlib import Path
from rec_sim.baseline.loader import load_kuairec, load_kuairec_items
from rec_sim.baseline.clustering import extract_user_features, cluster_users
from rec_sim.baseline.distribution import extract_archetype_distributions
from rec_sim.runner import run_simulation, SimulationConfig
from rec_sim.report import generate_report

REPORT_DIR = Path("/Users/bytedance/Desktop/hehe/research/rec_sim/reports")


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

    report = generate_report(
        result=result,
        distributions=archetype_distributions,
        config=config,
        output_path=REPORT_DIR / "latest_report.json",
    )

    assert result.summary["n_agents"] == 100
    assert result.summary["total_steps"] > 0
    assert 0 < result.summary["avg_watch_pct"] < 1

    assert report["fidelity"]["F_overall"] > 0
    assert "target_wr_distribution" in report["fidelity"]
    assert "simulated_wr_distribution" in report["fidelity"]
    assert len(report["category_breakdown"]["counts"]) > 0

    print(f"\n{'='*50}")
    print(f"  RecSim E2E Report")
    print(f"{'='*50}")
    print(f"  Agents:        {report['metadata']['n_agents']}")
    print(f"  Total steps:   {report['metadata']['total_steps']}")
    print(f"  Archetypes:    {report['metadata']['n_archetypes']}")
    print(f"{'─'*50}")
    print(f"  Avg Watch %:   {report['summary']['avg_watch_pct']:.1%}")
    print(f"  Skip Rate:     {report['summary']['skip_rate']:.1%}")
    print(f"  Exit Rate:     {report['summary']['exit_rate']:.1%}")
    print(f"  Like Rate:     {report['summary']['like_rate']:.1%}")
    print(f"{'─'*50}")
    print(f"  Fidelity F:    {report['fidelity']['F_overall']:.4f}")
    print(f"  KL Divergence: {report['fidelity']['watch_ratio_kl']:.4f}")
    print(f"  JS Divergence: {report['fidelity']['watch_ratio_js']:.4f}")
    print(f"  Wasserstein:   {report['fidelity']['watch_ratio_wasserstein']:.4f}")
    print(f"  Target WR:     {report['fidelity']['target_avg_wr']:.4f}")
    print(f"  Actual WR:     {report['fidelity']['actual_avg_wr']:.4f}")
    print(f"{'─'*50}")
    print(f"  L0 Avg Factor: {report['decision_layer']['avg_l0_factor']:.4f}")
    print(f"  Decision Tags: {report['decision_layer']['fidelity_tag_counts']}")
    print(f"{'='*50}")
    print(f"  Report saved to: {REPORT_DIR / 'latest_report.json'}")
    print(f"  Open dashboard.html and load the JSON to visualize")
    print(f"{'='*50}")
