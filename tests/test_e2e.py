"""End-to-end: load KuaiRec -> cluster -> generate personas -> simulate -> report -> check fidelity."""
from __future__ import annotations
import pytest
import numpy as np
from pathlib import Path
from rec_sim.baseline.loader import load_kuairec, load_kuairec_items
from rec_sim.baseline.clustering import extract_user_features, cluster_users
from rec_sim.baseline.distribution import extract_archetype_distributions
from rec_sim.baseline.interest import (
    build_category_map, get_all_categories,
    build_user_interest_vectors, build_archetype_interest_vectors,
)
from rec_sim.runner import run_simulation, SimulationConfig, RealDataContext
from rec_sim.report import generate_report

REPORT_DIR = Path("/Users/bytedance/Desktop/hehe/research/rec_sim/reports")


@pytest.fixture(scope="module")
def real_pipeline():
    interactions = load_kuairec(use_small=True)
    items = load_kuairec_items()
    features, feature_names = extract_user_features(interactions, items)
    labels, _ = cluster_users(features, n_clusters=20)
    user_ids = interactions.groupby("user_id").first().index.values
    dists = extract_archetype_distributions(interactions, labels, user_ids)

    cat_map = build_category_map(items)
    all_cats = get_all_categories(cat_map)
    user_vecs = build_user_interest_vectors(interactions, cat_map, all_cats)
    arch_vecs = build_archetype_interest_vectors(user_vecs, user_ids, labels)

    # Real category distribution from interactions
    real_cat_counts = {}
    for _, row in interactions.iterrows():
        item_id = int(row["item_id"])
        cats = cat_map.get(item_id, [])
        for c in cats:
            key = str(c)
            real_cat_counts[key] = real_cat_counts.get(key, 0) + 1

    # Real category distribution as proportions
    total_cat = sum(real_cat_counts.values()) or 1
    cat_dist = {k: v / total_cat for k, v in real_cat_counts.items()}

    # Real watch_ratio by category
    real_wr_by_cat = {}
    for _, row in interactions.iterrows():
        item_id = int(row["item_id"])
        wr = float(row["watch_ratio"])
        cats = cat_map.get(item_id, [])
        for c in cats:
            real_wr_by_cat.setdefault(str(c), []).append(wr)

    # Real interactions per user
    ipu = interactions.groupby("user_id").size().values.astype(float)

    real_data = RealDataContext(
        category_distribution=cat_dist,
        archetype_interest_vectors=arch_vecs,
        item_ids=list(cat_map.keys()),
        item_category_map=cat_map,
        all_categories=all_cats,
        real_cat_counts=real_cat_counts,
        real_wr_by_category=real_wr_by_cat,
        real_interactions_per_user=ipu,
        real_user_features=features[:, :3],
    )

    return dists, real_data, features


def test_e2e_pipeline(real_pipeline):
    dists, real_data, features = real_pipeline
    config = SimulationConfig(n_agents=100, videos_per_session=10, seed=42)
    result = run_simulation(config, dists, real_data=real_data)

    report = generate_report(
        result=result,
        distributions=dists,
        config=config,
        output_path=REPORT_DIR / "latest_report.json",
        real_data=real_data,
    )

    assert result.summary["n_agents"] == 100
    assert result.summary["total_steps"] > 0
    assert 0 < result.summary["avg_watch_pct"] < 1
    assert report["fidelity"]["F_overall"] > 0
    assert report["fidelity_multidim"]["F_multidim"] > 0

    fm = report["fidelity_multidim"]
    f = report["fidelity"]

    print(f"\n{'='*60}")
    print(f"  RecSim E2E Report (with real category data)")
    print(f"{'='*60}")
    print(f"  Agents:        {report['metadata']['n_agents']}")
    print(f"  Total steps:   {report['metadata']['total_steps']}")
    print(f"  Archetypes:    {report['metadata']['n_archetypes']}")
    print(f"{'─'*60}")
    print(f"  Avg Watch %:   {report['summary']['avg_watch_pct']:.1%}")
    print(f"  Skip Rate:     {report['summary']['skip_rate']:.1%}")
    print(f"  Exit Rate:     {report['summary']['exit_rate']:.1%}")
    print(f"  Like Rate:     {report['summary']['like_rate']:.1%}")
    print(f"{'─'*60}")
    print(f"  F_overall:     {f['F_overall']:.4f}")
    print(f"  F_multidim:    {fm['F_multidim']:.4f}")
    print(f"{'─'*60}")
    print(f"  Per dimension:")
    for k, v in fm["per_dimension"].items():
        print(f"    {k:30s} {v:.4f}")
    print(f"{'─'*60}")
    print(f"  Raw metrics:")
    for k, v in fm["raw_metrics"].items():
        print(f"    {k:30s} {v:.6f}")
    print(f"{'='*60}")
    print(f"  Report: {REPORT_DIR / 'latest_report.json'}")
    print(f"{'='*60}")
