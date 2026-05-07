"""Full pipeline: data → cluster → persona → simulate(L0+L1+L2) → calibrate → extrapolate → evaluate."""
from __future__ import annotations
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rec_sim.baseline.loader import load_kuairec, load_kuairec_items
from rec_sim.baseline.clustering import extract_user_features, cluster_users
from rec_sim.baseline.distribution import extract_archetype_distributions
from rec_sim.baseline.interest import (
    build_category_map, get_all_categories,
    build_user_interest_vectors, build_archetype_interest_vectors,
)
from rec_sim.runner import run_simulation, SimulationConfig, RealDataContext
from rec_sim.report import generate_report
from rec_sim.llm.provider import create_provider
from rec_sim.calibration.loop import calibrate, CalibrationConfig
from rec_sim.extrapolation.scaler import fit_and_scale, generate_traffic_report
from rec_sim.evaluation.abtest import run_abtest, ABTestConfig
import numpy as np

REPORT_DIR = Path("/Users/bytedance/Desktop/hehe/research/rec_sim/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def build_real_data():
    """Build RealDataContext from KuaiRec."""
    print("=" * 60)
    print("  Step 1: Loading KuaiRec data...")
    print("=" * 60)
    t0 = time.time()
    interactions = load_kuairec(use_small=True)
    items = load_kuairec_items()
    print(f"  Loaded {len(interactions)} interactions, {len(items)} items in {time.time()-t0:.1f}s")

    print("\n  Step 2: Clustering users...")
    t0 = time.time()
    features, feature_names = extract_user_features(interactions, items)
    labels, _ = cluster_users(features, n_clusters=20)
    user_ids = interactions.groupby("user_id").first().index.values
    dists = extract_archetype_distributions(interactions, labels, user_ids)
    print(f"  {len(dists)} archetypes from {len(user_ids)} users in {time.time()-t0:.1f}s")

    print("\n  Step 3: Building interest vectors...")
    t0 = time.time()
    cat_map = build_category_map(items)
    all_cats = get_all_categories(cat_map)
    user_vecs = build_user_interest_vectors(interactions, cat_map, all_cats)
    arch_vecs = build_archetype_interest_vectors(user_vecs, user_ids, labels)
    print(f"  {len(all_cats)} categories, {len(arch_vecs)} archetype vectors in {time.time()-t0:.1f}s")

    # Real category distribution
    real_cat_counts = {}
    for _, row in interactions.iterrows():
        item_id = int(row["item_id"])
        cats = cat_map.get(item_id, [])
        for c in cats:
            key = str(c)
            real_cat_counts[key] = real_cat_counts.get(key, 0) + 1
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


def run_pipeline():
    dists, real_data, features = build_real_data()

    # --- Step 4: Simulate with DeepSeek LLM ---
    print("\n" + "=" * 60)
    print("  Step 4: Running simulation with DeepSeek LLM (Layer 2)...")
    print("=" * 60)

    import os
    api_key = os.environ.get("LLM_API_KEY", "")
    if api_key:
        llm_provider = create_provider("deepseek", api_key=api_key)
        print("  Using DeepSeek LLM for Layer 2 decisions")
    else:
        llm_provider = create_provider("mock")
        print("  No LLM_API_KEY found, using MockProvider")

    config = SimulationConfig(n_agents=100, videos_per_session=10, seed=42)
    t0 = time.time()
    result = run_simulation(config, dists, real_data=real_data, llm_provider=llm_provider)
    sim_time = time.time() - t0
    print(f"  {len(result.logs)} decisions in {sim_time:.1f}s")

    # Count Layer 2 decisions
    l2_count = sum(1 for l in result.logs if l.get("decision_layer") == 2)
    print(f"  Layer 2 (LLM) decisions: {l2_count}/{len(result.logs)} ({l2_count/max(len(result.logs),1):.1%})")

    # Generate report
    report = generate_report(
        result, dists, config,
        output_path=REPORT_DIR / "latest_report.json",
        real_data=real_data,
    )
    print(f"\n  F_overall:  {report['fidelity']['F_overall']:.4f}")
    print(f"  F_multidim: {report['fidelity_multidim']['F_multidim']:.4f}")
    for k, v in report["fidelity_multidim"]["per_dimension"].items():
        print(f"    {k:30s} {v:.4f}")

    # --- Step 5: Calibration ---
    print("\n" + "=" * 60)
    print("  Step 5: Running calibration loop...")
    print("=" * 60)
    t0 = time.time()
    cal_config = CalibrationConfig(
        max_outer_iterations=2,
        max_mid_iterations=5,
        f_multidim_target=0.7,
        learning_rate=0.15,
    )
    cal_result = calibrate(
        dists, config, real_data, cal_config,
        report_dir=str(REPORT_DIR),
    )
    cal_time = time.time() - t0
    print(f"  Converged: {cal_result.converged}")
    print(f"  Iterations: {cal_result.iterations_used}")
    print(f"  F: {cal_result.initial_f:.4f} -> {cal_result.final_f:.4f}")
    print(f"  Time: {cal_time:.1f}s")

    # --- Step 6: Extrapolation ---
    print("\n" + "=" * 60)
    print("  Step 6: Extrapolating to 1 billion users...")
    print("=" * 60)
    t0 = time.time()

    # Use calibrated distributions for the final simulation
    final_result = run_simulation(config, cal_result.final_distributions, real_data=real_data, llm_provider=llm_provider)
    scaled = fit_and_scale(final_result, target_population=1_000_000_000, n_representatives=5000, seed=42)
    traffic = generate_traffic_report(scaled)
    ext_time = time.time() - t0
    print(f"  Population: {traffic['population']['target']:,}")
    print(f"  Representatives: {traffic['population']['representatives']}")
    print(f"  GMM components: {traffic['population']['gmm_components']}")
    print(f"\n  Overall metrics:")
    for k, v in traffic["overall"].items():
        print(f"    {k:25s} {v:.4f}")
    print(f"\n  Segments:")
    for seg_name, seg in traffic["segments"].items():
        print(f"    {seg['label']:25s} {seg['user_count']:>15,} users ({seg['proportion']:.1%})")
    print(f"\n  Quality:")
    for k, v in traffic["quality"].items():
        print(f"    {k:35s} {v:.6f}")
    print(f"  Time: {ext_time:.1f}s")

    # Save traffic report
    with open(REPORT_DIR / "traffic_report_1B.json", "w") as f:
        json.dump(traffic, f, indent=2, default=str)

    # --- Step 7: A/B Test ---
    print("\n" + "=" * 60)
    print("  Step 7: Running A/B test (10 vs 30 videos/session)...")
    print("=" * 60)
    t0 = time.time()
    ab_config = ABTestConfig(
        name="session_length_experiment",
        control_label="10_videos",
        treatment_label="30_videos",
        treatment_overrides={"videos_per_session": 30},
    )
    ab_result = run_abtest(ab_config, cal_result.final_distributions, config, real_data=real_data)
    ab_time = time.time() - t0
    print(ab_result.summary)
    print(f"\n  Statistical tests:")
    for test_name, test_data in ab_result.statistical_tests.items():
        if isinstance(test_data, dict):
            print(f"    {test_name}: p={test_data.get('p_value', 'N/A'):.4f} sig={test_data.get('significant', 'N/A')}")
        else:
            print(f"    {test_name}: {test_data:.4f}")
    print(f"  Time: {ab_time:.1f}s")

    # Save A/B test result
    ab_report = {
        "name": ab_result.name,
        "control": ab_result.control_metrics,
        "treatment": ab_result.treatment_metrics,
        "deltas": ab_result.deltas,
        "statistical_tests": {k: v for k, v in ab_result.statistical_tests.items()},
        "winner": ab_result.winner,
    }
    with open(REPORT_DIR / "abtest_result.json", "w") as f:
        json.dump(ab_report, f, indent=2, default=str)

    # --- Final Summary ---
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Reports saved to: {REPORT_DIR}")
    print(f"  - latest_report.json (simulation + fidelity)")
    print(f"  - traffic_report_1B.json (1B extrapolation)")
    print(f"  - abtest_result.json (A/B test)")
    print(f"  Dashboard: http://localhost/research/rec_sim/reports/dashboard.html")
    print("=" * 60)
    print("PIPELINE_DONE")


if __name__ == "__main__":
    run_pipeline()
