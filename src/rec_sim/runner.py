"""Simulation runner: N agents x M videos per session."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from rec_sim.persona.skeleton import generate_skeletons
from rec_sim.baseline.distribution import ArchetypeDistribution
from rec_sim.interaction.engine import DecisionEngine, VideoItem
from rec_sim.interaction.infra import sample_infra_state
from rec_sim.interaction.context import sample_session_context
from rec_sim.fidelity.metrics import relative_error
from rec_sim.llm.provider import LLMProvider


@dataclass
class SimulationConfig:
    n_agents: int = 100
    videos_per_session: int = 30
    seed: int = 42


@dataclass
class RealDataContext:
    """Real data for realistic category sampling and interest matching."""
    category_distribution: dict[str, float] = field(default_factory=dict)
    archetype_interest_vectors: dict[int, np.ndarray] = field(default_factory=dict)
    item_ids: list[int] = field(default_factory=list)
    item_category_map: dict[int, list[int]] = field(default_factory=dict)
    all_categories: list[int] = field(default_factory=list)
    real_cat_counts: dict[str, int] = field(default_factory=dict)
    real_wr_by_category: dict[str, list[float]] = field(default_factory=dict)
    real_interactions_per_user: np.ndarray = field(default_factory=lambda: np.array([]))
    real_user_features: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class SimulationResult:
    logs: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def run_simulation(
    config: SimulationConfig,
    distributions: list[ArchetypeDistribution],
    real_data: RealDataContext | None = None,
    llm_provider: LLMProvider | None = None,
) -> SimulationResult:
    rng = np.random.default_rng(config.seed)
    skeletons = generate_skeletons(distributions, config.n_agents, config.seed)
    engine = DecisionEngine(seed=config.seed, llm_provider=llm_provider)

    # Category sampling setup
    if real_data and real_data.category_distribution:
        cat_names = list(real_data.category_distribution.keys())
        cat_probs = np.array(list(real_data.category_distribution.values()))
        cat_probs = cat_probs / cat_probs.sum()
        use_real_cats = True
    else:
        cat_names = ["food", "travel", "tech", "beauty", "comedy", "sports", "music", "education"]
        cat_probs = np.ones(len(cat_names)) / len(cat_names)
        use_real_cats = False

    # Interest vector setup
    use_real_interest = (
        real_data is not None
        and len(real_data.archetype_interest_vectors) > 0
        and len(real_data.item_ids) > 0
    )

    logs = []

    for skeleton in skeletons:
        session_id = f"s_{skeleton.agent_id}"
        session_type = rng.choice(["first_visit", "normal", "return_user"], p=[0.1, 0.8, 0.1])

        # Get this agent's interest vector
        agent_interest_vec = None
        if use_real_interest:
            agent_interest_vec = real_data.archetype_interest_vectors.get(skeleton.archetype_id)

        for step in range(config.videos_per_session):
            cat = str(rng.choice(cat_names, p=cat_probs))

            # Compute interest match
            if use_real_interest and agent_interest_vec is not None:
                # Pick a random real item and compute cosine similarity
                item_id = int(rng.choice(real_data.item_ids))
                from rec_sim.baseline.interest import build_item_vector, compute_interest_match
                item_vec = build_item_vector(item_id, real_data.item_category_map, real_data.all_categories)
                interest = compute_interest_match(agent_interest_vec, item_vec)
                # Use the item's actual category instead of the sampled one
                item_cats = real_data.item_category_map.get(item_id, [])
                if item_cats:
                    cat = str(item_cats[0])
            else:
                interest = float(rng.beta(2, 2))

            video = VideoItem(
                video_id=f"v_{rng.integers(0, 100000)}",
                category=cat,
                duration_ms=int(rng.lognormal(9.5, 0.6)),
                interest_match=interest,
            )
            ctx = sample_session_context(session_type=session_type, step_index=step,
                                         seed=config.seed + step)
            infra = sample_infra_state(network=ctx.network,
                                       seed=config.seed + skeleton.agent_id + step)
            result = engine.step(skeleton, video, infra, ctx)
            log = result.to_log(session_id=session_id)
            log["step_index"] = step
            log["video_id"] = video.video_id
            log["category"] = cat
            log["context"] = {"session_type": session_type, "time_slot": ctx.time_slot,
                              "network": ctx.network, "fatigue": ctx.fatigue}
            logs.append(log)
            if result.action == "exit_app":
                break

    watch_pcts = [l["watch_pct"] for l in logs if l["action"] == "watch"]
    avg_wr = float(np.mean(watch_pcts)) if watch_pcts else 0.0
    target_wr = float(np.mean([d.watch_ratio_mean for d in distributions]))

    summary = {
        "n_agents": config.n_agents,
        "total_steps": len(logs),
        "avg_watch_pct": avg_wr,
        "exit_rate": sum(1 for l in logs if l["action"] == "exit_app") / max(len(logs), 1),
        "skip_rate": sum(1 for l in logs if l["action"] == "skip") / max(len(logs), 1),
        "fidelity": {
            "F_overall": 1.0 - relative_error(target_wr, avg_wr),
            "target_watch_ratio": target_wr,
            "actual_watch_ratio": avg_wr,
        },
    }
    return SimulationResult(logs=logs, summary=summary)
