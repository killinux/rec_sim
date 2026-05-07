"""Simulation runner: N agents x M videos per session."""
from dataclasses import dataclass, field
import numpy as np
from rec_sim.persona.skeleton import generate_skeletons
from rec_sim.baseline.distribution import ArchetypeDistribution
from rec_sim.interaction.engine import DecisionEngine, VideoItem
from rec_sim.interaction.infra import sample_infra_state
from rec_sim.interaction.context import sample_session_context
from rec_sim.fidelity.metrics import relative_error


@dataclass
class SimulationConfig:
    n_agents: int = 100
    videos_per_session: int = 30
    seed: int = 42


@dataclass
class SimulationResult:
    logs: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def run_simulation(config: SimulationConfig,
                   distributions: list[ArchetypeDistribution]) -> SimulationResult:
    rng = np.random.default_rng(config.seed)
    skeletons = generate_skeletons(distributions, config.n_agents, config.seed)
    engine = DecisionEngine(seed=config.seed)

    logs = []
    categories = ["food", "travel", "tech", "beauty", "comedy", "sports", "music", "education"]

    for skeleton in skeletons:
        session_id = f"s_{skeleton.agent_id}"
        session_type = rng.choice(["first_visit", "normal", "return_user"], p=[0.1, 0.8, 0.1])

        for step in range(config.videos_per_session):
            cat = rng.choice(categories)
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
