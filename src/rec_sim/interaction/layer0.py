"""Layer 0: Experience decision — infrastructure quality impact on user behavior."""
from dataclasses import dataclass
import numpy as np
from rec_sim.interaction.infra import InfraState
from rec_sim.persona.skeleton import PersonaSkeleton

QUALITY_SCORE = {"360p": 0.3, "480p": 0.55, "720p": 0.8, "1080p": 1.0}


@dataclass
class ExperienceResult:
    watch_pct_factor: float
    force_skip: bool
    force_exit: bool
    detail: dict


def experience_decision(skeleton: PersonaSkeleton, infra: InfraState,
                        is_first_visit: bool = False, seed: int = None) -> ExperienceResult:
    rng = np.random.default_rng(seed)
    tolerance_mult = 0.5 if is_first_visit else 1.0
    effective_tolerance = skeleton.stall_tolerance * tolerance_mult

    stall_penalty = 0.0
    if infra.stall_duration_ms > 0:
        ratio = infra.stall_duration_ms / max(effective_tolerance, 1)
        stall_penalty = 1.0 - np.exp(-0.7 * ratio)

    quality_score = QUALITY_SCORE.get(infra.quality, 0.5)
    quality_penalty = skeleton.quality_sensitivity * (1.0 - quality_score) * 0.4

    first_frame_penalty = 0.0
    threshold = 1000 if is_first_visit else 2000
    if infra.first_frame_ms > threshold:
        over = (infra.first_frame_ms - threshold) / threshold
        first_frame_penalty = min(over * 0.3, 0.5)

    total_penalty = min(stall_penalty + quality_penalty + first_frame_penalty, 1.0)
    noise = rng.normal(0, 0.03)
    watch_pct_factor = float(np.clip(1.0 - total_penalty + noise, 0.05, 1.0))

    force_skip = rng.random() < (stall_penalty * 0.8 + first_frame_penalty * 0.5)
    force_exit = (is_first_visit and total_penalty > 0.7 and rng.random() < 0.5)

    return ExperienceResult(watch_pct_factor=watch_pct_factor, force_skip=force_skip,
                            force_exit=force_exit,
                            detail={"stall_penalty": stall_penalty, "quality_penalty": quality_penalty,
                                    "first_frame_penalty": first_frame_penalty})
