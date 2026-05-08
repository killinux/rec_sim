"""Layer 1: Content decision — interest matching and engagement."""
from dataclasses import dataclass
import numpy as np
from rec_sim.persona.skeleton import PersonaSkeleton


@dataclass
class ContentResult:
    watch_pct: float
    liked: bool
    commented: bool
    shared: bool


def content_decision(skeleton: PersonaSkeleton, interest_match: float,
                     l0_factor: float = 1.0, fatigue: float = 0.0, seed: int = None) -> ContentResult:
    rng = np.random.default_rng(seed)
    base = skeleton.watch_ratio_baseline

    # Snap decision: high-interest → hooked, low-interest → instant skip
    # Creates bimodal distribution peaks matching real user behavior
    snap_roll = rng.random()
    if interest_match > 0.65 and snap_roll < 0.3:
        raw_pct = rng.uniform(0.85, 1.0)
    elif interest_match < 0.25 and snap_roll < 0.2:
        raw_pct = rng.uniform(0.0, 0.12)
    else:
        interest_boost = (interest_match - 0.5) * 0.7
        fatigue_penalty = fatigue * 0.3
        noise = rng.normal(0, 0.12)
        raw_pct = base + interest_boost - fatigue_penalty + noise

    watch_pct = float(np.clip(raw_pct * l0_factor, 0.0, 1.0))
    engagement_base = watch_pct * interest_match
    liked = bool(rng.random() < engagement_base * 0.25)
    commented = bool(rng.random() < engagement_base * 0.03)
    shared = bool(rng.random() < engagement_base * 0.02)
    return ContentResult(watch_pct=watch_pct, liked=liked, commented=commented, shared=shared)
