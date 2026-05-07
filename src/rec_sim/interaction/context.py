"""Session context model."""
from dataclasses import dataclass
import numpy as np

TIME_SLOTS = ("morning", "noon", "afternoon", "evening", "night")
TIME_SLOT_WEIGHTS = [0.1, 0.15, 0.15, 0.35, 0.25]
SESSION_TYPES = ("first_visit", "normal", "return_user")


@dataclass
class SessionContext:
    session_type: str
    time_slot: str
    network: str
    step_index: int
    fatigue: float


def sample_session_context(session_type: str = "normal", step_index: int = 0, seed: int = None) -> SessionContext:
    rng = np.random.default_rng(seed)
    time_slot = rng.choice(TIME_SLOTS, p=TIME_SLOT_WEIGHTS)
    network = rng.choice(["2g", "3g", "4g", "wifi"], p=[0.02, 0.08, 0.45, 0.45])
    fatigue = 1.0 - np.exp(-0.05 * step_index)
    return SessionContext(session_type=session_type, time_slot=time_slot,
                          network=network, step_index=step_index, fatigue=fatigue)
