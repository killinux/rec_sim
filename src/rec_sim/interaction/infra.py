"""Infrastructure state model for video delivery simulation."""
from dataclasses import dataclass
import numpy as np

QUALITY_LEVELS = ("360p", "480p", "720p", "1080p")
CODECS = ("h264", "h265", "av1")

NETWORK_PROFILES = {
    "2g":   {"quality_dist": [0.6, 0.3, 0.1, 0.0], "stall_rate": 0.3, "first_frame_base": 3000},
    "3g":   {"quality_dist": [0.2, 0.4, 0.3, 0.1], "stall_rate": 0.15, "first_frame_base": 1500},
    "4g":   {"quality_dist": [0.05, 0.15, 0.5, 0.3], "stall_rate": 0.05, "first_frame_base": 800},
    "wifi": {"quality_dist": [0.0, 0.05, 0.35, 0.6], "stall_rate": 0.02, "first_frame_base": 400},
}

DEVICE_DECODE_PENALTY = {"low": 1.5, "mid": 1.0, "high": 0.8}
BITRATE_MAP = {"360p": 600, "480p": 1200, "720p": 2400, "1080p": 4800}


@dataclass
class InfraState:
    quality: str
    bitrate_kbps: int
    codec: str
    first_frame_ms: int
    stall_count: int
    stall_duration_ms: int


def sample_infra_state(network: str = "4g", device_tier: str = "mid", seed: int = None) -> InfraState:
    rng = np.random.default_rng(seed)
    profile = NETWORK_PROFILES.get(network, NETWORK_PROFILES["4g"])
    quality = rng.choice(QUALITY_LEVELS, p=profile["quality_dist"])
    codec = rng.choice(CODECS, p=[0.4, 0.45, 0.15])
    bitrate = BITRATE_MAP[quality]
    device_mult = DEVICE_DECODE_PENALTY.get(device_tier, 1.0)
    first_frame = int(profile["first_frame_base"] * device_mult * rng.uniform(0.7, 1.3))
    stall_count = rng.poisson(profile["stall_rate"] * 5)
    stall_duration = int(stall_count * rng.uniform(300, 2000)) if stall_count > 0 else 0
    return InfraState(quality=quality, bitrate_kbps=bitrate, codec=codec,
                      first_frame_ms=first_frame, stall_count=stall_count, stall_duration_ms=stall_duration)
