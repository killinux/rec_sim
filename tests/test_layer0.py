import pytest
import numpy as np
from rec_sim.interaction.layer0 import experience_decision, ExperienceResult
from rec_sim.interaction.infra import InfraState
from rec_sim.persona.skeleton import PersonaSkeleton


def _make_skeleton(**overrides):
    defaults = dict(agent_id=0, archetype_id=0, watch_ratio_baseline=0.6,
                    duration_baseline=10000, stall_tolerance=1500,
                    quality_sensitivity=0.5, fatigue_rate=0.05, first_session_patience=5)
    defaults.update(overrides)
    return PersonaSkeleton(**defaults)


def _make_infra(**overrides):
    defaults = dict(quality="720p", bitrate_kbps=2400, codec="h265",
                    first_frame_ms=500, stall_count=0, stall_duration_ms=0)
    defaults.update(overrides)
    return InfraState(**defaults)


def test_no_stall_no_penalty():
    result = experience_decision(_make_skeleton(), _make_infra(), is_first_visit=False, seed=42)
    assert isinstance(result, ExperienceResult)
    assert result.watch_pct_factor >= 0.9
    assert not result.force_skip


def test_heavy_stall_causes_skip():
    skeleton = _make_skeleton(stall_tolerance=500)
    infra = _make_infra(stall_count=5, stall_duration_ms=8000)
    result = experience_decision(skeleton, infra, is_first_visit=False, seed=42)
    assert result.watch_pct_factor < 0.5 or result.force_skip


def test_first_visit_lower_tolerance():
    skeleton = _make_skeleton(stall_tolerance=1500)
    infra = _make_infra(first_frame_ms=3000, stall_count=1, stall_duration_ms=2000)
    result_first = experience_decision(skeleton, infra, is_first_visit=True, seed=42)
    result_normal = experience_decision(skeleton, infra, is_first_visit=False, seed=42)
    assert result_first.watch_pct_factor <= result_normal.watch_pct_factor


def test_low_quality_reduces_factor():
    skeleton = _make_skeleton(quality_sensitivity=0.8)
    infra_good = _make_infra(quality="1080p")
    infra_bad = _make_infra(quality="360p")
    r_good = experience_decision(skeleton, infra_good, is_first_visit=False, seed=42)
    r_bad = experience_decision(skeleton, infra_bad, is_first_visit=False, seed=42)
    assert r_bad.watch_pct_factor < r_good.watch_pct_factor
