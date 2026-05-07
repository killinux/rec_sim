import pytest
import numpy as np
from rec_sim.interaction.layer1 import content_decision, ContentResult
from rec_sim.persona.skeleton import PersonaSkeleton


def _make_skeleton(**overrides):
    defaults = dict(agent_id=0, archetype_id=0, watch_ratio_baseline=0.6,
                    duration_baseline=10000, stall_tolerance=1500,
                    quality_sensitivity=0.5, fatigue_rate=0.05, first_session_patience=5)
    defaults.update(overrides)
    return PersonaSkeleton(**defaults)


def test_content_decision_returns_result():
    result = content_decision(skeleton=_make_skeleton(), interest_match=0.8,
                              l0_factor=1.0, fatigue=0.0, seed=42)
    assert isinstance(result, ContentResult)
    assert 0 <= result.watch_pct <= 1
    assert isinstance(result.liked, bool)


def test_high_interest_higher_completion():
    sk = _make_skeleton(watch_ratio_baseline=0.5)
    r_high = content_decision(sk, interest_match=0.9, l0_factor=1.0, fatigue=0.0, seed=42)
    r_low = content_decision(sk, interest_match=0.1, l0_factor=1.0, fatigue=0.0, seed=42)
    assert r_high.watch_pct > r_low.watch_pct


def test_l0_factor_reduces_completion():
    sk = _make_skeleton(watch_ratio_baseline=0.7)
    r_full = content_decision(sk, interest_match=0.8, l0_factor=1.0, fatigue=0.0, seed=42)
    r_degraded = content_decision(sk, interest_match=0.8, l0_factor=0.5, fatigue=0.0, seed=42)
    assert r_degraded.watch_pct < r_full.watch_pct


def test_fatigue_reduces_completion():
    sk = _make_skeleton()
    r_fresh = content_decision(sk, interest_match=0.7, l0_factor=1.0, fatigue=0.0, seed=42)
    r_tired = content_decision(sk, interest_match=0.7, l0_factor=1.0, fatigue=0.8, seed=42)
    assert r_tired.watch_pct < r_fresh.watch_pct
