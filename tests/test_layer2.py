"""Tests for Layer 2 LLM decisions."""
from __future__ import annotations
import pytest
from rec_sim.llm.provider import MockProvider, create_provider, LLMResponse
from rec_sim.interaction.layer2 import (
    should_trigger_layer2, build_layer2_prompt, layer2_decision, Layer2Result,
)
from rec_sim.interaction.infra import InfraState
from rec_sim.interaction.context import SessionContext
from rec_sim.interaction.engine import VideoItem
from rec_sim.persona.skeleton import PersonaSkeleton


def _make_skeleton(**overrides):
    defaults = dict(agent_id=0, archetype_id=0, watch_ratio_baseline=0.6,
                    duration_baseline=10000, stall_tolerance=1500,
                    quality_sensitivity=0.5, fatigue_rate=0.05, first_session_patience=5)
    defaults.update(overrides)
    return PersonaSkeleton(**defaults)


def _make_video(**overrides):
    defaults = dict(video_id="v1", category="food", duration_ms=15000, interest_match=0.7)
    defaults.update(overrides)
    return VideoItem(**defaults)


def _make_infra(**overrides):
    defaults = dict(quality="720p", bitrate_kbps=2400, codec="h265",
                    first_frame_ms=500, stall_count=0, stall_duration_ms=0)
    defaults.update(overrides)
    return InfraState(**defaults)


def _make_ctx(**overrides):
    defaults = dict(session_type="normal", time_slot="evening", network="wifi",
                    step_index=5, fatigue=0.2)
    defaults.update(overrides)
    return SessionContext(**defaults)


# --- Provider tests ---

def test_mock_provider_returns_response():
    provider = MockProvider()
    response = provider.decide("test prompt")
    assert isinstance(response, LLMResponse)
    assert 0 <= response.watch_pct <= 1


def test_mock_provider_context_sensitive():
    provider = MockProvider()
    r_normal = provider.decide("normal user watching 1080p video")
    r_bad = provider.decide("first_visit user, 360p quality, stall_count=3")
    assert r_bad.watch_pct < r_normal.watch_pct


def test_create_provider_mock():
    p = create_provider("mock")
    assert isinstance(p, MockProvider)


def test_create_provider_unknown_raises():
    with pytest.raises(ValueError):
        create_provider("nonexistent")


# --- Trigger tests ---

def test_trigger_first_visit():
    skeleton = _make_skeleton(first_session_patience=5)
    video = _make_video()
    ctx = _make_ctx(session_type="first_visit", step_index=2)
    reason = should_trigger_layer2(skeleton, video, ctx, l0_factor=0.9)
    assert reason == "first_visit_critical"


def test_trigger_severe_infra():
    skeleton = _make_skeleton()
    video = _make_video()
    ctx = _make_ctx(session_type="normal", step_index=10)
    reason = should_trigger_layer2(skeleton, video, ctx, l0_factor=0.3)
    assert reason == "severe_infra_degradation"


def test_trigger_conflict():
    skeleton = _make_skeleton()
    video = _make_video(interest_match=0.85)
    ctx = _make_ctx(session_type="normal", step_index=10)
    reason = should_trigger_layer2(skeleton, video, ctx, l0_factor=0.6)
    assert reason == "interest_infra_conflict"


def test_no_trigger_normal():
    skeleton = _make_skeleton()
    video = _make_video(interest_match=0.5)
    ctx = _make_ctx(session_type="normal", step_index=10)
    reason = should_trigger_layer2(skeleton, video, ctx, l0_factor=0.9, sample_rate=0.0)
    assert reason is None


# --- Prompt tests ---

def test_prompt_contains_key_info():
    skeleton = _make_skeleton()
    video = _make_video(category="food", interest_match=0.8)
    infra = _make_infra(quality="480p", stall_count=2, stall_duration_ms=3000)
    ctx = _make_ctx(session_type="first_visit", step_index=1)
    prompt = build_layer2_prompt(skeleton, video, infra, ctx, "first_visit_critical")
    assert "food" in prompt
    assert "480p" in prompt
    assert "stuttered" in prompt or "stall" in prompt.lower()
    assert "FIRST TIME" in prompt
    assert "watch_pct" in prompt


# --- Decision tests ---

def test_layer2_decision_with_mock():
    provider = MockProvider()
    skeleton = _make_skeleton()
    video = _make_video()
    infra = _make_infra()
    ctx = _make_ctx()
    result = layer2_decision(provider, skeleton, video, infra, ctx, "random_audit")
    assert isinstance(result, Layer2Result)
    assert 0 <= result.watch_pct <= 1
    assert result.triggered_by == "random_audit"


def test_layer2_first_visit_lower_engagement():
    provider = MockProvider()
    skeleton = _make_skeleton()
    video = _make_video()
    infra = _make_infra(quality="360p", stall_count=2, stall_duration_ms=4000)
    ctx_first = _make_ctx(session_type="first_visit", step_index=0)
    ctx_normal = _make_ctx(session_type="normal", step_index=10)
    r_first = layer2_decision(provider, skeleton, video, infra, ctx_first, "first_visit_critical")
    r_normal = layer2_decision(provider, skeleton, video, infra, ctx_normal, "random_audit")
    assert r_first.watch_pct <= r_normal.watch_pct
