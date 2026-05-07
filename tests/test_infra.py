import pytest
from rec_sim.interaction.infra import InfraState, sample_infra_state
from rec_sim.interaction.context import SessionContext, sample_session_context


def test_infra_state_fields():
    state = InfraState(quality="720p", bitrate_kbps=2400, codec="h265",
                       first_frame_ms=800, stall_count=0, stall_duration_ms=0)
    assert state.quality == "720p"
    assert state.bitrate_kbps == 2400


def test_sample_infra_state():
    state = sample_infra_state(network="4g", device_tier="mid", seed=42)
    assert state.quality in ("360p", "480p", "720p", "1080p")
    assert state.first_frame_ms > 0
    assert state.stall_count >= 0


def test_session_context_fields():
    ctx = SessionContext(session_type="first_visit", time_slot="evening",
                         network="wifi", step_index=0, fatigue=0.0)
    assert ctx.session_type == "first_visit"
    assert ctx.fatigue == 0.0


def test_sample_session_context():
    ctx = sample_session_context(session_type="normal", step_index=5, seed=42)
    assert ctx.time_slot in ("morning", "noon", "afternoon", "evening", "night")
    assert ctx.fatigue >= 0
