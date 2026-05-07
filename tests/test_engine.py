import pytest
from rec_sim.interaction.engine import DecisionEngine, VideoItem, StepResult
from rec_sim.interaction.infra import InfraState
from rec_sim.interaction.context import SessionContext
from rec_sim.persona.skeleton import PersonaSkeleton


def _make_skeleton(**overrides):
    defaults = dict(agent_id=0, archetype_id=0, watch_ratio_baseline=0.6,
                    duration_baseline=10000, stall_tolerance=1500,
                    quality_sensitivity=0.5, fatigue_rate=0.05, first_session_patience=5)
    defaults.update(overrides)
    return PersonaSkeleton(**defaults)


def test_engine_step_returns_result():
    engine = DecisionEngine(seed=42)
    skeleton = _make_skeleton()
    video = VideoItem(video_id="v1", category="food", duration_ms=15000, interest_match=0.7)
    infra = InfraState("720p", 2400, "h265", 500, 0, 0)
    ctx = SessionContext("normal", "evening", "wifi", 0, 0.0)
    result = engine.step(skeleton, video, infra, ctx)
    assert isinstance(result, StepResult)
    assert result.action in ("watch", "skip", "exit_app")
    assert 0 <= result.watch_pct <= 1
    assert result.decision_layer in (0, 1)
    assert result.fidelity_tag in ("rule", "parametric")


def test_engine_exit_on_heavy_stall_first_visit():
    engine = DecisionEngine(seed=0)
    skeleton = _make_skeleton(stall_tolerance=500)
    video = VideoItem("v1", "tech", 10000, 0.5)
    infra = InfraState("360p", 600, "h264", 5000, 8, 15000)
    ctx = SessionContext("first_visit", "morning", "3g", 0, 0.0)
    results = [engine.step(skeleton, video, infra, ctx) for _ in range(20)]
    exit_count = sum(1 for r in results if r.action == "exit_app")
    assert exit_count > 0


def test_engine_step_result_log_format():
    engine = DecisionEngine(seed=42)
    skeleton = _make_skeleton()
    video = VideoItem("v1", "food", 15000, 0.8)
    infra = InfraState("1080p", 4800, "h265", 300, 0, 0)
    ctx = SessionContext("normal", "evening", "wifi", 0, 0.0)
    result = engine.step(skeleton, video, infra, ctx)
    log = result.to_log(session_id="s1")
    assert log["session_id"] == "s1"
    assert log["agent_id"] == 0
    assert "infra_state" in log
    assert "fidelity_tag" in log
