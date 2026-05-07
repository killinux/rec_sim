"""Layer 2: LLM-driven emergent decisions for complex scenarios."""
from __future__ import annotations
from dataclasses import dataclass
from rec_sim.llm.provider import LLMProvider, LLMResponse
from rec_sim.persona.skeleton import PersonaSkeleton
from rec_sim.interaction.infra import InfraState
from rec_sim.interaction.context import SessionContext
from rec_sim.interaction.engine import VideoItem


@dataclass
class Layer2Result:
    watch_pct: float
    liked: bool
    commented: bool
    shared: bool
    reason: str
    triggered_by: str


def should_trigger_layer2(
    skeleton: PersonaSkeleton,
    video: VideoItem,
    ctx: SessionContext,
    l0_factor: float,
    sample_rate: float = 0.1,
    seed: int = 0,
) -> str | None:
    """Determine if Layer 2 should be triggered. Returns trigger reason or None."""
    import numpy as np
    rng = np.random.default_rng(seed)

    # First visit, first N videos — critical retention path
    if ctx.session_type == "first_visit" and ctx.step_index < skeleton.first_session_patience:
        return "first_visit_critical"

    # Severe infrastructure degradation during content consumption
    if l0_factor < 0.5:
        return "severe_infra_degradation"

    # High interest but bad experience — conflict scenario
    if video.interest_match > 0.7 and l0_factor < 0.7:
        return "interest_infra_conflict"

    # Random sampling for calibration audit
    if rng.random() < sample_rate:
        return "random_audit"

    return None


def build_layer2_prompt(
    skeleton: PersonaSkeleton,
    video: VideoItem,
    infra: InfraState,
    ctx: SessionContext,
    trigger_reason: str,
) -> str:
    """Build the prompt for Layer 2 LLM decision."""
    session_desc = {
        "first_visit": "This is your FIRST TIME using this app.",
        "normal": "You are a regular user.",
        "return_user": "You left this app before and just came back to try again.",
    }.get(ctx.session_type, "You are a regular user.")

    quality_desc = {
        "360p": "very blurry, hard to see details",
        "480p": "somewhat blurry",
        "720p": "decent quality",
        "1080p": "clear, high quality",
    }.get(infra.quality, "average quality")

    stall_desc = ""
    if infra.stall_count > 0:
        stall_desc = f"The video stuttered {infra.stall_count} time(s), total freeze of {infra.stall_duration_ms}ms."

    fatigue_desc = ""
    if ctx.fatigue > 0.3:
        fatigue_desc = f"You've been watching for a while and are getting tired (fatigue level: {ctx.fatigue:.0%})."

    prompt = f"""You are simulating a short-video app user making a viewing decision.

USER PROFILE:
- Typical completion rate: {skeleton.watch_ratio_baseline:.0%} of videos
- Stall tolerance: {"low" if skeleton.stall_tolerance < 1000 else "medium" if skeleton.stall_tolerance < 2500 else "high"}
- Quality sensitivity: {"high" if skeleton.quality_sensitivity > 0.6 else "medium" if skeleton.quality_sensitivity > 0.3 else "low"}

CURRENT SITUATION:
- {session_desc}
- This is video #{ctx.step_index + 1} in this session.
- Time: {ctx.time_slot}
- Network: {ctx.network}
{fatigue_desc}

VIDEO:
- Category: {video.category}
- Interest match: {video.interest_match:.0%}
- Visual quality: {infra.quality} ({quality_desc})
- First frame loaded in {infra.first_frame_ms}ms
{stall_desc}

DECISION CONTEXT: {trigger_reason}

Based on this user's profile and the current situation, respond with ONLY this JSON:
{{"watch_pct": <0.0 to 1.0>, "liked": <true/false>, "commented": <true/false>, "shared": <true/false>, "reason": "<brief explanation>"}}"""

    return prompt


def layer2_decision(
    provider: LLMProvider,
    skeleton: PersonaSkeleton,
    video: VideoItem,
    infra: InfraState,
    ctx: SessionContext,
    trigger_reason: str,
) -> Layer2Result:
    """Execute a Layer 2 LLM decision."""
    prompt = build_layer2_prompt(skeleton, video, infra, ctx, trigger_reason)
    response = provider.decide(prompt)

    return Layer2Result(
        watch_pct=max(0.0, min(1.0, response.watch_pct)),
        liked=response.liked,
        commented=response.commented,
        shared=response.shared,
        reason=response.reason,
        triggered_by=trigger_reason,
    )
