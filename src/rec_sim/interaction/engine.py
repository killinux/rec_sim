"""Decision engine: orchestrates Layer 0 + Layer 1 per video."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from rec_sim.persona.skeleton import PersonaSkeleton
from rec_sim.interaction.infra import InfraState
from rec_sim.interaction.context import SessionContext
from rec_sim.interaction.layer0 import experience_decision
from rec_sim.interaction.layer1 import content_decision
from rec_sim.llm.provider import LLMProvider


@dataclass
class VideoItem:
    video_id: str
    category: str
    duration_ms: int
    interest_match: float


@dataclass
class StepResult:
    action: str
    watch_pct: float
    liked: bool
    commented: bool
    shared: bool
    decision_layer: int
    fidelity_tag: str
    l0_factor: float
    l1_base_pct: float
    agent_id: int

    def to_log(self, session_id: str) -> dict:
        return {
            "session_id": session_id,
            "agent_id": self.agent_id,
            "action": self.action,
            "watch_pct": self.watch_pct,
            "liked": self.liked,
            "commented": self.commented,
            "shared": self.shared,
            "decision_layer": self.decision_layer,
            "fidelity_tag": self.fidelity_tag,
            "l0_factor": self.l0_factor,
            "l1_base_pct": self.l1_base_pct,
            "infra_state": {},
        }


class DecisionEngine:
    def __init__(self, seed: int = 42, llm_provider: LLMProvider | None = None,
                 l2_sample_rate: float = 0.1):
        self._seed_base = seed
        self._call_count = 0
        self._llm_provider = llm_provider
        self._l2_sample_rate = l2_sample_rate

    def step(self, skeleton: PersonaSkeleton, video: VideoItem,
             infra: InfraState, ctx: SessionContext) -> StepResult:
        seed = self._seed_base + self._call_count
        self._call_count += 1

        is_first = ctx.session_type == "first_visit"
        l0 = experience_decision(skeleton, infra, is_first_visit=is_first, seed=seed)

        if l0.force_exit:
            return StepResult(action="exit_app", watch_pct=0.0, liked=False, commented=False,
                              shared=False, decision_layer=0, fidelity_tag="rule",
                              l0_factor=l0.watch_pct_factor, l1_base_pct=0.0, agent_id=skeleton.agent_id)

        if l0.force_skip:
            return StepResult(action="skip", watch_pct=0.0, liked=False, commented=False,
                              shared=False, decision_layer=0, fidelity_tag="rule",
                              l0_factor=l0.watch_pct_factor, l1_base_pct=0.0, agent_id=skeleton.agent_id)

        trigger = None
        if self._llm_provider is not None:
            from rec_sim.interaction.layer2 import should_trigger_layer2, layer2_decision
            trigger = should_trigger_layer2(
                skeleton, video, ctx, l0.watch_pct_factor,
                sample_rate=self._l2_sample_rate, seed=seed + 2000,
            )

        if trigger and self._llm_provider:
            from rec_sim.interaction.layer2 import layer2_decision
            l2 = layer2_decision(self._llm_provider, skeleton, video, infra, ctx, trigger)
            return StepResult(
                action="watch" if l2.watch_pct > 0.05 else "skip",
                watch_pct=l2.watch_pct, liked=l2.liked,
                commented=l2.commented, shared=l2.shared,
                decision_layer=2, fidelity_tag="llm",
                l0_factor=l0.watch_pct_factor,
                l1_base_pct=l2.watch_pct / max(l0.watch_pct_factor, 0.01),
                agent_id=skeleton.agent_id,
            )

        l1 = content_decision(skeleton, interest_match=video.interest_match,
                              l0_factor=l0.watch_pct_factor, fatigue=ctx.fatigue, seed=seed + 1000)
        action = "watch" if l1.watch_pct > 0.05 else "skip"

        return StepResult(action=action, watch_pct=l1.watch_pct,
                          liked=l1.liked, commented=l1.commented, shared=l1.shared,
                          decision_layer=1, fidelity_tag="parametric",
                          l0_factor=l0.watch_pct_factor,
                          l1_base_pct=l1.watch_pct / max(l0.watch_pct_factor, 0.01),
                          agent_id=skeleton.agent_id)
