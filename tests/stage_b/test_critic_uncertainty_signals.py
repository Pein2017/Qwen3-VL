from src.stage_b.io.export import serialize_trajectory
from src.stage_b.types import (
    CriticOutput,
    DecodeConfig,
    DeterministicSignals,
    ParsedTrajectory,
    Trajectory,
    TrajectoryWithSignals,
)
from datetime import datetime


def test_export_includes_uncertainty_fields():
    base = Trajectory(
        group_id="g1",
        mission="M",
        candidate_index=0,
        decode=DecodeConfig(temperature=0.7, top_p=0.9, max_new_tokens=32),
        response_text="...",
        created_at=datetime.utcnow(),
    )
    parsed = ParsedTrajectory(base=base, verdict="pass", reason="ok", confidence=None, format_ok=True)
    signals = DeterministicSignals(label_match=True, self_consistency=None, confidence=None)
    critic = CriticOutput(
        summary="s",
        critique="c",
        needs_recheck=True,
        uncertainty_reason="角度不足",
        evidence_quality_level="中",
        evidence_sufficiency=False,
        label_consistency="不确定",
        suspected_label_noise=False,
        recommended_action="人工复核",
    )
    item = TrajectoryWithSignals(parsed=parsed, signals=signals, critic=critic)
    payload = serialize_trajectory(item, reflection_cycle=0, guidance_step=1)
    critic_payload = payload["result"]["critic"]
    assert critic_payload["needs_recheck"] is True
    assert critic_payload["evidence_sufficiency"] is False

