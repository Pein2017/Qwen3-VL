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


def make_item(include_critic: bool) -> TrajectoryWithSignals:
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
    critic = None
    if include_critic:
        critic = CriticOutput(summary="s", critique="c")
    return TrajectoryWithSignals(parsed=parsed, signals=signals, summary=None, critique=None, critic=critic)


def test_serialize_trajectory_no_candidate_ops():
    item = make_item(include_critic=True)
    payload = serialize_trajectory(item, reflection_cycle=0, guidance_step=1)
    critic = payload["result"]["critic"]
    assert critic is not None
    # Ensure candidate_ops is not present in exports
    assert "candidate_ops" not in critic

