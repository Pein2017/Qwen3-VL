"""Tests for SampleSummarizer integration in selection and exports."""

from datetime import datetime, timezone

from src.stage_b.config import SelectionConfig
from src.stage_b.io.export import serialize_selection, serialize_trajectory
from src.stage_b.reflection.summarizer import SampleSummarizer
from src.stage_b.scoring.selection import select_for_group
from src.stage_b.types import (
    DecodeConfig,
    DeterministicSignals,
    GroupTicket,
    ParsedTrajectory,
    StageASummaries,
    Trajectory,
    TrajectoryWithSignals,
)


def _build_candidate(*, verdict: str, reason: str, label_match: bool) -> TrajectoryWithSignals:
    decode_cfg = DecodeConfig(temperature=0.6, top_p=0.9, max_new_tokens=128)
    base = Trajectory(
        group_id="QC-SELECT-001",
        mission="挡风板安装检查",
        candidate_index=0,
        decode=decode_cfg,
        response_text=f"Verdict: {verdict}\nReason: {reason}",
        created_at=datetime.now(timezone.utc),
    )

    parsed = ParsedTrajectory(
        base=base,
        verdict=verdict,
        reason=reason,
        confidence=0.82,
        format_ok=True,
    )

    signals = DeterministicSignals(
        label_match=label_match,
        self_consistency=0.5,
        candidate_agreement=label_match,
        confidence=0.82,
        label_trust=0.8 if label_match else 0.5,
        semantic_advantage=1.25,
    )

    return TrajectoryWithSignals(parsed=parsed, signals=signals)


def test_select_for_group_produces_summary_and_critique():
    ticket = GroupTicket(
        group_id="QC-SELECT-001",
        mission="挡风板安装检查",
        label="pass",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "BBU存在"}),
    )

    candidate = _build_candidate(
        verdict="fail",
        reason="未检测到BBU设备且挡风板缺失",
        label_match=False,
    )

    config = SelectionConfig(policy="top_label", tie_break="confidence")
    summarizer = SampleSummarizer()

    selection = select_for_group(
        ticket,
        [candidate],
        guidance_step=1,
        reflection_cycle=0,
        reflection_change=None,
        config=config,
        summarizer=summarizer,
    )

    assert selection.summary is not None
    assert selection.critique is not None
    assert "候选 0" in selection.summary
    assert "候选 0" in selection.critique

    serialized = serialize_selection(selection)
    assert serialized["result"]["summary"] == selection.summary
    assert serialized["result"]["critique"] == selection.critique


def test_serialize_trajectory_includes_summary_and_critique():
    candidate = _build_candidate(
        verdict="fail",
        reason="检测到BBU设备且挡风板安装正确",
        label_match=True,
    )

    enriched_candidate = TrajectoryWithSignals(
        parsed=candidate.parsed,
        signals=candidate.signals,
        summary="候选 0 判定为 fail，与标注标签一致。理由是：检测到BBU设备且挡风板安装正确",
        critique="候选 0 判定正确且置信度合理，可作为参考范例。",
    )

    payload = serialize_trajectory(
        enriched_candidate,
        reflection_cycle=2,
        guidance_step=3,
    )

    assert payload["result"]["summary"].startswith("候选 0 判定为 fail")
    assert payload["result"]["critique"].startswith("候选 0 判定正确")

