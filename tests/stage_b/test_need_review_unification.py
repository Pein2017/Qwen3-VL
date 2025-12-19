from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.stage_b.io.guidance import GuidanceRepository
from src.stage_b.runner import (
    _unexplainable_need_review_items,
    _write_need_review_summary,
)
from src.stage_b.types import (
    DeterministicSignals,
    ExperienceCandidate,
    ExperienceMetadata,
    ExperienceRecord,
    GroupTicket,
    MissionGuidance,
    ReflectionOutcome,
    ReflectionProposal,
    StageASummaries,
)


def test_increment_hit_count_updates_confidence(tmp_path: Path) -> None:
    repo = GuidanceRepository(tmp_path / "guidance.json", retention=2)
    now = datetime.now(timezone.utc)
    repo._write(
        {
            "m": MissionGuidance(
                mission="m",
                experiences={"G0": "初始经验", "G1": "规则1"},
                step=1,
                updated_at=now,
                metadata={
                    "G1": ExperienceMetadata(
                        updated_at=now,
                        reflection_id="r",
                        sources=("QC-1",),
                        rationale=None,
                        hit_count=0,
                        miss_count=2,
                        confidence=0.0,
                    )
                },
            )
        }
    )

    repo.increment_hit_count("m", ["G1"])
    updated = repo.load()["m"].metadata["G1"]
    assert updated.hit_count == 1
    assert updated.miss_count == 2
    assert 0.0 < updated.confidence < 1.0


def test_unexplainable_need_review_items_emits_entry(tmp_path: Path) -> None:
    ticket = GroupTicket(
        group_id="QC-001",
        mission="m",
        label="fail",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "摘要"}),
    )
    cand = ExperienceCandidate(
        candidate_index=0,
        verdict="pass",
        reason="模型判通过",
        signals=DeterministicSignals(label_match=False, self_consistency=None),
    )
    record = ExperienceRecord(
        ticket=ticket,
        candidates=(cand,),
        winning_candidate=0,
        guidance_step=1,
    )
    proposal = ReflectionProposal(
        action="noop",  # type: ignore[arg-type]
        summary="noop",
        critique="noop",
        operations=tuple(),
        evidence_group_ids=("QC-001",),
        uncertainty_note="no_evidence_for_label",
        text=None,
    )
    outcome = ReflectionOutcome(
        reflection_id="r1",
        mission="m",
        proposal=proposal,
        applied=True,
        guidance_step_before=1,
        guidance_step_after=1,
        operations=tuple(),
        eligible=True,
        applied_epoch=None,
        ineligible_reason=None,
        warnings=tuple(),
    )

    items = _unexplainable_need_review_items(
        [record],
        outcome=outcome,
        epoch=3,
        reflection_cycle=7,
    )
    assert len(items) == 1
    _, entry = items[0]
    assert entry["ticket_key"] == "QC-001::fail"
    assert entry["gt_label"] == "fail"
    assert entry["pred_verdict"] == "pass"
    assert entry["reason_code"] == "reflection_no_evidence_after_gt"
    assert entry["reflection_id"] == "r1"
    assert entry["reflection_cycle"] == 7


def test_unexplainable_need_review_items_skips_explainable_case(tmp_path: Path) -> None:
    ticket = GroupTicket(
        group_id="QC-002",
        mission="m",
        label="fail",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "摘要"}),
    )
    cand = ExperienceCandidate(
        candidate_index=0,
        verdict="pass",
        reason="模型判通过",
        signals=DeterministicSignals(label_match=False, self_consistency=None),
    )
    record = ExperienceRecord(
        ticket=ticket,
        candidates=(cand,),
        winning_candidate=0,
        guidance_step=1,
    )
    proposal = ReflectionProposal(
        action="refine",  # type: ignore[arg-type]
        summary="refine",
        critique="refine",
        operations=tuple(),
        evidence_group_ids=("QC-002",),
        uncertainty_note=None,
        no_evidence_group_ids=tuple(),
        text=None,
    )
    outcome = ReflectionOutcome(
        reflection_id="r2",
        mission="m",
        proposal=proposal,
        applied=True,
        guidance_step_before=1,
        guidance_step_after=2,
        operations=tuple(),
        eligible=True,
        applied_epoch=None,
        ineligible_reason=None,
        warnings=tuple(),
    )

    items = _unexplainable_need_review_items(
        [record],
        outcome=outcome,
        epoch=1,
        reflection_cycle=0,
    )
    assert items == []


def test_unexplainable_need_review_items_uses_no_evidence_group_ids(
    tmp_path: Path,
) -> None:
    ticket = GroupTicket(
        group_id="QC-003",
        mission="m",
        label="fail",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "摘要"}),
    )
    cand = ExperienceCandidate(
        candidate_index=0,
        verdict="pass",
        reason="模型判通过",
        signals=DeterministicSignals(label_match=False, self_consistency=None),
    )
    record = ExperienceRecord(
        ticket=ticket,
        candidates=(cand,),
        winning_candidate=0,
        guidance_step=1,
    )
    proposal = ReflectionProposal(
        action="refine",  # type: ignore[arg-type]
        summary="refine",
        critique="refine",
        operations=tuple(),
        evidence_group_ids=("QC-003::fail",),
        uncertainty_note=None,
        no_evidence_group_ids=("QC-003::fail",),
        text=None,
    )
    outcome = ReflectionOutcome(
        reflection_id="r3",
        mission="m",
        proposal=proposal,
        applied=True,
        guidance_step_before=1,
        guidance_step_after=2,
        operations=tuple(),
        eligible=True,
        applied_epoch=None,
        ineligible_reason=None,
        warnings=tuple(),
    )

    items = _unexplainable_need_review_items(
        [record],
        outcome=outcome,
        epoch=1,
        reflection_cycle=0,
    )
    assert len(items) == 1


def test_unexplainable_need_review_items_reflection_is_single_source_of_truth(
    tmp_path: Path,
) -> None:
    """Need-review routing MUST be decided by reflection at group_id granularity.

    Even if a candidate pool contains a GT-aligned candidate (label_match=True),
    the ticket MUST still be routed to need-review when reflection explicitly
    marks the group as "no evidence after GT".
    """
    ticket = GroupTicket(
        group_id="QC-004",
        mission="m",
        label="fail",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "摘要"}),
    )
    cand = ExperienceCandidate(
        candidate_index=0,
        verdict="fail",
        reason="模型判不通过（但理由缺乏可审计证据）",
        signals=DeterministicSignals(label_match=True, self_consistency=None),
    )
    record = ExperienceRecord(
        ticket=ticket,
        candidates=(cand,),
        winning_candidate=0,
        guidance_step=1,
    )
    proposal = ReflectionProposal(
        action="refine",  # type: ignore[arg-type]
        summary="refine",
        critique="refine",
        operations=tuple(),
        evidence_group_ids=("QC-004::fail",),
        uncertainty_note=None,
        no_evidence_group_ids=("QC-004::fail",),
        text=None,
    )
    outcome = ReflectionOutcome(
        reflection_id="r4",
        mission="m",
        proposal=proposal,
        applied=True,
        guidance_step_before=1,
        guidance_step_after=2,
        operations=tuple(),
        eligible=True,
        applied_epoch=None,
        ineligible_reason=None,
        warnings=tuple(),
    )

    items = _unexplainable_need_review_items(
        [record],
        outcome=outcome,
        epoch=1,
        reflection_cycle=0,
    )
    assert len(items) == 1
    _, entry = items[0]
    assert entry["group_id"] == "QC-004"
    assert entry["reason_code"] == "reflection_no_evidence_after_gt"


def test_write_need_review_summary_is_deterministic(tmp_path: Path) -> None:
    queue_path = tmp_path / "need_review_queue.jsonl"
    queue_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"ticket_key": "b", "group_id": "2", "reason_code": "z"},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {"ticket_key": "a", "group_id": "1", "reason_code": "y"},
                    ensure_ascii=False,
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "need_review.json"
    _write_need_review_summary(
        mission="m",
        need_review_queue_path=queue_path,
        output_path=out_path,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["mission"] == "m"
    assert payload["n"] == 2
    assert payload["items"][0]["ticket_key"] == "a"
    assert payload["items"][1]["ticket_key"] == "b"
    assert payload["by_reason_code"] == {"y": 1, "z": 1}
