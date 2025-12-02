import json
from datetime import datetime, timezone

from src.stage_b.io.guidance import GuidanceRepository
from src.stage_b.types import ReflectionProposal, ExperienceOperation


def seed_repo(tmp_path):
    guidance_path = tmp_path / "guidance.json"
    guidance_path.write_text(json.dumps({
        "M1": {
            "focus": "M1任务要点",
            "step": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "experiences": {"G0": "初始经验"}
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    repo = GuidanceRepository(guidance_path, retention=2)
    return repo


def test_remove_warns_missing_key_and_merge_validates(tmp_path):
    repo = seed_repo(tmp_path)
    # upsert G1 so we can merge it into G0
    ops = (
        ExperienceOperation(op="upsert", key="G1", text="text1", rationale=None),
    )
    proposal = ReflectionProposal(action="refine", summary=None, critique=None, operations=ops, evidence_group_ids=("g0",))
    _updated = repo.preview_reflection(  # noqa: F841
        mission="M1",
        proposal=proposal,
        reflection_id="r0",
        source_group_ids=("g0",),
    )

    # Now merge G1 into G0 and also reference a missing G2
    ops2 = (
        ExperienceOperation(op="merge", key="G0", text="merged", rationale=None, merged_from=("G1", "G2")),
        ExperienceOperation(op="remove", key="G9", text=None, rationale=None),
    )
    proposal2 = ReflectionProposal(action="refine", summary=None, critique=None, operations=ops2, evidence_group_ids=("g1",))
    updated2 = repo.preview_reflection(
        mission="M1",
        proposal=proposal2,
        reflection_id="r1",
        source_group_ids=("g1",),
    )

    assert "G0" in updated2.experiences and updated2.experiences["G0"] == "merged"
    # G1 should be removed by merge; G2 never existed so it's ignored
    assert "G1" not in updated2.experiences
    # Remove of G9 should have no effect
    assert "G9" not in updated2.experiences


def test_reject_ops_that_empty_experiences(tmp_path):
    repo = seed_repo(tmp_path)

    # Attempt to remove G0 only experience; should raise
    ops = (
        ExperienceOperation(op="remove", key="G0", text=None, rationale=None),
    )
    proposal = ReflectionProposal(action="refine", summary=None, critique=None, operations=ops, evidence_group_ids=("g2",))
    try:
        repo.preview_reflection(
            mission="M1",
            proposal=proposal,
            reflection_id="r2",
            source_group_ids=("g2",),
        )
    except Exception as e:
        assert "non-empty" in str(e).lower() or "cannot empty" in str(e).lower()
    else:
        raise AssertionError("Expected exception rejecting empty experiences")

