"""Unit tests for updated guidance repository schema (step, experiences dict)."""

import json
import re
from datetime import datetime, timezone

import pytest

from src.stage_b.io import GuidanceRepository, MissionGuidanceError
from src.stage_b.types import ExperienceOperation, ReflectionProposal


def test_guidance_repository_load_step_and_experiences(tmp_path):
    """Test loading guidance with step and experiences dict schema."""
    guidance_path = tmp_path / "guidance.json"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Test case 1: Valid guidance with step and experiences dict
    payload1 = {
        "挡风板安装检查": {
            "focus": "挡风板安装检查任务要点",
            "step": 1,
            "updated_at": now_iso,
            "experiences": {
                "G0": "若挡风板缺失则判定不通过",
                "G1": "摘要置信度低时请返回不通过并说明原因",
            },
        }
    }

    with guidance_path.open("w", encoding="utf-8") as fh:
        json.dump(payload1, fh, ensure_ascii=False, indent=2)

    repo = GuidanceRepository(guidance_path, retention=1)
    guidance_map = repo.load()

    assert "挡风板安装检查" in guidance_map
    guidance = guidance_map["挡风板安装检查"]
    assert guidance.step == 1
    assert guidance.experiences == {
        "G0": "若挡风板缺失则判定不通过",
        "G1": "摘要置信度低时请返回不通过并说明原因",
    }
    assert guidance.metadata == {}
    assert guidance.focus == "挡风板安装检查任务要点"
    assert guidance.updated_at.isoformat() == now_iso


def test_guidance_repository_write_step_and_experiences(tmp_path):
    """Test writing guidance with step and experiences dict schema."""
    guidance_path = tmp_path / "guidance.json"

    # Write guidance with step and experiences directly to file
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = {
        "挡风板安装检查": {
            "focus": "挡风板安装检查任务要点",
            "step": 2,
            "updated_at": now_iso,
            "experiences": {
                "G0": "若挡风板缺失则判定不通过",
                "G1": "摘要置信度低时请返回不通过并说明原因",
            },
        }
    }

    with guidance_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # Create repo and load (should invalidate cache)
    repo = GuidanceRepository(guidance_path, retention=1)
    repo.invalidate()  # Clear any cache
    guidance_map = repo.load()
    assert "挡风板安装检查" in guidance_map
    guidance = guidance_map["挡风板安装检查"]
    assert guidance.step == 2
    assert guidance.experiences == {
        "G0": "若挡风板缺失则判定不通过",
        "G1": "摘要置信度低时请返回不通过并说明原因",
    }


def test_guidance_repository_apply_reflection_with_operations(tmp_path):
    """Test applying incremental reflection operations."""
    guidance_path = tmp_path / "guidance.json"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Initialize with step 1
    initial_payload = {
        "挡风板安装检查": {
            "focus": "挡风板安装检查任务要点",
            "step": 1,
            "updated_at": now_iso,
            "experiences": {
                "G0": "若挡风板缺失则判定不通过",
            },
        }
    }

    with guidance_path.open("w", encoding="utf-8") as fh:
        json.dump(initial_payload, fh, ensure_ascii=False, indent=2)

    repo = GuidanceRepository(guidance_path, retention=1)

    proposal = ReflectionProposal(
        action="refine",
        summary="挡板缺失仍被放行",
        critique="需要补充挡板缺失规则",
        operations=(
            ExperienceOperation(
                op="upsert",
                key="G0",
                text="若挡风板缺失则判定不通过并提示复检",
                rationale="现有规则过于宽泛",
                evidence=("QC-001",),
            ),
            ExperienceOperation(
                op="upsert",
                key=None,
                text="若摘要提到挡板松动，应标记为不通过并要求现场复查",
                rationale=None,
                evidence=("QC-002",),
            ),
        ),
        evidence_group_ids=("QC-001", "QC-002"),
        uncertainty_note=None,
        text=None,
    )

    updated_guidance = repo.apply_reflection(
        mission="挡风板安装检查",
        proposal=proposal,
        reflection_id="test123",
        source_group_ids=["QC-001", "QC-002"],
        applied_epoch=1,
        operations=proposal.operations,
    )

    # Verify step was incremented
    assert updated_guidance.step == 2

    # Verify experiences were merged incrementally
    assert updated_guidance.experiences["G0"] == "若挡风板缺失则判定不通过并提示复检"
    assert "挡板松动" in updated_guidance.experiences["G1"]

    # Metadata should be recorded per key
    meta_g0 = updated_guidance.metadata["G0"]
    assert meta_g0.reflection_id == "test123"
    assert meta_g0.sources == ("QC-001", "QC-002")
    assert meta_g0.rationale == "现有规则过于宽泛"

    meta_g1 = updated_guidance.metadata["G1"]
    assert meta_g1.sources == ("QC-002", "QC-001")

    # Reload and verify persistence
    guidance_map = repo.load()
    reloaded = guidance_map["挡风板安装检查"]
    assert reloaded.step == 2
    assert reloaded.experiences["G0"] == "若挡风板缺失则判定不通过并提示复检"
    assert "挡板松动" in reloaded.experiences["G1"]


def test_guidance_repository_empty_experiences_raises_error(tmp_path):
    """Test that empty experiences dict raises error."""
    guidance_path = tmp_path / "guidance.json"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Test case 1: Empty experiences dict in file
    payload1 = {
        "挡风板安装检查": {
            "focus": "挡风板安装检查任务要点",
            "step": 1,
            "updated_at": now_iso,
            "experiences": {},
        }
    }

    with guidance_path.open("w", encoding="utf-8") as fh:
        json.dump(payload1, fh, ensure_ascii=False, indent=2)

    repo = GuidanceRepository(guidance_path, retention=1)

    with pytest.raises(MissionGuidanceError):
        repo.load()


def test_guidance_repository_missing_step_raises_error(tmp_path):
    """Test that missing step field raises error."""
    guidance_path = tmp_path / "guidance.json"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Test case: Missing step field
    payload = {
        "挡风板安装检查": {
            "focus": "挡风板安装检查任务要点",
            "updated_at": now_iso,
            "experiences": {
                "G0": "若挡风板缺失则判定不通过",
            },
        }
    }

    with guidance_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    repo = GuidanceRepository(guidance_path, retention=1)

    with pytest.raises(MissionGuidanceError):
        repo.load()


def test_guidance_repository_remove_all_experiences_raises(tmp_path):
    """Removing all experiences should raise MissionGuidanceError."""

    guidance_path = tmp_path / "guidance.json"
    repo = GuidanceRepository(guidance_path, retention=2)
    repo.ensure_initialized()
    repo.ensure_mission("挡风板安装检查")

    original_guidance = repo.get("挡风板安装检查")
    assert original_guidance.experiences

    proposal = ReflectionProposal(
        action="refine",
        summary=None,
        critique=None,
        operations=(
            ExperienceOperation(
                op="remove",
                key=next(iter(original_guidance.experiences.keys())),
                text=None,
                rationale=None,
                evidence=(),
            ),
        ),
        evidence_group_ids=("QC-REMOVE",),
        uncertainty_note=None,
        text=None,
    )

    with pytest.raises(MissionGuidanceError, match="must be non-empty"):
        repo.apply_reflection(
            mission="挡风板安装检查",
            proposal=proposal,
            reflection_id="remove-all",
            source_group_ids=["QC-REMOVE"],
            operations=proposal.operations,
        )


def test_guidance_repository_creates_snapshot_atomic_write(tmp_path):
    """Updating guidance should snapshot previous file with microsecond timestamp."""

    guidance_path = tmp_path / "guidance.json"
    repo = GuidanceRepository(guidance_path, retention=2)
    repo.ensure_initialized()
    repo.ensure_mission("挡风板安装检查")

    with guidance_path.open("r", encoding="utf-8") as fh:
        original_payload = json.load(fh)

    snapshot_dir = guidance_path.parent / "snapshots"
    if snapshot_dir.exists():
        for path in snapshot_dir.glob("guidance-*.json"):
            path.unlink()

    proposal = ReflectionProposal(
        action="refine",
        summary="新增挡板缺失提醒",
        critique="原有经验未覆盖挡板缺失场景",
        operations=(
            ExperienceOperation(
                op="upsert",
                key="G0",
                text="若挡风板缺失或方向错误，应判定为不通过并提示复检",
                rationale="补齐挡板缺失规则",
                evidence=("QC-001",),
            ),
        ),
        evidence_group_ids=("QC-001",),
        uncertainty_note=None,
        text=None,
    )

    updated = repo.apply_reflection(
        mission="挡风板安装检查",
        proposal=proposal,
        reflection_id="snapshot-test",
        source_group_ids=["QC-001"],
        operations=proposal.operations,
    )

    assert updated.step == original_payload["挡风板安装检查"]["step"] + 1
    assert "挡风板缺失" in updated.experiences["G0"]

    assert guidance_path.exists()
    with guidance_path.open("r", encoding="utf-8") as fh:
        refreshed_payload = json.load(fh)
    assert "挡风板缺失" in refreshed_payload["挡风板安装检查"]["experiences"]["G0"]

    assert snapshot_dir.exists()
    snapshots = sorted(snapshot_dir.glob("guidance-*.json"))
    assert snapshots, "Expected snapshot file to be created"
    snapshot_name = snapshots[0].name
    assert re.match(r"guidance-\d{8}-\d{6}-\d{6}\.json", snapshot_name)

    with snapshots[0].open("r", encoding="utf-8") as fh:
        snapshot_payload = json.load(fh)
    assert snapshot_payload == original_payload
