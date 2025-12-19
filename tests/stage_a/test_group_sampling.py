from __future__ import annotations

from pathlib import Path

from src.stage_a.inference import GroupInfo, _sample_groups


def _mk(group_id: str, label: str) -> GroupInfo:
    return GroupInfo(paths=[Path(f"/tmp/{group_id}.jpg")], label=label, mission="M", group_id=group_id)


def test_sample_groups_disabled_returns_all() -> None:
    groups = [_mk("g0", "fail"), _mk("g1", "pass"), _mk("g2", "fail")]
    sampled, stats = _sample_groups(groups, pass_target=None, fail_target=None, seed=42)
    assert sampled == groups
    assert stats["pass_total"] == 1
    assert stats["fail_total"] == 2


def test_sample_groups_zero_targets_returns_empty() -> None:
    groups = [_mk("g0", "fail"), _mk("g1", "pass"), _mk("g2", "fail")]
    sampled, stats = _sample_groups(groups, pass_target=0, fail_target=0, seed=42)
    assert sampled == []
    assert stats["pass_selected"] == 0
    assert stats["fail_selected"] == 0


def test_sample_groups_preserves_original_order() -> None:
    groups = [
        _mk("g0", "fail"),
        _mk("g1", "pass"),
        _mk("g2", "pass"),
        _mk("g3", "fail"),
        _mk("g4", "pass"),
    ]
    sampled, stats = _sample_groups(groups, pass_target=2, fail_target=1, seed=1)
    assert stats["pass_selected"] == 2
    assert stats["fail_selected"] == 1
    # Whatever was selected, it must appear in the same relative order as `groups`.
    sampled_ids = [g.group_id for g in sampled]
    assert sampled_ids == sorted(sampled_ids, key=lambda gid: [g.group_id for g in groups].index(gid))

