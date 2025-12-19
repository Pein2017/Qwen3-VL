from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.stage_a.inference import GroupInfo, _run_cross_group_batches


def _write_png(path: Path, *, color: tuple[int, int, int] = (0, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (8, 8), color=color)
    img.save(path)


def _make_group(
    root: Path, *, group_base: str, label: str, n: int, corrupt_last: bool = False
) -> GroupInfo:
    label_dir = "审核通过" if label == "pass" else "审核不通过"
    group_dir = root / "挡风板安装检查" / label_dir / group_base
    paths: list[Path] = []
    for i in range(1, n + 1):
        p = group_dir / f"{group_base}-{i:03d}.png"
        if corrupt_last and i == n:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"not-a-real-png")
        else:
            _write_png(p, color=(i % 255, 0, 0))
        paths.append(p)
    paths.sort(key=lambda x: x.name)
    return GroupInfo(paths=paths, label=label, mission="挡风板安装检查", group_id=group_base)


def _read_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


@pytest.fixture(autouse=True)
def _stub_inference(monkeypatch: pytest.MonkeyPatch):
    import src.stage_a.inference as stage_a_inf

    def fake_infer_batch(*args, **kwargs):
        images = args[2] if len(args) > 2 else kwargs["images"]
        out = []
        for img in images:
            name = Path(getattr(img, "filename", "")).name
            out.append((f"raw:{name}", f"clean:{name}"))
        return out

    def fake_infer_one_image(*args, **kwargs):
        image = args[2] if len(args) > 2 else kwargs["image"]
        name = Path(getattr(image, "filename", "")).name
        return (f"raw:{name}", f"clean:{name}")

    monkeypatch.setattr(stage_a_inf, "infer_batch", fake_infer_batch)
    monkeypatch.setattr(stage_a_inf, "infer_one_image", fake_infer_one_image)
    monkeypatch.setattr(stage_a_inf, "apply_exif_orientation", lambda img: img)


def test_cross_group_reaggregation_and_structure(tmp_path: Path) -> None:
    g1 = _make_group(tmp_path, group_base="QC-ABC-20250101-100", label="pass", n=2)
    g2 = _make_group(tmp_path, group_base="QC-ABC-20250101-200", label="fail", n=2)
    out = tmp_path / "out.jsonl"

    processed, errors = _run_cross_group_batches(
        groups=[g1, g2],
        model=None,  # type: ignore[arg-type]
        processor=None,  # type: ignore[arg-type]
        mission="挡风板安装检查",
        gen_config={},
        batch_size=4,
        include_mission_focus=True,
        verify_inputs=False,
        output_path=out,
        pbar=None,
        distributed=False,
    )
    assert (processed, errors) == (2, 0)

    items = _read_jsonl(out)
    assert [it["group_id"] for it in items] == [g1.group_id, g2.group_id]
    for gi, it in zip([g1, g2], items):
        assert set(it.keys()) == {"group_id", "mission", "label", "images", "per_image"}
        assert it["images"] == [p.name for p in gi.paths]
        assert list(it["per_image"].keys()) == [f"image_{i}" for i in range(1, len(gi.paths) + 1)]
        assert list(it["per_image"].values()) == [f"clean:{p.name}" for p in gi.paths]


def test_cross_group_preserves_group_output_order(tmp_path: Path) -> None:
    g1 = _make_group(tmp_path, group_base="QC-XYZ-20250101-100", label="pass", n=5)
    g2 = _make_group(tmp_path, group_base="QC-XYZ-20250101-200", label="pass", n=1)
    out = tmp_path / "out.jsonl"

    processed, errors = _run_cross_group_batches(
        groups=[g1, g2],
        model=None,  # type: ignore[arg-type]
        processor=None,  # type: ignore[arg-type]
        mission="挡风板安装检查",
        gen_config={},
        batch_size=4,
        include_mission_focus=True,
        verify_inputs=False,
        output_path=out,
        pbar=None,
        distributed=False,
    )
    assert (processed, errors) == (2, 0)
    items = _read_jsonl(out)
    assert [it["group_id"] for it in items] == [g1.group_id, g2.group_id]


def test_cross_group_failure_does_not_block_later_groups(tmp_path: Path) -> None:
    g_bad = _make_group(
        tmp_path, group_base="QC-BAD-20250101-100", label="pass", n=2, corrupt_last=True
    )
    g2 = _make_group(tmp_path, group_base="QC-GOOD-20250101-200", label="pass", n=1)
    g3 = _make_group(tmp_path, group_base="QC-GOOD-20250101-300", label="fail", n=1)
    out = tmp_path / "out.jsonl"

    processed, errors = _run_cross_group_batches(
        groups=[g_bad, g2, g3],
        model=None,  # type: ignore[arg-type]
        processor=None,  # type: ignore[arg-type]
        mission="挡风板安装检查",
        gen_config={},
        batch_size=4,
        include_mission_focus=True,
        verify_inputs=False,
        output_path=out,
        pbar=None,
        distributed=False,
    )
    assert processed == 2
    assert errors == 1
    items = _read_jsonl(out)
    assert [it["group_id"] for it in items] == [g2.group_id, g3.group_id]


def test_cross_group_rank_local_shard_invariants(tmp_path: Path) -> None:
    groups = [
        _make_group(tmp_path, group_base="QC-SHARD-20250101-100", label="pass", n=1),
        _make_group(tmp_path, group_base="QC-SHARD-20250101-200", label="pass", n=1),
        _make_group(tmp_path, group_base="QC-SHARD-20250101-300", label="pass", n=1),
        _make_group(tmp_path, group_base="QC-SHARD-20250101-400", label="pass", n=1),
    ]
    shard0 = groups[0::2]
    shard1 = groups[1::2]

    out0 = tmp_path / "rank0.jsonl"
    out1 = tmp_path / "rank1.jsonl"

    _run_cross_group_batches(
        groups=shard0,
        model=None,  # type: ignore[arg-type]
        processor=None,  # type: ignore[arg-type]
        mission="挡风板安装检查",
        gen_config={},
        batch_size=8,
        include_mission_focus=True,
        verify_inputs=False,
        output_path=out0,
        pbar=None,
        distributed=False,
    )
    _run_cross_group_batches(
        groups=shard1,
        model=None,  # type: ignore[arg-type]
        processor=None,  # type: ignore[arg-type]
        mission="挡风板安装检查",
        gen_config={},
        batch_size=8,
        include_mission_focus=True,
        verify_inputs=False,
        output_path=out1,
        pbar=None,
        distributed=False,
    )

    items0 = _read_jsonl(out0)
    items1 = _read_jsonl(out1)
    assert [it["group_id"] for it in items0] == [g.group_id for g in shard0]
    assert [it["group_id"] for it in items1] == [g.group_id for g in shard1]
