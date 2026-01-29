from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.stage_a.inference import (
    GroupInfo,
    _build_image_jobs,
    _merge_per_image_outputs,
    _run_per_image_jobs,
)


def _write_png(path: Path, *, color: tuple[int, int, int] = (0, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (8, 8), color=color)
    img.save(path)


def _make_group(root: Path, *, group_base: str, label: str, n: int) -> GroupInfo:
    label_dir = "审核通过" if label == "pass" else "审核不通过"
    group_dir = root / "挡风板安装检查" / label_dir / group_base
    paths: list[Path] = []
    for i in range(1, n + 1):
        p = group_dir / f"{group_base}-{i:03d}.png"
        _write_png(p, color=(i % 255, 0, 0))
        paths.append(p)
    paths.sort(key=lambda x: x.name)
    return GroupInfo(
        paths=paths, label=label, mission="挡风板安装检查", group_id=group_base
    )


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
        images = args[1] if len(args) > 1 else kwargs["images"]
        out = []
        for img in images:
            name = Path(getattr(img, "filename", "")).name
            out.append((f"raw:{name}", f"clean:{name}"))
        return out

    def fake_infer_one_image(*args, **kwargs):
        image = args[1] if len(args) > 1 else kwargs["image"]
        name = Path(getattr(image, "filename", "")).name
        return (f"raw:{name}", f"clean:{name}")

    monkeypatch.setattr(stage_a_inf, "infer_batch", fake_infer_batch)
    monkeypatch.setattr(stage_a_inf, "infer_one_image", fake_infer_one_image)
    monkeypatch.setattr(stage_a_inf, "apply_exif_orientation", lambda img: img)


def test_per_image_single_rank_end_to_end_merge_and_cleanup(tmp_path: Path) -> None:
    g1 = _make_group(tmp_path, group_base="QC-IMG-20250101-100", label="pass", n=2)
    g2 = _make_group(tmp_path, group_base="QC-IMG-20250101-200", label="fail", n=1)
    groups = [g1, g2]

    out_dir = tmp_path / "out"
    mission = "挡风板安装检查"
    intermediate = out_dir / f"{mission}_stage_a.images.rank0.jsonl"

    jobs = _build_image_jobs(groups)
    processed, errors = _run_per_image_jobs(
        jobs=jobs,
        groups=groups,
        engine=None,  # type: ignore[arg-type]
        mission=mission,
        dataset="bbu",
        prompt_profile="summary_runtime",
        gen_config={},
        batch_size=2,
        verify_inputs=False,
        output_path=intermediate,
        pbar=None,
        distributed=False,
    )
    assert (processed, errors) == (3, 0)

    merged, failed = _merge_per_image_outputs(
        groups=groups,
        output_dir=out_dir,
        mission=mission,
        world_size=1,
        keep_intermediate_outputs=False,
        dataset="bbu",
    )
    assert (merged, failed) == (2, 0)
    assert not intermediate.exists()

    final = out_dir / f"{mission}_stage_a.jsonl"
    items = _read_jsonl(final)
    assert [it["group_id"] for it in items] == [g1.group_id, g2.group_id]
    for gi, it in zip(groups, items):
        assert set(it.keys()) == {"group_id", "mission", "label", "images", "per_image"}
        assert it["images"] == [p.name for p in gi.paths]
        assert list(it["per_image"].keys()) == [
            f"image_{i}" for i in range(1, len(gi.paths) + 1)
        ]
        assert list(it["per_image"].values()) == [f"clean:{p.name}" for p in gi.paths]


def test_per_image_merge_marks_incomplete_group_failed_and_continues(
    tmp_path: Path,
) -> None:
    g1 = _make_group(tmp_path, group_base="QC-MISS-20250101-100", label="pass", n=2)
    g2 = _make_group(tmp_path, group_base="QC-MISS-20250101-200", label="fail", n=1)
    groups = [g1, g2]

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    mission = "挡风板安装检查"
    intermediate = out_dir / f"{mission}_stage_a.images.rank0.jsonl"

    # Only emit image_1 for group_seq=0 (missing image_2), but a full record for group_seq=1.
    lines = [
        {"group_seq": 0, "image_index": 1, "ok": True, "summary": "ok-0-1"},
        {"group_seq": 1, "image_index": 1, "ok": True, "summary": "ok-1-1"},
    ]
    intermediate.write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n",
        encoding="utf-8",
    )

    merged, failed = _merge_per_image_outputs(
        groups=groups,
        output_dir=out_dir,
        mission=mission,
        world_size=1,
        keep_intermediate_outputs=True,
        dataset="bbu",
    )
    assert (merged, failed) == (1, 1)
    assert intermediate.exists()

    final = out_dir / f"{mission}_stage_a.jsonl"
    items = _read_jsonl(final)
    assert [it["group_id"] for it in items] == [g2.group_id]
