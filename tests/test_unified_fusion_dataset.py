from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import pytest

from src.config.prompts import (
    SYSTEM_PROMPT_JSON,
    USER_PROMPT_JSON,
)
from src.datasets.fusion import FusionConfig
from src.datasets.preprocessors.augmentation import AugmentationPreprocessor
from src.datasets.unified_fusion_dataset import (
    FusionCaptionDataset,
    UnifiedFusionDataset,
)


@dataclass
class _StubTemplate:
    system: str = "BASE_SYS"

    def __post_init__(self) -> None:
        self.mode = "train"
        self.encode_calls: List[str] = []

    def encode(self, merged: Any, return_length: bool = True) -> dict[str, Any]:
        # Record the system prompt used for encoding for assertions
        self.encode_calls.append(self.system)
        return {
            "input_ids": [1, 2, 3],
            "pixel_values": [42],
            "image_grid_thw": [1, 1, 1],
        }

    def set_mode(self, mode: str) -> None:
        self.mode = mode


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(rec) for rec in records), encoding="utf-8")


def _basic_record(count: int = 1) -> dict[str, Any]:
    objects = [
        {"bbox_2d": [0, 0, 10, 10], "desc": f"obj_{idx}"} for idx in range(count)
    ]
    return {"images": ["img.jpg"], "objects": objects, "width": 10, "height": 10}


def _write_fusion_config(
    path: Path,
    *,
    target_train: Path,
    source_train: Path,
    ratio: float = 1.0,
    target_name: str = "target",
    source_name: str = "source",
    source_overrides: dict[str, Any] | None = None,
    target_val: Path | None = None,
    source_val: Path | None = None,
) -> None:
    target_section: dict[str, Any] = {
        "dataset": "bbu",
        "name": target_name,
        "train_jsonl": str(target_train),
    }
    if target_val is not None:
        target_section["val_jsonl"] = str(target_val)
    source_section: dict[str, Any] = {
        "dataset": "coco",
        "name": source_name,
        "train_jsonl": str(source_train),
        "ratio": ratio,
    }
    if source_val is not None:
        source_section["val_jsonl"] = str(source_val)
    if source_overrides:
        source_section.update(source_overrides)
    payload = {"target": target_section, "sources": [source_section]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_unified_fusion_prompt_priority(tmp_path: Path) -> None:
    target_jsonl = tmp_path / "target.jsonl"
    source_jsonl = tmp_path / "source.jsonl"
    _write_jsonl(target_jsonl, [_basic_record()])
    _write_jsonl(source_jsonl, [_basic_record()])

    config_path = tmp_path / "fusion.json"
    _write_fusion_config(
        config_path,
        target_train=target_jsonl,
        source_train=source_jsonl,
        source_overrides={
            "user_prompt": "SRC_USER",
            "system_prompt": "SRC_SYSTEM",
        },
    )

    fusion_config = FusionConfig.from_file(str(config_path))
    template = _StubTemplate()
    dataset = FusionCaptionDataset(
        fusion_config=fusion_config,
        base_template=template,
        user_prompt="DEFAULT_USER",
        emit_norm="none",
        json_format="standard",
        augmenter=None,
        bypass_prob=0.0,
        curriculum_state=None,
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
        seed=7,
        shuffle=False,
    )

    target_sample = dataset[0]
    assert target_sample["messages"][0]["content"] == SYSTEM_PROMPT_JSON
    assert target_sample["messages"][1]["content"][-1]["text"] == USER_PROMPT_JSON
    assert dataset.last_sample_debug["prompt_source"] == "domain"

    source_sample = dataset[1]
    assert source_sample["messages"][0]["content"] == "SRC_SYSTEM"
    assert source_sample["messages"][1]["content"][-1]["text"] == "SRC_USER"
    assert dataset.last_sample_debug["prompt_source"] == "dataset"
    assert template.system == "BASE_SYS"


def test_unified_fusion_respects_source_augmentation_gating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_jsonl = tmp_path / "target.jsonl"
    source_jsonl = tmp_path / "source.jsonl"
    _write_jsonl(target_jsonl, [_basic_record()])
    _write_jsonl(source_jsonl, [_basic_record()])
    config_path = tmp_path / "fusion.json"
    _write_fusion_config(
        config_path,
        target_train=target_jsonl,
        source_train=source_jsonl,
        target_name="target_ds",
        source_name="src_ds",
    )

    calls: list[str | None] = []

    def _fake_preprocess(self, row: Any) -> Any:  # noqa: ANN001
        calls.append(row.get("metadata", {}).get("_fusion_source"))
        return row

    monkeypatch.setattr(AugmentationPreprocessor, "preprocess", _fake_preprocess)

    dataset = FusionCaptionDataset(
        fusion_config=FusionConfig.from_file(str(config_path)),
        base_template=_StubTemplate(),
        user_prompt="DEFAULT_USER",
        emit_norm="none",
        json_format="standard",
        augmenter=object(),  # trigger augmentation preprocessor construction
        bypass_prob=0.0,
        curriculum_state=None,
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
        seed=13,
        shuffle=False,
    )

    _ = dataset[0]  # target sample
    _ = dataset[1]  # source sample

    assert calls == ["target_ds"]
    assert dataset.epoch_plan["target_ds"]["augmentation"] is True
    assert dataset.epoch_plan["src_ds"]["augmentation"] is False
    assert dataset.last_sample_debug["augmentation_enabled"] is False


def test_unified_fusion_applies_object_cap_after_augmentation(tmp_path: Path) -> None:
    target_jsonl = tmp_path / "target.jsonl"
    source_jsonl = tmp_path / "source.jsonl"
    _write_jsonl(target_jsonl, [_basic_record(count=1)])
    _write_jsonl(source_jsonl, [_basic_record(count=4)])
    config_path = tmp_path / "fusion.json"
    _write_fusion_config(
        config_path,
        target_train=target_jsonl,
        source_train=source_jsonl,
        source_overrides={"max_objects_per_image": 2},
    )

    dataset = FusionCaptionDataset(
        fusion_config=FusionConfig.from_file(str(config_path)),
        base_template=_StubTemplate(),
        user_prompt="DEFAULT_USER",
        emit_norm="none",
        json_format="standard",
        augmenter=None,
        bypass_prob=0.0,
        curriculum_state=None,
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
        seed=19,
        shuffle=False,
    )

    source_sample = dataset[1]
    assert len(source_sample["assistant_payload"]) == 2
    assert dataset.last_sample_debug["object_cap_applied"] is True
    assert dataset.last_sample_debug["object_cap_limit"] == 2


def test_unified_fusion_default_source_cap(tmp_path: Path) -> None:
    target_jsonl = tmp_path / "target.jsonl"
    source_jsonl = tmp_path / "source.jsonl"
    _write_jsonl(target_jsonl, [_basic_record(count=1)])
    _write_jsonl(source_jsonl, [_basic_record(count=70)])
    config_path = tmp_path / "fusion.json"
    _write_fusion_config(
        config_path,
        target_train=target_jsonl,
        source_train=source_jsonl,
    )

    dataset = FusionCaptionDataset(
        fusion_config=FusionConfig.from_file(str(config_path)),
        base_template=_StubTemplate(),
        user_prompt="DEFAULT_USER",
        emit_norm="none",
        json_format="standard",
        augmenter=None,
        bypass_prob=0.0,
        curriculum_state=None,
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
        seed=23,
        shuffle=False,
    )

    source_sample = dataset[1]
    assert len(source_sample["assistant_payload"]) == 64  # default source cap
    assert dataset.last_sample_debug["object_cap_applied"] is True
    assert dataset.last_sample_debug["object_cap_limit"] == 64


def test_unified_fusion_resamples_sources_per_epoch(tmp_path: Path) -> None:
    target_jsonl = tmp_path / "target.jsonl"
    source_jsonl = tmp_path / "source.jsonl"
    _write_jsonl(target_jsonl, [_basic_record(), _basic_record(), _basic_record()])
    _write_jsonl(source_jsonl, [_basic_record(), _basic_record()])
    config_path = tmp_path / "fusion.json"
    _write_fusion_config(
        config_path,
        target_train=target_jsonl,
        source_train=source_jsonl,
        ratio=0.5,
        source_name="src",
    )

    dataset = FusionCaptionDataset(
        fusion_config=FusionConfig.from_file(str(config_path)),
        base_template=_StubTemplate(),
        user_prompt="DEFAULT_USER",
        emit_norm="none",
        json_format="standard",
        augmenter=None,
        bypass_prob=0.0,
        curriculum_state=None,
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
        seed=31,
        shuffle=False,
    )

    assert dataset.aux_quota["src"] == 2
    assert len(dataset) == 5
    dataset.set_epoch(1)
    assert dataset.aux_quota["src"] == 2
    assert len(dataset) == 5


def test_unified_fusion_includes_optional_source_eval(tmp_path: Path) -> None:
    target_train = tmp_path / "target_train.jsonl"
    target_val = tmp_path / "target_val.jsonl"
    source_train = tmp_path / "source_train.jsonl"
    source_val = tmp_path / "source_val.jsonl"
    _write_jsonl(target_train, [_basic_record()])
    _write_jsonl(target_val, [_basic_record()])
    _write_jsonl(source_train, [_basic_record()])
    _write_jsonl(source_val, [_basic_record(), _basic_record()])

    config_path = tmp_path / "fusion.json"
    _write_fusion_config(
        config_path,
        target_train=target_train,
        source_train=source_train,
        target_val=target_val,
        source_val=source_val,
        source_name="src_eval",
    )

    dataset = FusionCaptionDataset(
        fusion_config=FusionConfig.from_file(str(config_path)),
        base_template=_StubTemplate(),
        user_prompt="DEFAULT_USER",
        emit_norm="none",
        json_format="standard",
        augmenter=None,
        bypass_prob=0.0,
        curriculum_state=None,
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
        seed=37,
        shuffle=False,
        split="eval",
    )

    assert len(dataset) == 3  # 1 target + 2 source eval samples
    assert dataset.aux_quota["src_eval"] == 2
    assert dataset.epoch_plan["src_eval"]["augmentation"] is False


def test_unified_fusion_alias_points_to_fusion_class() -> None:
    assert UnifiedFusionDataset is FusionCaptionDataset
