from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import pytest

from src.config.prompts import (
    SYSTEM_PROMPT_AUX,
    SYSTEM_PROMPT_JSON,
    USER_PROMPT_AUX,
    USER_PROMPT_JSON,
    get_template_prompts,
)
from src.datasets.fusion import FusionConfig
from src.datasets.preprocessors.augmentation import AugmentationPreprocessor
from src.datasets.unified_fusion_dataset import (
    FusionCaptionDataset,
    UnifiedFusionDataset,
    fusion_pack_group_key,
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
    objects: list[dict[str, Any]] = [
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
    target_mode: str | None = None,
    source_mode: str | None = None,
    source_overrides: dict[str, Any] | None = None,
    target_val: Path | None = None,
    source_val: Path | None = None,
    target_template: str = "target_dense",
    source_template: str = "source_dense",
) -> None:
    target_section: dict[str, Any] = {
        "dataset": "bbu",
        "name": target_name,
        "train_jsonl": str(target_train),
        "template": target_template,
    }
    if target_mode:
        target_section["mode"] = target_mode
    if target_val is not None:
        target_section["val_jsonl"] = str(target_val)
    source_section: dict[str, Any] = {
        "dataset": "coco",
        "name": source_name,
        "train_jsonl": str(source_train),
        "ratio": ratio,
        "template": source_template,
    }
    if source_mode:
        source_section["mode"] = source_mode
    if source_val is not None:
        source_section["val_jsonl"] = str(source_val)
    if source_overrides:
        source_section.update(source_overrides)
    payload = {"targets": [target_section], "sources": [source_section]}
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
        target_template="target_dense",
        source_template="source_dense",
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


def test_unified_fusion_mixed_modes(tmp_path: Path) -> None:
    target_jsonl = tmp_path / "target.jsonl"
    source_jsonl = tmp_path / "source.jsonl"
    _write_jsonl(
        target_jsonl,
        [{"images": ["img.jpg"], "summary": "设备完好", "width": 10, "height": 10}],
    )
    _write_jsonl(source_jsonl, [_basic_record()])

    config_path = tmp_path / "fusion.json"
    _write_fusion_config(
        config_path,
        target_train=target_jsonl,
        source_train=source_jsonl,
        target_mode="summary",
        source_mode="dense",
        target_template="summary",
        source_template="source_dense",
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
        seed=11,
        shuffle=False,
    )

    target_sample = dataset[0]
    assert dataset.epoch_plan["target"]["mode"] == "summary"
    summary_system, summary_user = get_template_prompts("summary")
    assert target_sample["messages"][0]["content"] == summary_system
    assert target_sample["messages"][1]["content"][-1]["text"] == summary_user
    assert dataset.last_sample_debug["mode"] == "summary"

    source_sample = dataset[1]
    assert dataset.epoch_plan["source"]["mode"] == "dense"
    assert source_sample["messages"][0]["content"] == SYSTEM_PROMPT_AUX
    assert source_sample["messages"][1]["content"][-1]["text"] == USER_PROMPT_AUX
    assert dataset.last_sample_debug["mode"] == "dense"
    assert template.system == "BASE_SYS"


def test_unified_fusion_irrelevant_dense_emits_single_line(tmp_path: Path) -> None:
    target_jsonl = tmp_path / "target.jsonl"
    source_jsonl = tmp_path / "source.jsonl"
    _write_jsonl(target_jsonl, [_basic_record()])
    _write_jsonl(
        source_jsonl,
        [
            {
                "images": ["img.jpg"],
                "objects": [{"bbox_2d": [0, 0, 10, 10], "desc": "irrelevant"}],
                "summary": "无关图片",
                "width": 10,
                "height": 10,
            }
        ],
    )

    config_path = tmp_path / "fusion.json"
    _write_fusion_config(
        config_path,
        target_train=target_jsonl,
        source_train=source_jsonl,
        ratio=1.0,
        source_name="irrelevant_dense",
        target_template="target_dense",
        source_template="source_dense",
        source_mode="dense",
    )

    dataset = FusionCaptionDataset(
        fusion_config=FusionConfig.from_file(str(config_path)),
        base_template=_StubTemplate(),
        user_prompt="DEFAULT_USER",
        emit_norm="none",
        json_format="standard",
        assistant_prefix_format="<DOMAIN={domain}>, <TASK={task}>",
        augmenter=None,
        bypass_prob=0.0,
        curriculum_state=None,
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
        seed=23,
        shuffle=False,
    )

    target_sample = dataset[0]
    target_text = target_sample["messages"][-1]["content"][0]["text"]
    assert target_text.splitlines()[0] == "<DOMAIN=BBU>, <TASK=DETECTION>"

    irrelevant_sample = dataset[1]
    irrelevant_text = irrelevant_sample["messages"][-1]["content"][0]["text"]
    assert irrelevant_text == "无关图片"


def test_unified_fusion_dense_requires_geometry(tmp_path: Path) -> None:
    target_jsonl = tmp_path / "target.jsonl"
    source_jsonl = tmp_path / "source.jsonl"
    _write_jsonl(
        target_jsonl,
        [
            {
                "images": ["img.jpg"],
                "objects": [{"desc": "missing geometry"}],
                "width": 10,
                "height": 10,
            }
        ],
    )
    _write_jsonl(source_jsonl, [_basic_record()])
    config_path = tmp_path / "fusion.json"
    _write_fusion_config(
        config_path,
        target_train=target_jsonl,
        source_train=source_jsonl,
        ratio=0.0,
        source_name="src",
        target_template="target_dense",
        source_template="source_dense",
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
        seed=17,
        shuffle=False,
    )

    with pytest.raises(ValueError):
        _ = dataset[0]


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
        target_template="target_dense",
        source_template="source_dense",
    )

    calls: list[str | None] = []

    def _fake_preprocess(self: AugmentationPreprocessor, row: Any) -> Any:
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
        target_template="target_dense",
        source_template="source_dense",
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
        target_template="target_dense",
        source_template="source_dense",
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
        target_template="target_dense",
        source_template="source_dense",
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
        target_template="target_dense",
        source_template="source_dense",
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

    assert len(dataset) == 1  # target-only eval
    assert dataset.aux_quota == {}
    assert dataset.epoch_plan["target"]["augmentation"] is False


def test_unified_fusion_alias_points_to_fusion_class() -> None:
    assert UnifiedFusionDataset is FusionCaptionDataset


def test_fusion_pack_group_key_respects_domain() -> None:
    rec_target: dict[str, object] = {"metadata": {"_fusion_domain": "target"}}
    rec_source: dict[str, object] = {"metadata": {"_fusion_domain": "source"}}
    rec_missing: dict[str, object] = {"metadata": {}}

    assert fusion_pack_group_key(rec_target) == "target"
    assert fusion_pack_group_key(rec_source) == "source"
    assert fusion_pack_group_key(rec_missing) == "target"
