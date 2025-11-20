"""Fusion helpers for multi-dataset dense-caption training."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple

from .geometry import points_to_xyxy
from .utils import load_jsonl


FALLBACK_OPTIONS = ("off", "bbox_2d")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    train_jsonl: Path
    template: str
    poly_fallback: Literal["off", "bbox_2d"] = "off"
    val_jsonl: Optional[Path] = None


@dataclass(frozen=True)
class AuxiliarySpec(DatasetSpec):
    ratio: float = 0.0


@dataclass(frozen=True)
class FusionConfig:
    target: DatasetSpec
    sources: Tuple[AuxiliarySpec, ...]

    @staticmethod
    def _as_path(value: Any, *, field_name: str) -> Path:
        if not value:
            raise ValueError(f"{field_name} is required")
        return Path(value)

    @staticmethod
    def _parse_template(entry: Mapping[str, Any], default: str) -> str:
        value = entry.get("template")
        return str(value).strip() if value else default

    @classmethod
    def from_file(cls, path: str) -> "FusionConfig":
        path_obj = Path(path)
        text = path_obj.read_text(encoding="utf-8")
        suffix = path_obj.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "pyyaml is required to parse fusion configs; install pyyaml"
                ) from exc
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError("fusion config must be a mapping")

        target_section = payload.get("target")
        if not isinstance(target_section, Mapping):
            raise ValueError("fusion config must contain a 'target' mapping")
        target_name = target_section.get("name")
        target_train = target_section.get("train_jsonl")
        if target_name is None or target_train is None:
            raise ValueError("target spec must include 'name' and 'train_jsonl'")
        target_template = cls._parse_template(target_section, default="bbu_dense")
        target_poly_fallback = cls._parse_fallback(target_section)
        target_val = target_section.get("val_jsonl")

        source_sections = payload.get("sources") or []
        if not isinstance(source_sections, Iterable):
            raise ValueError("fusion sources must be an iterable")

        sources: list[AuxiliarySpec] = []
        for entry in source_sections:
            if not isinstance(entry, Mapping):
                raise ValueError("each source spec must be a mapping")
            name = entry.get("name")
            train_jsonl = entry.get("train_jsonl")
            ratio = entry.get("ratio")
            if name is None or train_jsonl is None:
                raise ValueError("auxiliary spec must include 'name' and 'train_jsonl'")
            if ratio is None:
                raise ValueError("auxiliary spec must include 'ratio'")
            try:
                ratio_value = float(ratio)
            except (TypeError, ValueError) as exc:
                raise ValueError("ratio must be numeric") from exc
            if ratio_value < 0:
                raise ValueError("ratio must be non-negative")
            template = cls._parse_template(entry, default="aux_dense")
            val_jsonl = entry.get("val_jsonl")
            sources.append(
                AuxiliarySpec(
                    name=str(name),
                    train_jsonl=Path(train_jsonl),
                    ratio=ratio_value,
                    template=template,
                    poly_fallback=cls._parse_fallback(entry),
                    val_jsonl=Path(val_jsonl) if val_jsonl else None,
                )
            )

        return cls(
            target=DatasetSpec(
                name=str(target_name),
                train_jsonl=Path(target_train),
                template=target_template,
                val_jsonl=Path(target_val) if target_val else None,
                poly_fallback=target_poly_fallback,
            ),
            sources=tuple(sources),
        )

    @staticmethod
    def _parse_fallback(entry: Mapping[str, Any]) -> Literal["off", "bbox_2d"]:
        fallback = entry.get("poly_fallback", "off")
        if fallback not in FALLBACK_OPTIONS:
            raise ValueError(f"poly_fallback must be one of {FALLBACK_OPTIONS}")
        return fallback


def _apply_poly_fallback(record: Dict[str, Any], fallback: Literal["off", "bbox_2d"]) -> None:
    if fallback != "bbox_2d":
        return
    objects = record.get("objects") or []
    for obj in objects:
        if "poly" in obj:
            pts = obj["poly"]
            obj["bbox_2d"] = points_to_xyxy(pts)
            obj.pop("poly", None)


def _annotate_record(
    record: Mapping[str, Any],
    dataset_name: str,
    template: str,
    fallback: Literal["off", "bbox_2d"],
) -> Dict[str, Any]:
    annotated = copy.deepcopy(record)
    metadata = annotated.get("metadata", {})
    if not isinstance(metadata, MutableMapping):
        metadata = {}
    metadata["dataset"] = dataset_name
    metadata["template"] = template
    annotated["metadata"] = metadata
    _apply_poly_fallback(annotated, fallback)
    return annotated


def _sample_with_replacement(
    records: Sequence[Mapping[str, Any]],
    count: int,
    rng: random.Random,
) -> list[Dict[str, Any]]:
    if not records or count <= 0:
        return []
    samples: list[Dict[str, Any]] = []
    for _ in range(count):
        choice = rng.choice(records)
        samples.append(copy.deepcopy(choice))
    return samples


def build_fused_jsonl(
    config: FusionConfig,
    output_path: str,
    *,
    seed: int = 2025,
    shuffle: bool = True,
) -> Path:
    target_records = load_jsonl(str(config.target.train_jsonl))
    rng = random.Random(seed)

    fused: list[Dict[str, Any]] = []
    for record in target_records:
        fused.append(
            _annotate_record(
                record,
                config.target.name,
                config.target.template,
                config.target.poly_fallback,
            )
        )

    for source in config.sources:
        quota = round(source.ratio * len(target_records))
        if quota <= 0:
            continue
        source_records = load_jsonl(str(source.train_jsonl))
        sampled = _sample_with_replacement(source_records, quota, rng)
        fused.extend(
            _annotate_record(
                record,
                source.name,
                source.template,
                source.poly_fallback,
            )
            for record in sampled
        )

    if shuffle:
        rng.shuffle(fused)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with output_path_obj.open("w", encoding="utf-8") as fout:
        for record in fused:
            fout.write(json.dumps(record, ensure_ascii=False))
            fout.write("\n")

    return output_path_obj


def prepare_record_for_dataset(
    record: Mapping[str, Any], spec: DatasetSpec
) -> Dict[str, Any]:
    return _annotate_record(
        record, spec.name, spec.template, spec.poly_fallback
    )


__all__ = [
    "DatasetSpec",
    "AuxiliarySpec",
    "FusionConfig",
    "build_fused_jsonl",
    "prepare_record_for_dataset",
]
