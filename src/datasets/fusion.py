"""Fusion helpers for multi-dataset dense-caption training."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Literal

from .fusion_types import AuxiliarySpec, DatasetSpec
from .geometry import points_to_xyxy
from .wrappers import build_dataset_spec
from .utils import load_jsonl


@dataclass(frozen=True)
class FusionConfig:
    target: DatasetSpec
    sources: Tuple[AuxiliarySpec, ...]

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

        target_spec, _ = cls._parse_dataset_entry(
            payload.get("target"), require_ratio=False
        )

        source_sections = payload.get("sources") or []
        if not isinstance(source_sections, Iterable):
            raise ValueError("fusion sources must be an iterable")

        sources: list[AuxiliarySpec] = []
        for entry in source_sections:
            spec, ratio = cls._parse_dataset_entry(entry, require_ratio=True)
            sources.append(cls._as_auxiliary(spec, ratio or 0.0))

        return cls(
            target=target_spec,
            sources=tuple(sources),
        )

    @staticmethod
    def _parse_dataset_entry(
        entry: Any, *, require_ratio: bool
    ) -> tuple[DatasetSpec, Optional[float]]:
        if not isinstance(entry, Mapping):
            raise ValueError("dataset entry must be a mapping")

        dataset_key = entry.get("dataset")
        name_override = entry.get("name") if dataset_key else None
        if dataset_key is None:
            dataset_key = entry.get("name")
            if dataset_key is None:
                raise ValueError("dataset entry must include 'dataset' or 'name'")
            dataset_key = str(dataset_key)
            name_override = str(entry.get("name")) if entry.get("name") else None

        params = entry.get("params")
        if params is None:
            params = {
                "train_jsonl": entry.get("train_jsonl"),
                "val_jsonl": entry.get("val_jsonl"),
                "template": entry.get("template"),
                "poly_fallback": entry.get("poly_fallback"),
                "poly_max_points": entry.get("poly_max_points"),
                "poly_min_ratio": entry.get("poly_min_ratio"),
                "augmentation_enabled": entry.get("augmentation_enabled"),
                "curriculum_enabled": entry.get("curriculum_enabled"),
                "max_objects_per_image": entry.get("max_objects_per_image"),
            }
        elif not isinstance(params, Mapping):
            raise TypeError("dataset params must be a mapping if provided")

        spec = build_dataset_spec(dataset_key, name=name_override, params=params)
        ratio_value: Optional[float] = None
        if require_ratio:
            ratio = entry.get("ratio")
            if ratio is None:
                raise ValueError("auxiliary spec must include 'ratio'")
            try:
                ratio_value = float(ratio)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError("ratio must be numeric") from exc
            if ratio_value < 0:
                raise ValueError("ratio must be non-negative")
        return spec, ratio_value

    @staticmethod
    def _as_auxiliary(spec: DatasetSpec, ratio: float) -> AuxiliarySpec:
        return AuxiliarySpec(
            key=spec.key,
            name=spec.name,
            train_jsonl=spec.train_jsonl,
            template=spec.template,
            domain=spec.domain,
            supports_augmentation=spec.supports_augmentation,
            supports_curriculum=spec.supports_curriculum,
            poly_fallback=spec.poly_fallback,
            poly_max_points=spec.poly_max_points,
            poly_min_ratio=spec.poly_min_ratio,
            max_objects_per_image=spec.max_objects_per_image,
            val_jsonl=spec.val_jsonl,
            ratio=ratio,
        )


def _flatten_poly_points(points: Any) -> Optional[list[float]]:
    if isinstance(points, (list, tuple)):
        if points and isinstance(points[0], (list, tuple)):
            flat: list[float] = []
            for pair in points:  # type: ignore[assignment]
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    return None
                flat.extend([float(pair[0]), float(pair[1])])
            return flat
        return [float(v) for v in points]  # type: ignore[arg-type]
    return None


def _apply_poly_policy(
    record: Dict[str, Any],
    fallback: Literal["off", "bbox_2d"],
    poly_max_points: Optional[int],
) -> None:
    objects = record.get("objects") or []
    for obj in objects:
        if "poly" not in obj:
            continue
        raw_pts = obj.get("poly")
        pts = _flatten_poly_points(raw_pts)
        if not pts:
            continue
        point_count = len(pts) // 2
        fallback_all = fallback == "bbox_2d"
        fallback_large = (
            poly_max_points is not None and point_count > poly_max_points
        )
        if fallback_all or fallback_large:
            obj["bbox_2d"] = points_to_xyxy(pts)
            obj.pop("poly", None)
            obj.pop("poly_points", None)


def _annotate_record(
    record: Mapping[str, Any],
    fallback: Literal["off", "bbox_2d"],
    poly_max_points: Optional[int],
) -> Dict[str, Any]:
    annotated = copy.deepcopy(record)
    _apply_poly_policy(annotated, fallback, poly_max_points)
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
    target_records = load_jsonl(str(config.target.train_jsonl), resolve_relative=True)
    rng = random.Random(seed)

    fused: list[Dict[str, Any]] = []
    for record in target_records:
        fused.append(
            _annotate_record(
                record,
                config.target.poly_fallback,
                config.target.poly_max_points,
            )
        )

    for source in config.sources:
        quota = round(source.ratio * len(target_records))
        if quota <= 0:
            continue
        source_records = load_jsonl(str(source.train_jsonl), resolve_relative=True)
        sampled = _sample_with_replacement(source_records, quota, rng)
        fused.extend(
            _annotate_record(
                record,
                source.poly_fallback,
                source.poly_max_points,
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
    return _annotate_record(record, spec.poly_fallback, spec.poly_max_points)


__all__ = [
    "DatasetSpec",
    "AuxiliarySpec",
    "FusionConfig",
    "build_fused_jsonl",
    "prepare_record_for_dataset",
]
