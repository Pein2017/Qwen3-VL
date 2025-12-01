"""Fusion helpers for multi-dataset dense-caption training."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .fusion_types import AuxiliarySpec, DatasetSpec, TargetSpec
from .wrappers import build_dataset_spec
from .utils import load_jsonl


@dataclass(frozen=True)
class FusionConfig:
    targets: Tuple[TargetSpec, ...]
    sources: Tuple[AuxiliarySpec, ...]

    @property
    def target(self) -> TargetSpec:
        """Backward compatibility: first target."""

        return self.targets[0]

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

        targets_section = payload.get("targets")
        if targets_section is None:
            legacy_target = payload.get("target")
            targets_section = [legacy_target] if legacy_target is not None else []
        if not isinstance(targets_section, Iterable) or isinstance(
            targets_section, (str, bytes)
        ):
            raise ValueError("fusion targets must be an iterable")

        target_specs: list[TargetSpec] = []
        for entry in targets_section:
            spec, ratio = cls._parse_dataset_entry(
                entry, require_ratio=False, allow_ratio=True
            )
            target_specs.append(cls._as_target(spec, ratio))

        if not target_specs:
            raise ValueError("fusion config requires at least one target")

        source_sections = payload.get("sources") or []
        if not isinstance(source_sections, Iterable):
            raise ValueError("fusion sources must be an iterable")

        sources: list[AuxiliarySpec] = []
        for entry in source_sections:
            spec, ratio = cls._parse_dataset_entry(entry, require_ratio=True)
            sources.append(cls._as_auxiliary(spec, ratio or 0.0))

        cls._validate_unique_names(target_specs, sources)

        return cls(
            targets=tuple(target_specs),
            sources=tuple(sources),
        )

    @staticmethod
    def _parse_dataset_entry(
        entry: Any, *, require_ratio: bool, allow_ratio: bool = False
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
            }
            if "val_jsonl" in entry:
                params["val_jsonl"] = entry.get("val_jsonl")
            if "template" in entry:
                params["template"] = entry.get("template")
            if "poly_fallback" in entry:
                params["poly_fallback"] = entry.get("poly_fallback")
            if "poly_max_points" in entry:
                params["poly_max_points"] = entry.get("poly_max_points")
            if "augmentation_enabled" in entry:
                params["augmentation_enabled"] = entry.get("augmentation_enabled")
            if "curriculum_enabled" in entry:
                params["curriculum_enabled"] = entry.get("curriculum_enabled")
            if "max_objects_per_image" in entry:
                params["max_objects_per_image"] = entry.get("max_objects_per_image")
            if "user_prompt" in entry:
                params["user_prompt"] = entry.get("user_prompt")
            if "system_prompt" in entry:
                params["system_prompt"] = entry.get("system_prompt")
            if "seed" in entry:
                params["seed"] = entry.get("seed")
        elif not isinstance(params, Mapping):
            raise TypeError("dataset params must be a mapping if provided")

        spec = build_dataset_spec(dataset_key, name=name_override, params=params)
        ratio_value: Optional[float] = None
        if require_ratio or allow_ratio:
            ratio = entry.get("ratio")
            if require_ratio and ratio is None:
                raise ValueError("auxiliary spec must include 'ratio'")
            if ratio is not None:
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
            max_objects_per_image=spec.max_objects_per_image,
            val_jsonl=spec.val_jsonl,
            prompt_user=spec.prompt_user,
            prompt_system=spec.prompt_system,
            seed=spec.seed,
            ratio=ratio,
        )

    @staticmethod
    def _as_target(spec: DatasetSpec, ratio: Optional[float]) -> TargetSpec:
        return TargetSpec(
            key=spec.key,
            name=spec.name,
            train_jsonl=spec.train_jsonl,
            template=spec.template,
            domain=spec.domain,
            supports_augmentation=spec.supports_augmentation,
            supports_curriculum=spec.supports_curriculum,
            poly_fallback=spec.poly_fallback,
            poly_max_points=spec.poly_max_points,
            max_objects_per_image=spec.max_objects_per_image,
            val_jsonl=spec.val_jsonl,
            prompt_user=spec.prompt_user,
            prompt_system=spec.prompt_system,
            seed=spec.seed,
            ratio=ratio,
        )

    @staticmethod
    def _validate_unique_names(
        targets: Sequence[TargetSpec], sources: Sequence[AuxiliarySpec]
    ) -> None:
        names: set[str] = set()
        for spec in list(targets) + list(sources):
            if spec.name in names:
                raise ValueError(f"Duplicate dataset name in fusion config: {spec.name}")
            names.add(spec.name)


def _compute_target_quotas(
    targets: Sequence[TargetSpec], pool_sizes: Mapping[str, int]
) -> tuple[dict[str, int], Optional[int]]:
    """Compute per-target quotas using ratio balancing.

    Returns (quota_map, base). base is None when no ratios are provided.
    """

    has_ratio = any(t.ratio is not None for t in targets)
    quotas: dict[str, int] = {}
    if not has_ratio:
        for spec in targets:
            quotas[spec.name] = pool_sizes.get(spec.name, 0)
        return quotas, None

    capacities: list[float] = []
    ratios: dict[str, float] = {}
    for spec in targets:
        ratio_val = spec.ratio if spec.ratio is not None else 1.0
        ratios[spec.name] = ratio_val
        length = pool_sizes.get(spec.name, 0)
        if length <= 0:
            raise ValueError(f"Target '{spec.name}' has empty pool for ratio balancing")
        capacities.append(length / ratio_val)

    base = math.floor(min(capacities))
    if base <= 0:
        raise ValueError("Computed base quota is non-positive; check target ratios and sizes")

    for spec in targets:
        ratio_val = ratios[spec.name]
        pool_len = pool_sizes.get(spec.name, 0)
        quota = round(base * ratio_val)
        quota = max(1, min(pool_len, quota))
        quotas[spec.name] = quota
    return quotas, base


def _annotate_record(record: Mapping[str, Any], spec: DatasetSpec) -> Dict[str, Any]:
    annotated = copy.deepcopy(record)
    metadata = annotated.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["_fusion_source"] = spec.name
    metadata["_fusion_template"] = spec.template
    metadata["_fusion_domain"] = spec.domain
    annotated["metadata"] = metadata
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
    rng = random.Random(seed)

    # Load target records and compute quotas (respect ratios when provided).
    target_pools: dict[str, list[Dict[str, Any]]] = {}
    pool_sizes: dict[str, int] = {}
    for target in config.targets:
        records = load_jsonl(str(target.train_jsonl), resolve_relative=True)
        target_pools[target.name] = records
        pool_sizes[target.name] = len(records)

    target_quotas, _ = _compute_target_quotas(config.targets, pool_sizes)

    fused: list[Dict[str, Any]] = []
    for target in config.targets:
        pool = target_pools[target.name]
        quota = target_quotas.get(target.name, 0)
        if quota <= 0:
            continue
        indices = list(range(len(pool)))
        rng_local = random.Random((seed ^ hash(target.name)) & 0xFFFFFFFF)
        if len(indices) > 1:
            rng_local.shuffle(indices)
        for idx in indices[:quota]:
            fused.append(_annotate_record(pool[idx], target))

    total_target = len([rec for rec in fused if rec.get("metadata", {}).get("_fusion_domain") == "target"])

    for source in config.sources:
        quota = round(source.ratio * total_target)
        if quota <= 0:
            continue
        source_records = load_jsonl(str(source.train_jsonl), resolve_relative=True)
        source_seed = (
            seed if source.seed is None else (seed ^ int(source.seed)) & 0xFFFFFFFF
        )
        sampled = _sample_with_replacement(
            source_records, quota, random.Random(source_seed)
        )
        fused.extend(_annotate_record(record, source) for record in sampled)

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
    return _annotate_record(record, spec)


__all__ = [
    "DatasetSpec",
    "TargetSpec",
    "AuxiliarySpec",
    "FusionConfig",
    "build_fused_jsonl",
    "prepare_record_for_dataset",
    "_compute_target_quotas",
]
