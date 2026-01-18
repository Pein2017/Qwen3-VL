"""Fusion helpers for multi-dataset dense-caption training."""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from src.utils import get_logger, require_mutable_mapping
from src.utils.unstructured import UnstructuredMapping, UnstructuredMutableMapping

from .contracts import ConversationRecord
from .fusion_types import AuxiliarySpec, DatasetSpec, TargetSpec
from .wrappers import build_dataset_spec
from .utils import load_jsonl

FusionPayload = UnstructuredMutableMapping
FusionEntry = UnstructuredMapping


def _normalize_extends(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _load_fusion_payload(path: Path) -> FusionPayload:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
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
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("fusion config must be a mapping")
    return require_mutable_mapping(payload, context="fusion.config")


def _merge_dicts(base: FusionEntry, override: FusionEntry) -> FusionPayload:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            merged[key] = _merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _dataset_entry_key(entry: FusionEntry, *, field_name: str) -> str:
    name = entry.get("name") or entry.get("dataset")
    if name is None:
        raise ValueError(
            f"{field_name} entry must include 'name' or 'dataset' for fusion merge"
        )
    return str(name)


def _ensure_entry_list(value: object, *, field_name: str) -> list[FusionEntry]:
    if value is None:
        return []
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError(f"fusion {field_name} must be an iterable")
    entries = list(value)
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{field_name} entry must be a mapping")
    return entries


def _merge_dataset_entries(
    base_value: object, override_value: object, *, field_name: str
) -> list[FusionEntry]:
    if isinstance(override_value, list) and not override_value:
        return []

    base_entries = _ensure_entry_list(base_value, field_name=field_name)
    override_entries = _ensure_entry_list(override_value, field_name=field_name)

    if not base_entries:
        return override_entries

    base_map: dict[str, FusionEntry] = {}
    for entry in base_entries:
        key = _dataset_entry_key(entry, field_name=field_name)
        if key in base_map:
            raise ValueError(
                f"Duplicate dataset name in base fusion config for {field_name}: {key}"
            )
        base_map[key] = entry

    override_map: dict[str, FusionEntry] = {}
    for entry in override_entries:
        key = _dataset_entry_key(entry, field_name=field_name)
        if key in override_map:
            raise ValueError(
                f"Duplicate dataset name in override fusion config for {field_name}: {key}"
            )
        override_map[key] = entry

    merged_entries: list[FusionEntry] = []
    for entry in base_entries:
        key = _dataset_entry_key(entry, field_name=field_name)
        if key in override_map:
            merged_entries.append(_merge_dicts(entry, override_map[key]))
        else:
            merged_entries.append(entry)

    for entry in override_entries:
        key = _dataset_entry_key(entry, field_name=field_name)
        if key not in base_map:
            merged_entries.append(entry)

    return merged_entries


def _merge_fusion_payload(base: FusionEntry, override: FusionEntry) -> FusionPayload:
    merged = dict(base)
    for key, value in override.items():
        if key in {"targets", "sources"}:
            merged[key] = _merge_dataset_entries(merged.get(key), value, field_name=key)
            continue
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            merged[key] = _merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _load_fusion_with_extends(
    path: Path, visited: set[Path] | None = None
) -> FusionPayload:
    abs_path = path.resolve()
    visited = set() if visited is None else visited
    if abs_path in visited:
        raise ValueError(f"Cyclic fusion config inheritance detected at: {abs_path}")
    visited.add(abs_path)

    payload = _load_fusion_payload(abs_path)
    if "inherit" in payload:
        raise ValueError(
            "Fusion config inheritance uses 'extends'; 'inherit' is not supported."
        )
    extends_value = payload.pop("extends", None)

    merged_base: FusionPayload = {}
    for base_ref in _normalize_extends(extends_value):
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = (abs_path.parent / base_path).resolve()
        base_payload = _load_fusion_with_extends(base_path, visited)
        merged_base = _merge_fusion_payload(merged_base, base_payload)

    return _merge_fusion_payload(merged_base, payload)


@dataclass(frozen=True)
class FusionConfig:
    targets: tuple[TargetSpec, ...]
    sources: tuple[AuxiliarySpec, ...]

    @property
    def target(self) -> TargetSpec:
        """Convenience alias for the first target."""

        return self.targets[0]

    @classmethod
    def from_file(cls, path: str) -> "FusionConfig":
        path_obj = Path(path)
        payload = _load_fusion_with_extends(path_obj)
        if not isinstance(payload, Mapping):
            raise ValueError("fusion config must be a mapping")

        targets_section = payload.get("targets")
        if targets_section is None:
            if "target" in payload:
                raise ValueError(
                    "fusion config must define 'targets' (list); 'target' is not supported."
                )
            targets_section = []
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
    ) -> tuple[DatasetSpec, float | None]:
        if not isinstance(entry, Mapping):
            raise ValueError("dataset entry must be a mapping")

        dataset_key_raw = entry.get("dataset")
        if dataset_key_raw is None:
            dataset_key_raw = entry.get("name")
            if dataset_key_raw is None:
                raise ValueError("dataset entry must include 'dataset' or 'name'")
        dataset_key = str(dataset_key_raw)
        name_override_raw = entry.get("name")
        name_override = (
            str(name_override_raw) if name_override_raw is not None else None
        )
        params = entry.get("params")
        if "summary_label_grouping" in entry:
            raise ValueError(
                "summary_label_grouping has been removed; delete it from fusion configs."
            )
        if params is None:
            params = {
                "train_jsonl": entry.get("train_jsonl"),
            }
            if "val_jsonl" in entry:
                params["val_jsonl"] = entry.get("val_jsonl")
            if "template" in entry:
                params["template"] = entry.get("template")
            if "mode" in entry:
                params["mode"] = entry.get("mode")
            if "use_summary" in entry:
                params["use_summary"] = entry.get("use_summary")
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
            if "sample_without_replacement" in entry:
                params["sample_without_replacement"] = entry.get(
                    "sample_without_replacement"
                )
        elif not isinstance(params, Mapping):
            raise TypeError("dataset params must be a mapping if provided")
        if "summary_label_grouping" in params:
            raise ValueError(
                "summary_label_grouping has been removed; delete it from fusion configs."
            )

        spec = build_dataset_spec(dataset_key, name=name_override, params=params)
        ratio_value: float | None = None
        if require_ratio or allow_ratio:
            ratio = entry.get("ratio")
            if require_ratio and ratio is None:
                raise ValueError("auxiliary spec must include 'ratio'")
            if ratio is not None:
                if not isinstance(ratio, (int, float, str)):
                    raise ValueError("ratio must be numeric")
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
            mode=spec.mode,
            supports_augmentation=spec.supports_augmentation,
            supports_curriculum=spec.supports_curriculum,
            poly_fallback=spec.poly_fallback,
            poly_max_points=spec.poly_max_points,
            max_objects_per_image=spec.max_objects_per_image,
            val_jsonl=spec.val_jsonl,
            prompt_user=spec.prompt_user,
            prompt_system=spec.prompt_system,
            seed=spec.seed,
            sample_without_replacement=spec.sample_without_replacement,
            ratio=ratio,
        )

    @staticmethod
    def _as_target(spec: DatasetSpec, ratio: float | None) -> TargetSpec:
        return TargetSpec(
            key=spec.key,
            name=spec.name,
            train_jsonl=spec.train_jsonl,
            template=spec.template,
            domain=spec.domain,
            mode=spec.mode,
            supports_augmentation=spec.supports_augmentation,
            supports_curriculum=spec.supports_curriculum,
            poly_fallback=spec.poly_fallback,
            poly_max_points=spec.poly_max_points,
            max_objects_per_image=spec.max_objects_per_image,
            val_jsonl=spec.val_jsonl,
            prompt_user=spec.prompt_user,
            prompt_system=spec.prompt_system,
            seed=spec.seed,
            sample_without_replacement=spec.sample_without_replacement,
            ratio=ratio,
        )

    @staticmethod
    def _validate_unique_names(
        targets: Sequence[TargetSpec], sources: Sequence[AuxiliarySpec]
    ) -> None:
        names: set[str] = set()
        for spec in list(targets) + list(sources):
            if spec.name in names:
                raise ValueError(
                    f"Duplicate dataset name in fusion config: {spec.name}"
                )
            names.add(spec.name)


def _compute_target_quotas(
    targets: Sequence[TargetSpec], pool_sizes: Mapping[str, int]
) -> tuple[dict[str, int], int | None]:
    """Compute per-target quotas using self-scaled ratios.

    Returns (quota_map, base). base is kept for legacy compatibility and is
    always None in the self-scaled regime.
    """

    quotas: dict[str, int] = {}
    for spec in targets:
        ratio_val = spec.ratio if spec.ratio is not None else 1.0
        if ratio_val < 0:
            raise ValueError("ratio must be non-negative")
        pool_len = pool_sizes.get(spec.name, 0)
        if pool_len <= 0:
            quotas[spec.name] = 0
            continue
        quota = round(pool_len * ratio_val)
        quotas[spec.name] = quota
    return quotas, None


def _annotate_record(
    record: ConversationRecord, spec: DatasetSpec
) -> ConversationRecord:
    annotated = cast(ConversationRecord, dict(copy.deepcopy(record)))
    metadata = annotated.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["_fusion_source"] = spec.name
    metadata["_fusion_template"] = spec.template
    metadata["_fusion_domain"] = spec.domain
    annotated["metadata"] = metadata
    return annotated


def _sample_with_replacement(
    records: Sequence[ConversationRecord],
    count: int,
    rng: random.Random,
) -> list[ConversationRecord]:
    if not records or count <= 0:
        return []
    samples: list[ConversationRecord] = []
    for _ in range(count):
        choice = rng.choice(records)
        samples.append(cast(ConversationRecord, dict(copy.deepcopy(choice))))
    return samples


def _sample_indices(
    pool_len: int,
    quota: int,
    rng: random.Random,
    *,
    sample_without_replacement: bool,
) -> tuple[list[int], bool]:
    """Return indices and whether we fell back to replacement.

    Fallback occurs when without-replacement is requested but quota exceeds pool.
    """

    if pool_len <= 0 or quota <= 0:
        return [], False

    if sample_without_replacement and quota <= pool_len:
        indices = list(range(pool_len))
        if pool_len > 1:
            rng.shuffle(indices)
        return indices[:quota], False

    fallback = sample_without_replacement and quota > pool_len
    indices = [rng.randrange(pool_len) for _ in range(quota)]
    return indices, fallback


def build_fused_jsonl(
    config: FusionConfig,
    output_path: str,
    *,
    seed: int = 2025,
    shuffle: bool = True,
) -> Path:
    rng = random.Random(seed)
    logger = get_logger(__name__)

    # Load target records and compute quotas (respect ratios when provided).
    target_pools: dict[str, list[ConversationRecord]] = {}
    pool_sizes: dict[str, int] = {}
    for target in config.targets:
        records = load_jsonl(str(target.train_jsonl), resolve_relative=True)
        target_pools[target.name] = records
        pool_sizes[target.name] = len(records)

    target_quotas, _ = _compute_target_quotas(config.targets, pool_sizes)

    fused: list[ConversationRecord] = []
    for target in config.targets:
        pool = target_pools[target.name]
        quota = target_quotas.get(target.name, 0)
        if quota <= 0:
            continue
        rng_local = random.Random((seed ^ hash(target.name)) & 0xFFFFFFFF)
        indices = list(range(len(pool)))
        if len(indices) > 1:
            rng_local.shuffle(indices)
        if quota <= len(pool):
            sampled_indices = indices[:quota]
        else:
            sampled_indices = indices[:]
            extra_needed = quota - len(pool)
            sampled_indices.extend(
                rng_local.randrange(len(pool)) for _ in range(extra_needed)
            )
        for idx in sampled_indices:
            fused.append(_annotate_record(pool[idx], target))

    total_target = len(
        [
            rec
            for rec in fused
            if rec.get("metadata", {}).get("_fusion_domain") == "target"
        ]
    )

    for source in config.sources:
        quota = round(source.ratio * total_target)
        if quota <= 0:
            continue
        source_records = load_jsonl(str(source.train_jsonl), resolve_relative=True)
        source_seed = (
            seed if source.seed is None else (seed ^ int(source.seed)) & 0xFFFFFFFF
        )
        rng_source = random.Random(source_seed)
        sampled_indices, fell_back = _sample_indices(
            len(source_records),
            quota,
            rng_source,
            sample_without_replacement=bool(source.sample_without_replacement),
        )
        sampled_records: list[ConversationRecord]
        if source.sample_without_replacement and not fell_back:
            sampled_records = [
                copy.deepcopy(source_records[i]) for i in sampled_indices
            ]
        else:
            sampled_records = _sample_with_replacement(
                source_records, quota, rng_source
            )
        if fell_back:
            try:
                logger.debug(
                    "fusion offline fallback to replacement",
                    extra={
                        "source": source.name,
                        "quota": quota,
                        "pool": len(source_records),
                    },
                )
            except Exception:
                pass
        fused.extend(_annotate_record(record, source) for record in sampled_records)

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
    record: ConversationRecord, spec: DatasetSpec
) -> ConversationRecord:
    return _annotate_record(record, spec)


__all__ = [
    "DatasetSpec",
    "TargetSpec",
    "AuxiliarySpec",
    "FusionConfig",
    "build_fused_jsonl",
    "prepare_record_for_dataset",
    "_compute_target_quotas",
    "_sample_indices",
]
