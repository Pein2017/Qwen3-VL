"""Dataset wrapper registry for fusion configs."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type, Literal

from ..fusion_types import DatasetSpec, FALLBACK_OPTIONS

DatasetDomain = Literal["target", "source"]


def _normalize_bool(value: Any, *, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in {0, 1, 0.0, 1.0}:
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean value")


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{field_name} must be a mapping")


def _parse_poly_max_points(value: Any, *, field_name: str) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _parse_poly_min_ratio(value: Any, *, field_name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a float in [0, 1]") from exc
    if not (0.0 <= parsed <= 1.0):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return parsed


def _parse_max_objects(value: Any, *, field_name: str) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


class DatasetWrapper(abc.ABC):
    """Base class for converting fusion config entries into DatasetSpec objects."""

    key: str
    default_name: str = "dataset"
    domain: DatasetDomain = "target"
    template_id: str = "bbu_dense"
    supports_augmentation: bool = True
    supports_curriculum: bool = True
    default_poly_fallback: Literal["off", "bbox_2d"] = "off"

    @classmethod
    def build_spec(
        cls,
        *,
        name: Optional[str],
        params: Mapping[str, Any],
    ) -> DatasetSpec:
        mapping = _as_mapping(params, field_name=f"{cls.__name__}.params")
        train_jsonl = mapping.get("train_jsonl")
        if not train_jsonl:
            raise ValueError(
                f"{cls.__name__} requires params.train_jsonl to be set"
            )
        val_jsonl = mapping.get("val_jsonl")
        template_value = mapping.get("template")
        template = (
            str(template_value).strip() if template_value else cls.template_id
        )
        poly_fallback = mapping.get("poly_fallback", cls.default_poly_fallback)
        if poly_fallback not in FALLBACK_OPTIONS:
            raise ValueError(
                f"{cls.__name__}.poly_fallback must be one of {FALLBACK_OPTIONS}"
            )
        poly_max_points = _parse_poly_max_points(
            mapping.get("poly_max_points"),
            field_name=f"{cls.__name__}.poly_max_points",
        )
        poly_min_ratio = _parse_poly_min_ratio(
            mapping.get("poly_min_ratio"),
            field_name=f"{cls.__name__}.poly_min_ratio",
        )
        max_objects = _parse_max_objects(
            mapping.get("max_objects_per_image"),
            field_name=f"{cls.__name__}.max_objects_per_image",
        )
        aug_flag = _normalize_bool(
            mapping.get("augmentation_enabled"),
            field_name=f"{cls.__name__}.augmentation_enabled",
            default=cls.supports_augmentation,
        )
        curriculum_flag = _normalize_bool(
            mapping.get("curriculum_enabled"),
            field_name=f"{cls.__name__}.curriculum_enabled",
            default=cls.supports_curriculum,
        )
        resolved_name = name or cls.default_name
        return DatasetSpec(
            key=cls.key,
            name=str(resolved_name),
            train_jsonl=Path(str(train_jsonl)),
            template=template,
            domain=cls.domain,
            supports_augmentation=aug_flag,
            supports_curriculum=curriculum_flag,
            poly_fallback=poly_fallback,  # type: ignore[arg-type]
            poly_max_points=poly_max_points,
            poly_min_ratio=poly_min_ratio,
            max_objects_per_image=max_objects,
            val_jsonl=Path(str(val_jsonl)) if val_jsonl else None,
        )


_WRAPPERS: Dict[str, Type[DatasetWrapper]] = {}


def register_dataset_wrapper(name: str):
    """Decorator to register wrapper classes by config key."""

    def decorator(cls: Type[DatasetWrapper]) -> Type[DatasetWrapper]:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("Wrapper name must be a non-empty string")
        cls.key = key
        _WRAPPERS[key] = cls
        return cls

    return decorator


def resolve_dataset_wrapper(dataset_key: str) -> Type[DatasetWrapper]:
    key = str(dataset_key).strip().lower()
    if not key:
        raise ValueError("dataset key must be provided")
    try:
        return _WRAPPERS[key]
    except KeyError as exc:
        raise KeyError(f"No dataset wrapper registered for '{dataset_key}'") from exc


def build_dataset_spec(
    dataset_key: str,
    *,
    name: Optional[str],
    params: Mapping[str, Any],
) -> DatasetSpec:
    wrapper_cls = resolve_dataset_wrapper(dataset_key)
    return wrapper_cls.build_spec(name=name, params=params)


class TargetDatasetWrapper(DatasetWrapper):
    domain: DatasetDomain = "target"
    template_id = "bbu_dense"
    supports_augmentation = True
    supports_curriculum = True


class PublicDetectionDatasetWrapper(DatasetWrapper):
    domain: DatasetDomain = "source"
    template_id = "aux_dense"
    supports_augmentation = False
    supports_curriculum = False


@register_dataset_wrapper("bbu")
class BbuDatasetWrapper(TargetDatasetWrapper):
    default_name = "bbu"


@register_dataset_wrapper("rru")
class RruDatasetWrapper(TargetDatasetWrapper):
    default_name = "rru"


@register_dataset_wrapper("coco")
class CocoDatasetWrapper(PublicDetectionDatasetWrapper):
    default_name = "coco"


@register_dataset_wrapper("objects365")
class Objects365DatasetWrapper(PublicDetectionDatasetWrapper):
    default_name = "objects365"


@register_dataset_wrapper("lvis")
class LvisDatasetWrapper(PublicDetectionDatasetWrapper):
    default_name = "lvis"


@register_dataset_wrapper("flickr3k")
class FlickrDatasetWrapper(PublicDetectionDatasetWrapper):
    default_name = "flickr3k"


__all__ = [
    "DatasetWrapper",
    "register_dataset_wrapper",
    "resolve_dataset_wrapper",
    "build_dataset_spec",
]
