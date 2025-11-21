"""Shared dataclasses for fusion datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

FALLBACK_OPTIONS = ("off", "bbox_2d")


@dataclass(frozen=True)
class DatasetSpec:
    """Normalized dataset description used by fusion builders."""

    key: str
    name: str
    train_jsonl: Path
    template: str
    domain: Literal["target", "source"]
    supports_augmentation: bool
    supports_curriculum: bool
    poly_fallback: Literal["off", "bbox_2d"] = "off"
    poly_max_points: Optional[int] = None
    val_jsonl: Optional[Path] = None
    # Optional per-source control: ensure at least this fraction of auxiliary
    # picks per epoch contain polygons after fallback. None = no constraint.
    poly_min_ratio: Optional[float] = None
    # Optional object cap applied at load/epoch time (random downsample of objects).
    max_objects_per_image: Optional[int] = None


@dataclass(frozen=True)
class AuxiliarySpec(DatasetSpec):
    ratio: float = 0.0


__all__ = ["DatasetSpec", "AuxiliarySpec", "FALLBACK_OPTIONS"]
