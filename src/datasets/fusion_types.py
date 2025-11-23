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
    # Optional object cap applied at load/epoch time (random downsample of objects).
    max_objects_per_image: Optional[int] = None
    # Optional prompt overrides (per-dataset) applied on top of domain/default prompts.
    prompt_user: Optional[str] = None
    prompt_system: Optional[str] = None
    # Optional deterministic seed for per-epoch sampling/shuffling.
    seed: Optional[int] = None


@dataclass(frozen=True)
class AuxiliarySpec(DatasetSpec):
    ratio: float = 0.0


__all__ = ["DatasetSpec", "AuxiliarySpec", "FALLBACK_OPTIONS"]
