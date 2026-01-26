"""Datasets package for dense captioning.

This package provides a modular, ms-swift-aligned architecture for training
dense caption models with configurable builders, preprocessors, and augmentation.

Submodules:
- utils: shared utilities (load_jsonl, extract_geometry, etc.)
- geometry: geometry ops (scaling, affine transformations)
- augment: image + geometry augmentations
- collators: data collation utilities
- preprocessors: row-level transformations (ms-swift style)
- builders: message builders for different output formats
- dense_caption: high-level dataset class for dense captioning

Main exports:
- DenseCaptionDataset: Primary dataset class
- Builders: JSONLinesBuilder
- Preprocessors: DenseCaptionPreprocessor, AugmentationPreprocessor
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Primary datasets
    "BaseCaptionDataset": ("src.datasets.dense_caption", "BaseCaptionDataset"),
    "DenseCaptionDataset": ("src.datasets.dense_caption", "DenseCaptionDataset"),
    "FusionCaptionDataset": (
        "src.datasets.unified_fusion_dataset",
        "FusionCaptionDataset",
    ),
    "UnifiedFusionDataset": (
        "src.datasets.unified_fusion_dataset",
        "UnifiedFusionDataset",
    ),
    # Utilities
    "load_jsonl": ("src.datasets.utils", "load_jsonl"),
    "extract_object_points": ("src.datasets.utils", "extract_object_points"),
    "extract_geometry": ("src.datasets.utils", "extract_geometry"),
    # Contracts
    "MessageContent": ("src.datasets.contracts", "MessageContent"),
    "MessageDict": ("src.datasets.contracts", "MessageDict"),
    "ConversationRecord": ("src.datasets.contracts", "ConversationRecord"),
    "DatasetImage": ("src.datasets.contracts", "DatasetImage"),
    "DatasetObject": ("src.datasets.contracts", "DatasetObject"),
    "GeometryDict": ("src.datasets.contracts", "GeometryDict"),
    "AugmentationTelemetry": ("src.datasets.contracts", "AugmentationTelemetry"),
    "validate_conversation_record": (
        "src.datasets.contracts",
        "validate_conversation_record",
    ),
    "validate_geometry_sequence": (
        "src.datasets.contracts",
        "validate_geometry_sequence",
    ),
    # Augmentation
    "Compose": ("src.datasets.augmentation.base", "Compose"),
    "register": ("src.datasets.augmentation.registry", "register"),
    "get": ("src.datasets.augmentation.registry", "get"),
    "available": ("src.datasets.augmentation.registry", "available"),
    # Builders
    "BaseBuilder": ("src.datasets.builders", "BaseBuilder"),
    "JSONLinesBuilder": ("src.datasets.builders", "JSONLinesBuilder"),
    # Fusion helpers
    "FusionConfig": ("src.datasets.fusion", "FusionConfig"),
    "build_fused_jsonl": ("src.datasets.fusion", "build_fused_jsonl"),
    # Preprocessors
    "BasePreprocessor": ("src.datasets.preprocessors", "BasePreprocessor"),
    "DenseCaptionPreprocessor": (
        "src.datasets.preprocessors",
        "DenseCaptionPreprocessor",
    ),
    "AugmentationPreprocessor": (
        "src.datasets.preprocessors",
        "AugmentationPreprocessor",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value  # cache
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))


__all__ = list(_LAZY_ATTRS.keys())
