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

from .contracts import (
    MessageContent,
    MessageDict,
    ConversationRecord,
    GeometryDict,
    AugmentationTelemetry,
    validate_conversation_record,
    validate_geometry_sequence,
)
from .dense_caption import DenseCaptionDataset
from .utils import load_jsonl, extract_object_points, extract_geometry

# AugmentationConfig removed with v1 API; use Compose pipelines directly
from .augmentation.base import Compose
from .augmentation.registry import register, get, available
from .builders import (
    BaseBuilder,
    JSONLinesBuilder,
)
from .preprocessors import (
    BasePreprocessor,
    DenseCaptionPreprocessor,
    AugmentationPreprocessor,
)

__all__ = [
    # Primary dataset
    "DenseCaptionDataset",
    # Utilities
    "load_jsonl",
    "extract_object_points",
    "extract_geometry",
    # Contracts
    "MessageContent",
    "MessageDict",
    "ConversationRecord",
    "GeometryDict",
    "AugmentationTelemetry",
    "validate_conversation_record",
    "validate_geometry_sequence",
    # Augmentation
    "Compose",
    "register",
    "get",
    "available",
    # Builders
    "BaseBuilder",
    "JSONLinesBuilder",
    # Preprocessors
    "BasePreprocessor",
    "DenseCaptionPreprocessor",
    "AugmentationPreprocessor",
]
