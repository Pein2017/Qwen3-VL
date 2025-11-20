"""
Modernized Qwen3-VL data conversion pipeline package.

This package groups the core processing modules (config, geometry, taxonomy,
validation, and the unified processor) under ``data_conversion.pipeline`` so
callers can import everything from a single namespace.
"""

from .config import DataConversionConfig, setup_logging, validate_config
from .constants import DEFAULT_LABEL_HIERARCHY, OBJECT_TYPES
from .coordinate_manager import CoordinateManager
from .data_splitter import DataSplitter
from .flexible_taxonomy_processor import HierarchicalProcessor
from .format_converter import FormatConverter
from .summary_builder import build_summary_from_objects
from .unified_processor import UnifiedProcessor
from .validation_manager import StructureValidator, ValidationManager
from .vision_process import ImageProcessor, smart_resize

__all__ = [
    "CoordinateManager",
    "DataConversionConfig",
    "DataSplitter",
    "DEFAULT_LABEL_HIERARCHY",
    "FormatConverter",
    "HierarchicalProcessor",
    "ImageProcessor",
    "OBJECT_TYPES",
    "StructureValidator",
    "UnifiedProcessor",
    "ValidationManager",
    "build_summary_from_objects",
    "setup_logging",
    "smart_resize",
    "validate_config",
]
