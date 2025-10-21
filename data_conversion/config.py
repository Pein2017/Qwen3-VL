#!/usr/bin/env python3
"""
Configuration Management for Data Conversion Pipeline

Provides structured, type-safe configuration with validation.
Replaces environment variables and complex command-line arguments.
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Configure UTF-8 encoding for stdout/stderr if supported
try:
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
except (AttributeError, TypeError):
    pass

try:
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")
except (AttributeError, TypeError):
    pass

logger = logging.getLogger(__name__)


@dataclass
class DataConversionConfig:
    """Configuration for the data conversion pipeline.

    All parameters are required and must be explicitly set.
    No default values to ensure manual configuration.
    """

    # Required fields (no defaults)
    input_dir: str
    output_dir: str
    object_types: List[
        str
    ]  # e.g., ["bbu", "label"] or ["fiber", "wire"] - REQUIRED, arbitrary combinations
    resize: bool  # True or False - REQUIRED
    val_ratio: float  # e.g., 0.1 for 10% validation - REQUIRED
    max_teachers: int  # e.g., 10 - REQUIRED
    seed: int  # e.g., 42 - REQUIRED

    # Optional fields (with defaults)
    dataset_name: Optional[str] = None  # Auto-detected from input_dir if not provided
    language: str = "chinese"  # Language for processing (chinese/english)
    response_types: List[str] = field(
        default_factory=list
    )  # Response types for processing
    output_image_dir: Optional[str] = None  # Output directory for images
    token_map_path: Optional[str] = None  # Path to token mapping file

    # Label hierarchy - OPTIONAL
    hierarchy_path: Optional[str] = None

    # Processing parameters - OPTIONAL WITH DEFAULTS
    log_level: str = "INFO"
    fail_fast: bool = True

    # Advanced processing options - OPTIONAL WITH DEFAULTS
    geometry_diversity_weight: float = (
        4.0  # Weight for geometry diversity in teacher selection
    )

    # Filtering options - OPTIONAL WITH DEFAULTS
    remove_occlusion_tokens: bool = False  # Drop tokens containing "遮挡" from desc
    sanitize_text: bool = True  # Apply text normalization/sanitization on descriptions
    standardize_label_desc: bool = True  # Standardize 标签/* to 标签/无法识别 when empty-like

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._setup_dataset_name()
        self._validate_paths()
        self._validate_parameters()

    def _setup_dataset_name(self) -> None:
        """Auto-detect dataset name from input directory if not provided."""
        if self.dataset_name is None:
            self.dataset_name = Path(self.input_dir).name
        logger.debug(f"Dataset name: {self.dataset_name}")

    def _validate_paths(self) -> None:
        """Validate and create necessary directories."""
        input_path = Path(self.input_dir)
        if not input_path.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_path}")

        # Create dataset-specific output directories
        dataset_output_path = self.get_dataset_output_dir()
        dataset_output_path.mkdir(parents=True, exist_ok=True)

        if self.resize:
            # Create images subdirectory within dataset directory
            image_output_path = self.get_dataset_image_dir()
            image_output_path.mkdir(parents=True, exist_ok=True)

    def _validate_parameters(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError(f"val_ratio must be between 0 and 1, got {self.val_ratio}")

        if self.max_teachers < 0:
            raise ValueError(
                f"max_teachers must be non-negative, got {self.max_teachers}"
            )

        if not self.object_types:
            raise ValueError("object_types cannot be empty")

        # Handle 'full' keyword for all object types
        if len(self.object_types) == 1 and self.object_types[0] == "full":
            self.object_types = [
                "bbu",
                "bbu_shield",
                "label",
                "fiber",
                "wire",
                "connect_point",
            ]
            logger.info(
                "Using 'full' object types: all 6 object types will be processed"
            )
        else:
            valid_object_types = {
                "bbu",
                "bbu_shield",
                "label",
                "fiber",
                "wire",
                "connect_point",
            }
            for obj_type in self.object_types:
                if obj_type not in valid_object_types:
                    raise ValueError(
                        f"Invalid object type: {obj_type}. Valid types: {valid_object_types} or 'full'"
                    )

    def get_dataset_output_dir(self) -> Path:
        """Get the dataset-specific output directory path."""
        if self.dataset_name is None:
            raise ValueError(
                "dataset_name must be set before getting dataset output directory"
            )
        return Path(self.output_dir) / self.dataset_name

    def get_dataset_image_dir(self) -> Path:
        """Get the dataset-specific image output directory path."""
        return self.get_dataset_output_dir() / "images"

    def to_dict(self) -> Dict:
        """Convert config to dictionary for compatibility."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "object_types": self.object_types,
            "resize": self.resize,
            "val_ratio": self.val_ratio,
            "max_teachers": self.max_teachers,
            "seed": self.seed,
            "hierarchy_path": self.hierarchy_path,
            "log_level": self.log_level,
            "remove_occlusion_tokens": self.remove_occlusion_tokens,
            "sanitize_text": self.sanitize_text,
            "standardize_label_desc": self.standardize_label_desc,
        }

    @classmethod
    def from_args(cls, args) -> "DataConversionConfig":
        """Create config from command line arguments."""
        config_dict = {}

        # Map argument names to config fields
        arg_mapping = {
            "input_dir": "input_dir",
            "output_dir": "output_dir",
            "dataset_name": "dataset_name",
            "object_types": "object_types",
            "resize": "resize",
            "val_ratio": "val_ratio",
            "max_teachers": "max_teachers",
            "seed": "seed",
            "hierarchy_path": "hierarchy_path",
            "log_level": "log_level",
            "geometry_diversity_weight": "geometry_diversity_weight",
            "strip_occlusion": "remove_occlusion_tokens",
            "sanitize_text": "sanitize_text",
            "standardize_label_desc": "standardize_label_desc",
        }

        for arg_name, config_field in arg_mapping.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    config_dict[config_field] = value

        return cls(**config_dict)

    @classmethod
    def from_env(cls) -> "DataConversionConfig":
        """Create config from environment variables (for backward compatibility)."""
        import os

        config_dict = {}

        # Map environment variables to config fields
        # Direct 1:1 mapping - no conversion needed
        for _ in cls.__dataclass_fields__:
            env_var = field.upper()
            value = os.environ.get(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if field == "resize":
                    config_dict[field] = value.lower() in ("true", "1", "yes")
                elif field == "val_ratio":
                    config_dict[field] = float(value)
                elif field in ("max_teachers", "seed"):
                    config_dict[field] = int(value)
                elif field == "object_types":
                    config_dict[field] = value.split()
                else:
                    config_dict[field] = value

        return cls(**config_dict)


def setup_logging(config: DataConversionConfig) -> None:
    """Setup logging based on configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        encoding="utf-8",
    )

    logger.info(f"Logging configured at {config.log_level} level")


def validate_config(config: DataConversionConfig) -> None:
    """Additional validation for configuration consistency."""
    # Check if hierarchy file exists
    if config.hierarchy_path:
        hierarchy_path = Path(config.hierarchy_path)
        if not hierarchy_path.exists():
            logger.warning(f"Label hierarchy file not found: {hierarchy_path}")

    # Log configuration summary
    logger.info("Configuration Summary:")
    logger.info(f"  Input: {config.input_dir} → Output: {config.output_dir}")
    logger.info(f"  Language: Chinese (default)")
    logger.info(f"  Object Types: {config.object_types}")
    logger.info(f"  Image Resize: {'Enabled' if config.resize else 'Disabled'}")
    logger.info(f"  Teachers: {config.max_teachers}, Val Ratio: {config.val_ratio}")
