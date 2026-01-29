#!/usr/bin/env python3
"""
Configuration Management for Data Conversion Pipeline

Provides structured, type-safe configuration with validation.
Replaces environment variables and complex command-line arguments.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, cast


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


@dataclass(frozen=True)
class DataConversionConfig:
    """Configuration for the data conversion pipeline.

    This is a frozen dataclass (immutable after creation) to enforce configuration
    stability and prevent accidental mutations. All parameters are required and must
    be explicitly set. No default values to ensure manual configuration.

    **Immutability Contract:**
    - Once created, the config cannot be modified (frozen=True)
    - If you need to set fields in __post_init__, use: object.__setattr__(self, "field_name", value)
    - This ensures configuration is stable and predictable throughout the pipeline

    **Validation:**
    - All validation happens in __post_init__() - fail fast on invalid config
    - Callers should not perform additional validation; trust the config object
    - Invalid configs raise ValueError with clear remediation hints
    """

    # Required fields (no defaults)
    input_dir: str
    output_dir: str
    resize: bool  # True or False - REQUIRED
    val_ratio: float  # e.g., 0.1 for 10% validation - REQUIRED
    seed: int  # e.g., 42 - REQUIRED

    # Image resize parameters - REQUIRED
    max_pixels: int  # Maximum pixels for image resizing (e.g., 768 * 32*32 = 786432)
    image_factor: int  # Factor for image dimensions (e.g., 32)

    # Optional fields (with defaults)
    dataset_name: Optional[str] = None  # Auto-detected from input_dir if not provided
    # Processing parameters - OPTIONAL WITH DEFAULTS
    log_level: str = "INFO"
    fail_fast: bool = True

    # Filtering options - OPTIONAL WITH DEFAULTS
    remove_occlusion_tokens: bool = False  # Drop tokens containing "遮挡" from desc
    sanitize_text: bool = True  # Apply text normalization/sanitization on descriptions
    standardize_label_desc: bool = True  # Legacy flag (no-op in key=value mode)

    # Ordering options - OPTIONAL WITH DEFAULTS
    preserve_annotation_order: bool = False  # Keep original object ordering when True
    object_ordering_policy: str = (
        "center_tlbr"  # TLBR ordering policy when reordering is enabled
    )

    # Debugging options - OPTIONAL WITH DEFAULTS
    limit: int = -1  # Limit number of images to process (-1 for all images, positive number for limit)
    validation_mode: str = "strict"
    min_object_size: int = 10
    enable_validation_reports: bool = True

    # Performance options - OPTIONAL WITH DEFAULTS
    num_workers: int = 1  # Number of parallel workers for multiprocessing (1 = sequential, >1 = parallel)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._setup_dataset_name()
        self._validate_paths()
        self._validate_parameters()

    def _setup_dataset_name(self) -> None:
        """Auto-detect dataset name from input directory if not provided."""
        if self.dataset_name is None:
            object.__setattr__(self, "dataset_name", Path(self.input_dir).name)
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

        if self.max_pixels <= 0:
            raise ValueError(f"max_pixels must be positive, got {self.max_pixels}")

        if self.image_factor <= 0:
            raise ValueError(f"image_factor must be positive, got {self.image_factor}")

        if self.limit < -1:
            raise ValueError(
                f"limit must be -1 (all images) or a positive number, got {self.limit}"
            )

        if self.min_object_size <= 0:
            raise ValueError(
                f"min_object_size must be positive, got {self.min_object_size}"
            )

        allowed_modes = {"strict", "lenient", "warning_only"}
        if self.validation_mode not in allowed_modes:
            raise ValueError(
                f"validation_mode must be one of {allowed_modes}, got {self.validation_mode}"
            )

        if self.num_workers < 1:
            raise ValueError(f"num_workers must be at least 1, got {self.num_workers}")

        try:
            from data_conversion.utils.sorting import normalize_object_ordering_policy

            normalized = normalize_object_ordering_policy(self.object_ordering_policy)
        except Exception as exc:
            raise ValueError(
                "object_ordering_policy must be one of {'reference_tlbr', 'center_tlbr'}; "
                f"got {self.object_ordering_policy!r}"
            ) from exc
        object.__setattr__(self, "object_ordering_policy", normalized)

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for compatibility."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "resize": self.resize,
            "val_ratio": self.val_ratio,
            "seed": self.seed,
            "log_level": self.log_level,
            "remove_occlusion_tokens": self.remove_occlusion_tokens,
            "sanitize_text": self.sanitize_text,
            "standardize_label_desc": self.standardize_label_desc,
            "preserve_annotation_order": self.preserve_annotation_order,
            "object_ordering_policy": self.object_ordering_policy,
            "validation_mode": self.validation_mode,
            "min_object_size": self.min_object_size,
            "enable_validation_reports": self.enable_validation_reports,
            "num_workers": self.num_workers,
        }

    @classmethod
    def from_args(cls, args: object) -> "DataConversionConfig":
        """Create config from command line arguments."""
        config_dict: dict[str, Any] = {}

        # Map argument names to config fields
        arg_mapping = {
            "input_dir": "input_dir",
            "output_dir": "output_dir",
            "dataset_name": "dataset_name",
            "resize": "resize",
            "val_ratio": "val_ratio",
            "seed": "seed",
            "log_level": "log_level",
            "strip_occlusion": "remove_occlusion_tokens",
            "sanitize_text": "sanitize_text",
            "standardize_label_desc": "standardize_label_desc",
            "fail_fast": "fail_fast",
            "max_pixels": "max_pixels",
            "image_factor": "image_factor",
            "limit": "limit",
            "preserve_annotation_order": "preserve_annotation_order",
            "object_ordering_policy": "object_ordering_policy",
            "validation_mode": "validation_mode",
            "min_object_size": "min_object_size",
            "enable_validation_reports": "enable_validation_reports",
            "num_workers": "num_workers",
        }

        for arg_name, config_field in arg_mapping.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    config_dict[config_field] = value

        # NOTE: kwargs are assembled dynamically from CLI/env; runtime validation in
        # __post_init__ enforces correctness. Cast to Any to keep type checkers
        # from treating the kwargs mapping as Unknown.
        return cls(**cast(Any, config_dict))

    @classmethod
    def from_env(cls) -> "DataConversionConfig":
        """Create config from environment variables.

        Raises:
            ValueError: If required environment variables are missing
        """
        import os

        required_env_vars = {
            "INPUT_DIR": "input_dir",
            "OUTPUT_DIR": "output_dir",
            "RESIZE": "resize",
            "VAL_RATIO": "val_ratio",
            "SEED": "seed",
            "MAX_PIXELS": "max_pixels",
            "IMAGE_FACTOR": "image_factor",
        }

        config_dict: dict[str, Any] = {}

        for env_var, config_field in required_env_vars.items():
            value = os.getenv(env_var)
            if value is None:
                raise ValueError(f"Required environment variable not set: {env_var}")

            # Type conversion
            if config_field in ("resize",):
                config_dict[config_field] = value.lower() in ("true", "1", "yes")
            elif config_field in ("val_ratio",):
                config_dict[config_field] = float(value)
            elif config_field in ("seed", "max_pixels", "image_factor"):
                config_dict[config_field] = int(value)
            else:
                config_dict[config_field] = value

        # Optional environment variables
        optional_env_vars = {
            "DATASET_NAME": "dataset_name",
            "LOG_LEVEL": "log_level",
            "STRIP_OCCLUSION": "remove_occlusion_tokens",
            "SANITIZE_TEXT": "sanitize_text",
            "STANDARDIZE_LABEL_DESC": "standardize_label_desc",
            "PRESERVE_ANNOTATION_ORDER": "preserve_annotation_order",
            "OBJECT_ORDERING_POLICY": "object_ordering_policy",
            "LIMIT": "limit",
            "VALIDATION_MODE": "validation_mode",
            "MIN_OBJECT_SIZE": "min_object_size",
            "ENABLE_VALIDATION_REPORTS": "enable_validation_reports",
            "NUM_WORKERS": "num_workers",
        }

        for env_var, config_field in optional_env_vars.items():
            value = os.getenv(env_var)
            if value is not None:
                if config_field in (
                    "remove_occlusion_tokens",
                    "sanitize_text",
                    "standardize_label_desc",
                    "preserve_annotation_order",
                ):
                    config_dict[config_field] = value.lower() in ("true", "1", "yes")
                elif config_field == "limit":
                    config_dict[config_field] = int(value)
                elif config_field == "min_object_size":
                    config_dict[config_field] = int(value)
                elif config_field == "enable_validation_reports":
                    config_dict[config_field] = value.lower() in ("true", "1", "yes")
                else:
                    config_dict[config_field] = value

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
    """Log configuration summary and check optional file paths.

    Note: Core validation happens in DataConversionConfig.__post_init__().
    This function is for logging and checking optional file existence.

    Args:
        config: DataConversionConfig instance (already validated)
    """
    # Log configuration summary
    logger.info("Configuration Summary:")
    logger.info(f"  Input: {config.input_dir} → Output: {config.output_dir}")
    logger.info(f"  Dataset: {config.dataset_name}")
    logger.info("  Object Types: all supported categories")
    logger.info(f"  Image Resize: {'Enabled' if config.resize else 'Disabled'}")
    logger.info(f"  Val Ratio: {config.val_ratio}")
    logger.info(f"  Seed: {config.seed}")
    logger.info(
        f"  Max Pixels: {config.max_pixels}, Image Factor: {config.image_factor}"
    )
    if config.preserve_annotation_order:
        logger.info(
            "  Object Ordering: preserve_annotation_order=true (no TLBR sort applied)"
        )
    else:
        logger.info(f"  Object Ordering: {config.object_ordering_policy}")
    if config.limit > 0:
        logger.info(f"  Limit: {config.limit} images")
