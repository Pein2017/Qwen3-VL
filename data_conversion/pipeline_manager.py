#!/usr/bin/env python3
"""
Unified Pipeline Manager

DEPRECATED: This alternative pipeline implementation is superseded by unified_processor.py
which is the standard entry point used by convert_dataset.sh.

This file is kept for reference but should not be used in production.
Use unified_processor.py directly or through convert_dataset.sh instead.

Original purpose: Replaces the complex bash script with a Python-based pipeline manager
that provides better error handling, logging, and progress tracking.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_conversion.config import DataConversionConfig
from data_conversion.coordinate_manager import FormatConverter
from data_conversion.unified_processor import UnifiedProcessor
from data_conversion.utils.file_ops import FileOperations
from data_conversion.vision_process import ImageProcessor


# Set UTF-8 encoding for stdout/stderr if supported
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


def setup_logging(config: DataConversionConfig) -> None:
    """Set up logging based on configuration."""
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger.info(f"Logging initialized at level {config.log_level}")


def validate_config(config: DataConversionConfig) -> bool:
    """Validate configuration settings."""
    # Check required paths
    input_dir = Path(config.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return False

    # Check output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check token map if specified
    if config.token_map_path and not Path(config.token_map_path).exists():
        logger.warning(f"Token map file not found: {config.token_map_path}")

    # Check hierarchy if specified
    if config.hierarchy_path and not Path(config.hierarchy_path).exists():
        logger.warning(f"Hierarchy file not found: {config.hierarchy_path}")

    return True


class TokenMapper:
    """Maps tokens in content according to a mapping dictionary."""

    def __init__(self, token_map: Dict[str, str]):
        """Initialize with token mapping dictionary."""
        self.token_map = token_map

    def apply_to_content_zh(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Apply token mapping to Chinese content dictionary."""
        if not content or not isinstance(content, dict):
            return content

        result = {}
        for key, value in content.items():
            if isinstance(value, str):
                # Apply mapping to string values
                mapped_value = self._map_string(value)
                result[key] = mapped_value
            elif isinstance(value, list):
                # Apply mapping to each item in list
                result[key] = [
                    self._map_string(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                # Keep other types unchanged
                result[key] = value

        return result

    def _map_string(self, text: str) -> str:
        """Apply token mapping to a single string."""
        if not text:
            return text

        result = text
        for old_token, new_token in self.token_map.items():
            result = result.replace(old_token, new_token)

        return result


class PipelineStep:
    """Represents a single step in the pipeline."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.success: bool = False
        self.error_message: Optional[str] = None

    def start(self) -> None:
        """Mark step as started."""
        self.start_time = time.time()
        logger.info(f"🔄 Step {self.name}: {self.description}")

    def complete(
        self, success: bool = True, error_message: Optional[str] = None
    ) -> None:
        """Mark step as completed."""
        self.end_time = time.time()
        self.success = success
        self.error_message = error_message

        duration = self.end_time - self.start_time if self.start_time else 0

        if success:
            logger.info(f"✅ Step {self.name} completed successfully ({duration:.2f}s)")
        else:
            logger.error(
                f"❌ Step {self.name} failed: {error_message} ({duration:.2f}s)"
            )

    def get_duration(self) -> float:
        """Get step duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


class PipelineManager:
    """Manages the complete data conversion pipeline."""

    def __init__(self, config: DataConversionConfig):
        """Initialize with configuration."""
        self.config = config
        self.steps: List[PipelineStep] = []

        # Initialize pipeline steps
        self._init_pipeline_steps()

        logger.info("PipelineManager initialized")

    def _init_pipeline_steps(self) -> None:
        """Initialize the pipeline steps."""
        self.steps = [
            PipelineStep("1", "Clean raw JSON annotation files"),
            PipelineStep("2", "Apply token mapping (if needed)"),
            PipelineStep("3", "Process samples and generate JSONL files"),
            PipelineStep("4", "Validate output files"),
            PipelineStep("5", "Generate summary report"),
        ]

    def clean_raw_json(self) -> bool:
        """Step 1: Clean raw JSON annotation files."""
        step = self.steps[0]
        step.start()

        try:
            input_dir = Path(self.config.input_dir)
            output_dir = (
                Path(self.config.output_image_dir)
                if self.config.output_image_dir
                else input_dir
            )

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Determine language parameter
            lang_map = {"chinese": "zh", "english": "en"}
            lang_map.get(self.config.language, "both")

            # Find and process JSON files
            json_files = FileOperations.find_json_files(input_dir)
            cleaned_count = 0

            for json_file in json_files:
                # Load original data
                original_data = FileOperations.load_json_data(json_file)

                # Clean the data
                cleaned_data = FormatConverter.clean_annotation_content(original_data)

                # Save cleaned data
                if output_dir != input_dir:
                    output_file = output_dir / json_file.name
                    FileOperations.save_json_data(cleaned_data, output_file)
                    cleaned_count += 1
                else:
                    # In-place cleaning
                    FileOperations.save_json_data(cleaned_data, json_file)
                    cleaned_count += 1

            # Process images if output directory is different
            if output_dir != input_dir:
                self._process_images_and_update_json(input_dir, output_dir)

            logger.info(f"Cleaned {cleaned_count} JSON files")
            step.complete(True)
            return True

        except Exception as e:
            step.complete(False, str(e))
            return False

    def apply_token_mapping(self) -> bool:
        """Step 2: Apply token mapping to cleaned JSON files."""
        step = self.steps[1]
        step.start()

        try:
            # Skip if no token mapping needed
            if not self.config.token_map_path or self.config.language != "chinese":
                logger.info("Token mapping skipped (not needed for this configuration)")
                step.complete(True)
                return True

            # Load token mapping
            token_map_path = Path(self.config.token_map_path)
            token_map = FileOperations.load_token_map(token_map_path)
            token_mapper = TokenMapper(token_map)

            # Apply to JSON files in output directory
            output_dir = (
                Path(self.config.output_image_dir)
                if self.config.output_image_dir
                else Path(self.config.input_dir)
            )
            json_files = list(output_dir.glob("*.json"))

            mapped_count = 0
            for json_file in json_files:
                # Load data
                data = FileOperations.load_json_data(json_file)

                # Apply token mapping to features
                if "markResult" in data and "features" in data["markResult"]:
                    for feature in data["markResult"]["features"]:
                        if (
                            "properties" in feature
                            and "contentZh" in feature["properties"]
                        ):
                            content_zh = feature["properties"]["contentZh"]
                            updated_content = token_mapper.apply_to_content_zh(
                                content_zh
                            )
                            feature["properties"]["contentZh"] = updated_content

                # Save updated data
                FileOperations.save_json_data(data, json_file)
                mapped_count += 1

            logger.info(f"Applied token mapping to {mapped_count} files")
            step.complete(True)
            return True

        except Exception as e:
            step.complete(False, str(e))
            return False

    def process_samples(self) -> Optional[Dict[str, int]]:
        """Step 3: Process samples and generate JSONL files."""
        step = self.steps[2]
        step.start()

        try:
            # Create modified config for processing
            # Use output_image_dir as input if files were moved there
            processing_input_dir = self.config.input_dir
            if (
                self.config.output_image_dir
                and Path(self.config.output_image_dir).exists()
            ):
                # Check if processed files are in output_image_dir
                output_image_path = Path(self.config.output_image_dir)
                json_files_in_output = list(output_image_path.glob("*.json"))
                if json_files_in_output:
                    processing_input_dir = str(output_image_path)

            # Create config for processing
            processing_config = DataConversionConfig(
                input_dir=processing_input_dir,
                output_dir=self.config.output_dir,
                object_types=self.config.object_types
                if hasattr(self.config, "object_types")
                else ["bbu", "label", "fiber", "wire"],
                resize=self.config.resize,
                val_ratio=self.config.val_ratio,
                max_teachers=self.config.max_teachers,
                seed=self.config.seed,
                output_image_dir=self.config.output_image_dir,
                language=self.config.language,
                response_types=self.config.response_types,
                token_map_path=self.config.token_map_path,
                hierarchy_path=self.config.hierarchy_path,
                log_level=self.config.log_level,
                fail_fast=self.config.fail_fast,
            )

            # Create unified processor with updated config
            # Ensure object_types is properly set in the config
            if (
                not hasattr(processing_config, "object_types")
                or not processing_config.object_types
            ):
                # Default to common object types if not specified
                processing_config.object_types = ["bbu", "label", "fiber", "wire"]
                logger.info(
                    f"Using default object types: {processing_config.object_types}"
                )

            processor = UnifiedProcessor(processing_config)

            # Execute processing
            result = processor.process()

            step.complete(True)
            return result

        except Exception as e:
            step.complete(False, str(e))
            return None

    def validate_output(self) -> bool:
        """Step 4: Validate output files."""
        step = self.steps[3]
        step.start()

        try:
            output_dir = Path(self.config.output_dir)

            # Check if output files exist
            required_files = [
                "train.jsonl",
                "val.jsonl",
                "teacher.jsonl",
                "all_samples.jsonl",
            ]
            for filename in required_files:
                file_path = output_dir / filename
                if not file_path.exists():
                    raise FileNotFoundError(f"Output file missing: {file_path}")

            # Count samples in each file
            counts = {}
            for filename in required_files:
                file_path = output_dir / filename
                with open(file_path, "r", encoding="utf-8") as f:
                    counts[filename] = sum(1 for line in f if line.strip())

            # Validate sample counts
            expected_total = (
                counts["train.jsonl"] + counts["val.jsonl"] + counts["teacher.jsonl"]
            )
            if counts["all_samples.jsonl"] != expected_total:
                raise ValueError(
                    f"Sample count mismatch: all_samples.jsonl has {counts['all_samples.jsonl']} "
                    f"but expected {expected_total}"
                )

            logger.info("Output validation passed")
            logger.info(f"Sample counts: {counts}")

            step.complete(True)
            return True

        except Exception as e:
            step.complete(False, str(e))
            return False

    def generate_summary(self, processing_result: Optional[Dict[str, int]]) -> bool:
        """Step 5: Generate summary report."""
        step = self.steps[4]
        step.start()

        try:
            # Calculate total pipeline duration
            total_duration = sum(
                s.get_duration() for s in self.steps[:-1]
            )  # Exclude current step

            # Generate summary
            summary = {
                "configuration": {
                    "language": self.config.language,
                    "response_types": self.config.response_types,
                    "resize": self.config.resize,
                    "input_dir": self.config.input_dir,
                    "output_dir": self.config.output_dir,
                },
                "processing_results": processing_result or {},
                "pipeline_duration": f"{total_duration:.2f} seconds",
                "steps": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "success": s.success,
                        "duration": f"{s.get_duration():.2f}s",
                        "error": s.error_message,
                    }
                    for s in self.steps[:-1]  # Exclude current step
                ],
            }

            # Save summary to file
            summary_path = Path(self.config.output_dir) / "pipeline_summary.json"
            FileOperations.save_json_data(summary, summary_path, indent=2)

            logger.info(f"Pipeline summary saved to {summary_path}")
            step.complete(True)
            return True

        except Exception as e:
            step.complete(False, str(e))
            return False

    def _process_images_and_update_json(
        self, input_dir: Path, output_dir: Path
    ) -> None:
        """Process images with smart resizing and update JSON dimensions accordingly."""
        # Create temporary image processor for this step
        temp_config = DataConversionConfig(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            object_types=self.config.object_types
            if hasattr(self.config, "object_types")
            else ["bbu", "label"],
            resize=self.config.resize,
            val_ratio=self.config.val_ratio
            if hasattr(self.config, "val_ratio")
            else 0.1,
            max_teachers=self.config.max_teachers
            if hasattr(self.config, "max_teachers")
            else 10,
            seed=self.config.seed if hasattr(self.config, "seed") else 42,
            output_image_dir=str(output_dir),  # Process to same directory
            language=self.config.language,
            log_level=self.config.log_level,
        )

        image_processor = ImageProcessor(temp_config)
        processed_count = 0

        # Process each image and update corresponding JSON
        for ext in ["*.jpeg", "*.jpg"]:
            for image_file in input_dir.glob(ext):
                try:
                    # Get original dimensions
                    original_width, original_height = (
                        FileOperations.get_image_dimensions(image_file)
                    )

                    # Process image (copy or resize)
                    _, final_width, final_height = image_processor.process_image(
                        image_file,
                        original_width,
                        original_height,
                        output_dir.parent,
                    )

                    # Update corresponding JSON file if dimensions changed
                    if final_width != original_width or final_height != original_height:
                        json_file = output_dir / f"{image_file.stem}.json"
                        if json_file.exists():
                            # Load JSON data
                            json_data = FileOperations.load_json_data(json_file)

                            # Update dimensions in info section
                            if "info" in json_data:
                                json_data["info"]["width"] = final_width
                                json_data["info"]["height"] = final_height

                                # Scale coordinates if necessary
                                if (
                                    "markResult" in json_data
                                    and "features" in json_data["markResult"]
                                ):
                                    x_scale = final_width / original_width
                                    y_scale = final_height / original_height

                                    for feature in json_data["markResult"]["features"]:
                                        geometry = feature.get("geometry", {})
                                        if geometry.get("type") == "ExtentPolygon":
                                            coordinates = geometry.get(
                                                "coordinates", []
                                            )
                                            if coordinates:
                                                scaled_coordinates = []
                                                for point in coordinates:
                                                    if len(point) >= 2:
                                                        scaled_x = point[0] * x_scale
                                                        scaled_y = point[1] * y_scale
                                                        scaled_coordinates.append(
                                                            [scaled_x, scaled_y]
                                                        )
                                                geometry["coordinates"] = (
                                                    scaled_coordinates
                                                )

                                # Save updated JSON
                                FileOperations.save_json_data(json_data, json_file)
                                logger.debug(
                                    f"Updated JSON dimensions for {json_file.name}: {original_width}x{original_height} → {final_width}x{final_height}"
                                )

                    processed_count += 1

                except Exception as e:
                    logger.error(f"Error processing image {image_file}: {e}")
                    if self.config.fail_fast:
                        raise

        logger.info(f"Processed {processed_count} images and updated JSON dimensions")

    def run(self) -> bool:
        """Execute the complete pipeline."""
        logger.info("🚀 Starting Data Conversion Pipeline")
        logger.info("=" * 60)

        start_time = time.time()
        processing_result = None

        try:
            # Step 1: Clean raw JSON files
            if not self.clean_raw_json():
                return False

            # Step 2: Apply token mapping
            if not self.apply_token_mapping():
                return False

            # Step 3: Process samples
            processing_result = self.process_samples()
            if processing_result is None:
                return False

            # Step 4: Validate output
            if not self.validate_output():
                return False

            # Step 5: Generate summary
            if not self.generate_summary(processing_result):
                return False

            # Final success message
            total_duration = time.time() - start_time
            logger.info("=" * 60)
            logger.info("🎉 Pipeline completed successfully!")
            logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
            logger.info("🚀 Ready for training!")

            return True

        except Exception as e:
            logger.error(f"Pipeline failed with unexpected error: {e}")
            return False

    def get_status(self) -> Dict:
        """Get current pipeline status."""
        completed_steps = sum(1 for s in self.steps if s.success)
        failed_steps = sum(1 for s in self.steps if s.error_message is not None)

        return {
            "total_steps": len(self.steps),
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "success_rate": completed_steps / len(self.steps) if self.steps else 0,
            "steps": [
                {
                    "name": s.name,
                    "description": s.description,
                    "success": s.success,
                    "duration": s.get_duration(),
                    "error": s.error_message,
                }
                for s in self.steps
            ],
        }


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Unified Data Conversion Pipeline Manager"
    )

    # Configuration options
    parser.add_argument(
        "--input_dir", default="ds", help="Input directory with JSON/image files"
    )
    parser.add_argument(
        "--output_dir", default="data", help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--output_image_dir", help="Output directory for processed images"
    )
    parser.add_argument(
        "--language",
        choices=["chinese", "english"],
        default="chinese",
        help="Language mode",
    )
    parser.add_argument(
        "--response_types",
        nargs="+",
        default=["object_type", "property"],
        help="Response types to include",
    )
    parser.add_argument("--resize", action="store_true", help="Enable image resizing")
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--max_teachers", type=int, default=10, help="Maximum teacher samples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--token_map_path", help="Path to token mapping file")
    parser.add_argument("--hierarchy_path", help="Path to label hierarchy file")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--standardize_label_desc",
        action="store_true",
        help="Standardize label descriptions: map '标签/*' empty-like values (空格/看不清/、 or empty) to '标签/无法识别'",
    )

    # Environment variable support
    parser.add_argument(
        "--from_env",
        action="store_true",
        help="Load configuration from environment variables",
    )

    args = parser.parse_args()

    # Create configuration
    if args.from_env:
        config = DataConversionConfig.from_env()
    else:
        config = DataConversionConfig.from_args(args)

    # Setup logging
    setup_logging(config)

    # Validate configuration
    validate_config(config)

    # Create and run pipeline
    pipeline = PipelineManager(config)
    success = pipeline.run()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
