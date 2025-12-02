#!/usr/bin/env python3
"""
Streamlined Unified Data Processor

Consolidates all data processing functionality with simplified architecture.
Merged SampleExtractor directly into UnifiedProcessor to eliminate redundancy.
"""

import logging
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from data_conversion.pipeline.config import (
    DataConversionConfig,
    setup_logging,
    validate_config,
)
from data_conversion.pipeline.constants import DEFAULT_LABEL_HIERARCHY, OBJECT_TYPES
from data_conversion.pipeline.coordinate_manager import CoordinateManager
from data_conversion.pipeline.data_splitter import DataSplitter
from data_conversion.pipeline.flexible_taxonomy_processor import (
    HierarchicalProcessor,
)
from data_conversion.pipeline.format_converter import FormatConverter
from data_conversion.pipeline.summary_builder import build_summary_from_objects
from data_conversion.utils.file_ops import FileOperations
from data_conversion.utils.sanitizer_pipeline import (
    SanitizerPipeline,
    SanitizerStep,
)
from data_conversion.utils.sanitizers import (
    remove_screw_completeness_attributes,
    sanitize_text,
    standardize_label_description,
    strip_annotator_notes,
    strip_occlusion_tokens,
)
from data_conversion.utils.sorting import sort_objects_tlbr
from data_conversion.utils.review_flagger import flag_objects_for_review
from data_conversion.pipeline.validation_manager import (
    StructureValidator,
    ValidationManager,
)
from data_conversion.pipeline.vision_process import ImageProcessor

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

# Global worker state - initialized once per worker process
_worker_processor = None


# ============================================================================
# Multiprocessing Worker Functions
# ============================================================================


def _init_worker(config: DataConversionConfig, label_hierarchy: Dict) -> None:
    """
    Initialize worker process state.

    This function is called once per worker when the Pool is created.
    It creates a single UnifiedProcessor instance that will be reused
    for all samples processed by this worker.

    Args:
        config: Data conversion configuration
        label_hierarchy: Label hierarchy dictionary
    """
    global _worker_processor

    # Suppress initialization logs in worker processes to avoid spam
    # Only show WARNING and above from workers
    logging.getLogger("data_conversion").setLevel(logging.WARNING)
    logging.getLogger("__main__").setLevel(logging.WARNING)

    # Create processor instance once per worker
    _worker_processor = UnifiedProcessor(config)
    _worker_processor.label_hierarchy = label_hierarchy


def _process_sample_worker(json_path: Path) -> Optional[Dict]:
    """
    Worker function for parallel sample processing.

    This function is called for each sample. It reuses the processor
    instance created in _init_worker() to avoid repeated initialization.

    Args:
        json_path: Path to the JSON file to process

    Returns:
        Processed sample dict or None if processing failed
    """
    global _worker_processor

    if _worker_processor is None:
        raise RuntimeError("Worker not initialized - _init_worker must be called first")

    # Process the single sample using the shared worker processor
    return _worker_processor.process_single_sample(json_path)


class UnifiedProcessor:
    """Streamlined orchestrator for the unified data processing pipeline."""

    def __init__(self, config: DataConversionConfig):
        """Initialize processor with configuration."""
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.output_dir = config.get_dataset_output_dir()
        self.is_rru = "rru" in str(self.input_dir).lower() or (
            getattr(config, "dataset_name", "") and "rru" in config.dataset_name.lower()
        )

        # Initialize components - no token mapping needed for Chinese-only

        # Use built-in hierarchy for consistent processing
        self.label_hierarchy = DEFAULT_LABEL_HIERARCHY

        # Initialize hierarchical processor for v2 data support (Chinese only)
        self.hierarchical_processor = HierarchicalProcessor(
            object_types=set(OBJECT_TYPES),
            label_hierarchy=self.label_hierarchy,
        )

        self.image_processor = ImageProcessor(config)
        self.data_splitter = DataSplitter(val_ratio=config.val_ratio, seed=config.seed)

        validation_mode = getattr(config, "validation_mode", "strict")
        require_desc = True
        check_bounds = True
        if validation_mode == "lenient":
            require_desc = False
        elif validation_mode == "warning_only":
            require_desc = False
            check_bounds = False

        self.validation_manager = ValidationManager(
            min_object_size=getattr(config, "min_object_size", 10),
            require_non_empty_description=require_desc,
            check_coordinate_bounds=check_bounds,
        )
        self.validation_mode = validation_mode
        self.enable_validation_reports = getattr(
            config, "enable_validation_reports", True
        )

        # Track invalid objects and samples for reporting / legacy compatibility
        self.invalid_objects: List[Dict] = []
        self.invalid_samples: List[Dict] = []

        logger.info("UnifiedProcessor initialized successfully (Chinese-only mode)")

    def extract_content_fields(self, source_dict: Dict) -> Dict[str, str]:
        """Extract and normalize content fields from Chinese contentZh format."""
        return self._extract_chinese_fields(source_dict)

    def _extract_chinese_fields(self, source_dict: Dict) -> Dict[str, str]:
        """Extract fields from Chinese contentZh format."""
        content_zh = source_dict.get("contentZh", {})
        if not content_zh:
            return {}

        # Extract label entries containing '标签' or '标签贴纸' (mapped version)
        label_values = []
        for key, value in content_zh.items():
            if "标签" in key:  # Matches both "标签" and "标签贴纸"
                if isinstance(value, list):
                    label_values.append(", ".join(map(str, value)))
                elif value:
                    label_values.append(str(value))

        if not label_values:
            return {}

        # Parse first label entry: "object_type/property/extra"
        label_string = label_values[0]
        parts = [p.strip() for p in label_string.split("/")]
        object_type = parts[0] if len(parts) >= 1 else ""
        property_value = parts[1] if len(parts) >= 2 else ""
        existing_extras = parts[2:] if len(parts) >= 3 else []

        # Collect additional extra_info from other contentZh entries
        additional_extras = []
        for key, value in content_zh.items():
            if "标签" not in key:
                if isinstance(value, list):
                    additional_extras.extend(str(item) for item in value if item)
                elif value:
                    additional_extras.append(str(value))

        extra_info = "/".join(existing_extras + additional_extras)

        return {
            "object_type": object_type,
            "property": property_value,
            "extra_info": extra_info,
        }

    def is_allowed_object(self, content_dict: Dict[str, str]) -> bool:
        """Check if object passes label hierarchy filtering."""
        obj_type = content_dict.get("object_type", "")
        prop = content_dict.get("property", "")

        # If no hierarchy is loaded, allow all objects
        if not self.label_hierarchy:
            return bool(obj_type)  # At least require an object type

        # Skip if object_type not in hierarchy
        if obj_type not in self.label_hierarchy:
            return False

        allowed_props = self.label_hierarchy.get(obj_type, [])

        # If no properties allowed, only accept empty property
        if not allowed_props:
            return prop == "" or prop is None

        # Check if property is directly allowed
        if prop in allowed_props:
            return True

        # Allow variant "obj_type/property" format stored in hierarchy
        combo = f"{obj_type}/{prop}" if prop else obj_type
        return combo in allowed_props

    def _sanitize_description(self, desc: str) -> str:
        steps: List[SanitizerStep] = []

        if getattr(self.config, "sanitize_text", False):
            steps.append(SanitizerStep("sanitize_text", sanitize_text, mandatory=True))
            # Always remove annotator notes when sanitize_text is enabled
            steps.append(
                SanitizerStep(
                    "strip_annotator_notes",
                    strip_annotator_notes,
                    mandatory=False,
                )
            )

        if getattr(self.config, "remove_occlusion_tokens", False):
            steps.append(
                SanitizerStep(
                    "strip_occlusion_tokens",
                    strip_occlusion_tokens,
                    mandatory=True,
                )
            )

        if getattr(self.config, "standardize_label_desc", False):
            steps.append(
                SanitizerStep(
                    "standardize_label_description",
                    standardize_label_description,
                    mandatory=True,
                )
            )

        # Always attempt to remove completeness attributes for screw objects
        steps.append(
            SanitizerStep(
                "remove_screw_completeness_attributes",
                remove_screw_completeness_attributes,
                mandatory=False,
            )
        )

        if not steps:
            return desc

        pipeline = SanitizerPipeline(
            steps,
            fail_fast=getattr(self.config, "fail_fast", True),
        )
        sanitized = pipeline.run(desc)
        return sanitized if sanitized is not None else ""

    def _rewrite_desc_with_remark(self, desc: str) -> str:
        """Rewrite hierarchical desc by folding trailing free-text level into ',备注:...'.

        Rules (from hierarchical_attribute_mapping.json):
        - Levels are separated by '/'; same-level attributes use ','.
        - Remark exists only for non-标签 types and is always the final level AFTER all structured levels.
          Structured levels depend on object type and L1 values:
            * BBU设备: if L1 contains '机柜空间充足需要安装', then level-2 is structured (挡风板符合性)
            * 螺丝、光纤插头: if L1 contains '不符合要求', then level-2 is structured (具体问题)
            * 光纤: if L1 contains '有保护措施', then level-2 is structured (保护细节)
            * 挡风板/电线: only level-1 is structured
            * 标签: no remark
        - If a remark is detected, remove its slash-level and append ',备注:{remark}'.
        - If no remark is detected, return desc unchanged.
        """
        try:
            # 如果包含组信息，直接跳过改写以避免把组当作备注
            if "组/" in (desc or ""):
                return desc
            parts = [p for p in (desc or "").split("/") if p != ""]
            if not parts:
                return desc
            obj = parts[0]
            levels = parts[1:]
            # 标签不支持备注
            if obj.startswith("标签"):
                return desc
            # 计算结构化层数
            structured_count = 1 if levels else 0
            l1_tokens = []
            if levels:
                l1_tokens = [t.strip() for t in levels[0].split(",") if t.strip()]
            if obj.startswith("BBU设备") and any(
                "机柜空间充足需要安装" in t for t in l1_tokens
            ):
                structured_count = min(2, len(levels))
            elif obj.startswith("螺丝、光纤插头") and any(
                "不符合要求" in t for t in l1_tokens
            ):
                structured_count = min(2, len(levels))
            elif obj.startswith("光纤") and any("有保护措施" in t for t in l1_tokens):
                structured_count = min(2, len(levels))
            elif obj.startswith("挡风板") or obj.startswith("电线"):
                structured_count = min(1, len(levels))

            # 如果存在额外层，视为备注层（最后一层）
            if len(levels) > structured_count:
                remark = levels[-1]
                base = (
                    "/".join([obj] + levels[:structured_count])
                    if structured_count > 0
                    else obj
                )
                # 将备注折叠到末尾，保持半角逗号和冒号风格
                return f"{base},备注:{remark}"
            return desc
        except Exception as e:
            logger.error(f"Error rewriting desc with remark: {e}")
            raise Exception("Error rewriting desc with remark")

    def extract_objects_from_datalist(self, data_list: List[Dict]) -> List[Dict]:
        """Extract objects from dataList format."""
        objects = []

        for item in data_list:
            coords = item.get("coordinates", [])
            if len(coords) < 2:
                logger.warning(f"Invalid coordinates in dataList item: {coords}")
                continue

            x1, y1 = coords[0]
            x2, y2 = coords[1]
            bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            # Clean bbox coordinates for VLM training - ensure proper ordering and bounds
            bbox = [max(0, coord) for coord in bbox]  # Remove any negative coordinates
            # Ensure min <= max for both x and y
            x_min, y_min, x_max, y_max = bbox
            bbox = [
                min(x_min, x_max),
                min(y_min, y_max),
                max(x_min, x_max),
                max(y_min, y_max),
            ]

            properties = item.get("properties", {}) or {}
            content_dict = self.extract_content_fields(properties)

            if not content_dict or not self.is_allowed_object(content_dict):
                continue

            desc = FormatConverter.format_description(content_dict)
            if desc:
                if (
                    getattr(self.config, "remove_occlusion_tokens", False)
                    or getattr(self.config, "sanitize_text", False)
                    or getattr(self.config, "standardize_label_desc", False)
                ):
                    desc = self._sanitize_description(desc)
                if desc:
                    desc = self._rewrite_desc_with_remark(desc)
                    objects.append({"bbox_2d": bbox, "desc": desc})

        return objects

    def extract_objects_from_markresult(
        self, features: List[Dict], image_id: Optional[str] = None
    ) -> List[Dict]:
        """Extract objects from markResult features with native geometry types.

        Args:
            features: List of V2 feature dictionaries
            image_id: Optional image identifier for logging
        """
        # Use hierarchical processor for V2 data support (pass image_id for detailed logging)
        objects = self.hierarchical_processor.extract_objects_from_markresult(
            features, image_id=image_id
        )
        # Apply sanitizer if configured
        if (
            getattr(self.config, "remove_occlusion_tokens", False)
            or getattr(self.config, "sanitize_text", False)
            or getattr(self.config, "standardize_label_desc", False)
        ):
            for obj in objects:
                d = obj.get("desc", "")
                if d:
                    obj["desc"] = self._sanitize_description(d)
        # Fold trailing free-text level into ',备注:...' deterministically
        for obj in objects:
            d = obj.get("desc", "")
            if d:
                obj["desc"] = self._rewrite_desc_with_remark(d)
        return objects

    def process_single_sample(self, json_path: Path) -> Optional[Dict]:
        """Process a single JSON/image pair into a clean sample."""
        try:
            # Load JSON and find corresponding image
            json_data = FileOperations.load_json_data(json_path)
            image_path = FileOperations.find_image_file(json_path)

            # Get dimensions from JSON (these should be the processed dimensions)
            info = json_data["info"]
            json_width = info["width"]
            json_height = info["height"]

            # Get actual image dimensions
            actual_width, actual_height = FileOperations.get_image_dimensions(
                image_path
            )

            # Detect dimension mismatch (after EXIF correction)
            if json_width != actual_width or json_height != actual_height:
                logger.info(
                    f"Dimension mismatch for {image_path.name}: "
                    f"JSON says {json_width}x{json_height} but EXIF-normalized image is "
                    f"{actual_width}x{actual_height}. Will apply coordinate rescaling."
                )

            # Keep JSON dimensions for coordinate transformation pipeline
            _, _ = json_width, json_height

            # Extract objects from JSON data
            objects = []
            if "dataList" in json_data:
                objects = self.extract_objects_from_datalist(json_data["dataList"])
            elif "markResult" in json_data and isinstance(
                json_data.get("markResult", {}).get("features"), list
            ):
                # Extract image ID from file path for detailed logging
                image_id = json_path.stem
                objects = self.extract_objects_from_markresult(
                    json_data["markResult"]["features"], image_id=image_id
                )

            if not objects:
                logger.warning(f"No valid objects found in {json_path.name}")
                return None

            # Apply unified coordinate transformation pipeline
            sample_data = {"objects": objects}
            processed_sample, final_width, final_height = (
                self._process_sample_coordinates_unified(
                    sample_data, image_path, json_width, json_height, self.config.resize
                )
            )
            objects = processed_sample["objects"]

            # Filter objects via strict validation layer
            objects = self._filter_valid_objects(
                objects, final_width, final_height, str(image_path.name)
            )

            if not objects:
                # Track invalid sample for reporting
                invalid_sample = {
                    "sample_id": str(json_path.name),
                    "reason": "no_valid_objects",
                    "original_object_count": len(sample_data.get("objects", [])),
                    "image_path": str(image_path),
                    "json_path": str(json_path),
                }
                self.invalid_samples.append(invalid_sample)

                # Skip sample if no objects
                logger.warning(
                    f"No valid objects for {json_path.name}, skipping sample"
                )
                return None

            # 组完整性检查：任一组只有1个成员则判定异常
            group_counts = {}
            for obj in objects:
                for g in obj.get("groups") or []:
                    gid = g.get("name") or g.get("id")
                    if gid is None:
                        continue
                    group_counts[gid] = group_counts.get(gid, 0) + 1
            if any(cnt == 1 for cnt in group_counts.values()):
                msg = f"Group integrity failed (singleton group) in {json_path.name}"
                logger.error(msg)
                if self.config.fail_fast:
                    raise ValueError(msg)
                invalid_sample = {
                    "sample_id": str(json_path.name),
                    "reason": "group_singleton",
                    "image_path": str(image_path),
                    "json_path": str(json_path),
                }
                self.invalid_samples.append(invalid_sample)
                return None

            # 输出前去掉 groups 字段，组信息已写入 desc
            for obj in objects:
                obj.pop("groups", None)

            # Sort objects unless compatibility mode requests preserving the original order
            if not getattr(self.config, "preserve_annotation_order", False):
                objects = sort_objects_tlbr(objects)
            else:
                logger.debug(
                    "Preserving legacy annotation order for %s", image_path.name
                )

            # Detection desc 后处理：将矛盾/不确定的标注改写为 “<type>/需复核”
            objects = flag_objects_for_review(objects)

            # Summary 直接继承处理后的 desc（含需复核）
            if self.is_rru:
                summary_text = self._build_simple_summary(objects)
            else:
                summary_objects = [obj.copy() for obj in objects]
                summary_text = build_summary_from_objects(summary_objects)

            # Process image (copy/resize) to match coordinate transformations
            processed_image_path, img_w, img_h = self.image_processor.process_image(
                image_path,
                json_width,
                json_height,
                final_width=final_width,
                final_height=final_height,
            )

            # Fail fast if image and geometry pipelines disagree on final size
            if (img_w, img_h) != (final_width, final_height):
                msg = (
                    f"ImageProcessor produced size {img_w}x{img_h} but "
                    f"CoordinateManager reported {final_width}x{final_height} "
                    f"for {image_path.name}"
                )
                logger.error(msg)
                if self.config.fail_fast:
                    raise ValueError(msg)

            # Build relative image path for JSONL (relative to dataset directory)
            rel_image_path = self.image_processor.get_relative_image_path(
                processed_image_path
            )

            return {
                "images": [rel_image_path],
                "objects": objects,
                "summary": summary_text,
                "width": final_width,
                "height": final_height,
            }

        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
            if self.config.fail_fast:
                raise
            return None

    def _filter_valid_objects(
        self, objects: List[Dict], img_w: int, img_h: int, image_id: str
    ) -> List[Dict]:
        """Filter objects using strict validation with reporting.

        This uses ValidationManager to validate each object and records
        any invalid objects for downstream analysis.
        """
        if not objects:
            return objects

        # Re-sanitize descriptions right before validation/output, in case any slipped through
        if (
            getattr(self.config, "remove_occlusion_tokens", False)
            or getattr(self.config, "sanitize_text", False)
            or getattr(self.config, "standardize_label_desc", False)
        ):
            for obj in objects:
                d = obj.get("desc", "")
                if d:
                    obj["desc"] = self._sanitize_description(d)

        # Validate objects via ValidationManager and track invalid ones
        valid_objects, invalid_objects = self.validation_manager.filter_valid_objects(
            objects,
            image_width=img_w,
            image_height=img_h,
            sample_id=image_id,
        )

        if invalid_objects:
            self.invalid_objects.extend(invalid_objects)
            logger.warning(
                "Filtered out %d invalid objects for image %s",
                len(invalid_objects),
                image_id,
            )

        return valid_objects

    def _process_sample_coordinates_unified(
        self,
        sample_data: Dict,
        image_path: Path,
        json_width: int,
        json_height: int,
        enable_smart_resize: bool = True,
    ) -> Tuple[Dict, int, int]:
        """
        Process sample coordinates using unified geometry transformation.

        This replaces the old bbox-only processing with geometry-aware processing
        that works with both simple bbox and complex geometries.
        """
        if "objects" not in sample_data or not sample_data["objects"]:
            # No objects to process, just get final dimensions
            if enable_smart_resize:
                from data_conversion.pipeline.vision_process import MIN_PIXELS
                from data_conversion.pipeline.vision_process import smart_resize

                _, _, _, final_w, final_h, _ = CoordinateManager.get_exif_transform_matrix(
                    image_path
                )

                resize_h, resize_w = smart_resize(
                    height=final_h,
                    width=final_w,
                    factor=self.config.image_factor,
                    min_pixels=MIN_PIXELS,
                    max_pixels=self.config.max_pixels,
                )
                return sample_data, resize_w, resize_h
            else:
                _, _, _, final_w, final_h, _ = CoordinateManager.get_exif_transform_matrix(
                    image_path
                )
                return sample_data, final_w, final_h

        # Process first object to get final dimensions
        first_obj = sample_data["objects"][0]

        # Get any coordinate for dimension calculation
        if "bbox_2d" in first_obj:
            geometry_input = first_obj["bbox_2d"]
        elif "poly" in first_obj:
            # Create bbox from poly for dimension calculation
            poly = first_obj["poly"]
            x_coords = [poly[i] for i in range(0, len(poly), 2)]
            y_coords = [poly[i] for i in range(1, len(poly), 2)]
            geometry_input = [
                min(x_coords),
                min(y_coords),
                max(x_coords),
                max(y_coords),
            ]
        elif "line" in first_obj:
            # Create bbox from line for dimension calculation
            line = first_obj["line"]
            x_coords = [line[i] for i in range(0, len(line), 2)]
            y_coords = [line[i] for i in range(1, len(line), 2)]
            geometry_input = [
                min(x_coords),
                min(y_coords),
                max(x_coords),
                max(y_coords),
            ]
        else:
            raise ValueError(f"No supported geometry type in first object: {first_obj}")

        # Use unified geometry transformation for dimension calculation
        _, _, final_width, final_height = CoordinateManager.transform_geometry_complete(
            geometry_input,
            image_path,
            json_width,
            json_height,
            smart_resize_factor=self.config.image_factor,
            max_pixels=self.config.max_pixels,
            enable_smart_resize=enable_smart_resize,
        )

        # Process all objects with their native geometry
        updated_objects = []
        for obj in sample_data["objects"]:
            # Transform coordinates based on geometry type
            updated_obj = obj.copy()

            if "bbox_2d" in obj:
                # Transform bbox coordinates
                transformed_coords = self._transform_coordinates(
                    obj["bbox_2d"],
                    image_path,
                    json_width,
                    json_height,
                    enable_smart_resize,
                )
                updated_obj["bbox_2d"] = [int(round(c)) for c in transformed_coords]

            elif "poly" in obj:
                # Transform poly coordinates (sequence of x,y pairs)
                # Strip any closing/duplicated point
                coords = obj["poly"]
                # Check if already closed - if so, remove closing point before transformation
                is_closed = (
                    len(coords) >= 4
                    and coords[0] == coords[-2]
                    and coords[1] == coords[-1]
                )
                coords_to_transform = coords[:-2] if is_closed else coords

                transformed_coords = []
                for i in range(0, len(coords_to_transform), 2):
                    if i + 1 < len(coords_to_transform):
                        point = [coords_to_transform[i], coords_to_transform[i + 1]]
                        transformed_point = self._transform_coordinates(
                            point,
                            image_path,
                            json_width,
                            json_height,
                            enable_smart_resize,
                            is_point=True,
                        )
                        transformed_coords.extend(
                            [
                                int(round(transformed_point[0])),
                                int(round(transformed_point[1])),
                            ]
                        )

                # Re-apply canonical ordering after transformation to ensure top-left start
                # This matches the prompt specification: "poly 排序参考点：使用第一个顶点 (x1, y1)"
                if len(transformed_coords) >= 8 and len(transformed_coords) % 2 == 0:
                    points_list = [
                        (transformed_coords[i], transformed_coords[i + 1])
                        for i in range(0, len(transformed_coords), 2)
                    ]
                    ordered_points = CoordinateManager.canonical_poly_ordering(
                        points_list
                    )
                    # Flatten back to coordinate list (without closing point)
                    transformed_coords = [
                        int(coord) for point in ordered_points for coord in point
                    ]

                # Store canonicalized coordinates without duplicate closing point
                updated_obj["poly"] = transformed_coords

            elif "line" in obj:
                # Transform line coordinates (sequence of x,y pairs)
                coords = obj["line"]
                transformed_coords = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        point = [coords[i], coords[i + 1]]
                        transformed_point = self._transform_coordinates(
                            point,
                            image_path,
                            json_width,
                            json_height,
                            enable_smart_resize,
                            is_point=True,
                        )
                        transformed_coords.extend(
                            [
                                int(round(transformed_point[0])),
                                int(round(transformed_point[1])),
                            ]
                        )
                updated_obj["line"] = transformed_coords

            # Apply coordinate normalization after transformation
            normalized_obj = CoordinateManager.normalize_object_coordinates(
                updated_obj, final_width, final_height
            )
            updated_objects.append(normalized_obj)

        updated_sample = sample_data.copy()
        updated_sample["objects"] = updated_objects

        return updated_sample, final_width, final_height

    def _transform_coordinates(
        self,
        coords,
        image_path,
        json_width,
        json_height,
        enable_smart_resize,
        is_point=False,
    ):
        """Simple coordinate transformation for any coordinate format."""
        # Use CoordinateManager for the transformation
        if is_point:
            # For single points, create a minimal bbox and extract the transformed point
            bbox = [coords[0], coords[1], coords[0], coords[1]]
            transformed_bbox, _, _, _ = CoordinateManager.transform_geometry_complete(
                bbox,
                image_path,
                json_width,
                json_height,
                smart_resize_factor=self.config.image_factor,
                max_pixels=self.config.max_pixels,
                enable_smart_resize=enable_smart_resize,
            )
            return [transformed_bbox[0], transformed_bbox[1]]  # Return just x, y
        else:
            # For bbox format
            transformed_bbox, _, _, _ = CoordinateManager.transform_geometry_complete(
                coords,
                image_path,
                json_width,
                json_height,
                smart_resize_factor=self.config.image_factor,
                max_pixels=self.config.max_pixels,
                enable_smart_resize=enable_smart_resize,
            )
            return transformed_bbox

    def process_all_samples(self) -> List[Dict]:
        """Process all samples in the input directory."""
        logger.info("🚀 Starting sample processing")

        # Find all JSON files
        json_files = FileOperations.find_json_files(self.input_dir)
        logger.info(f"📁 Found {len(json_files)} JSON files")

        # Apply limit if specified (for debugging)
        if self.config.limit > 0:
            original_count = len(json_files)
            json_files = json_files[: self.config.limit]
            logger.info(
                f"🔢 Limiting processing to {len(json_files)} images (out of {original_count} total)"
            )

        # Determine number of workers
        num_workers = getattr(self.config, "num_workers", 1)

        # Use parallel processing if num_workers > 1
        if num_workers > 1:
            return self._process_samples_parallel(json_files, num_workers)
        else:
            return self._process_samples_sequential(json_files)

    def _process_samples_sequential(self, json_files: List[Path]) -> List[Dict]:
        """Process samples sequentially (original implementation)."""
        logger.info("📝 Processing samples sequentially (num_workers=1)")

        # Process all samples
        all_samples: List[Dict] = []
        processed_count = 0
        skipped_count = 0

        # Initialize progress bar if tqdm is available
        pbar = None
        if tqdm is not None:
            pbar = tqdm(total=len(json_files), desc="Processing samples", unit="sample")
        else:
            logger.info(f"Processing {len(json_files)} samples...")

        try:
            for json_file in json_files:
                sample = self.process_single_sample(json_file)
                if not sample:
                    skipped_count += 1
                    if pbar is not None:
                        pbar.update(1)
                    continue

                # Validate full sample structure/content via ValidationManager
                sample_id = str(json_file.name)
                image_width = sample.get("width")
                image_height = sample.get("height")

                is_valid, report = self.validation_manager.validate_sample(
                    sample,
                    sample_id=sample_id,
                    image_width=image_width,
                    image_height=image_height,
                )

                if not is_valid:
                    invalid_sample = {
                        "sample_id": sample_id,
                        "reason": "validation_failed",
                        "image_path": sample.get("images", [None])[0],
                        "json_path": str(json_file),
                        "validation_errors": [e.to_dict() for e in report.errors],
                    }
                    self.invalid_samples.append(invalid_sample)

                    message = (
                        f"Sample {sample_id} failed validation with "
                        f"{len(report.errors)} errors; skipping from outputs"
                    )

                    if getattr(self.config, "fail_fast", True):
                        raise ValueError(message)

                    logger.warning(message)
                    skipped_count += 1
                    if pbar is not None:
                        pbar.update(1)
                    continue

                all_samples.append(sample)
                processed_count += 1

                if pbar is not None:
                    pbar.update(1)
                    # Update progress bar description with current stats
                    pbar.set_postfix(
                        {"processed": processed_count, "skipped": skipped_count}
                    )
        finally:
            if pbar is not None:
                pbar.close()

        logger.info(
            f"✅ Sample processing complete: {processed_count} processed, {skipped_count} skipped"
        )

        if not all_samples:
            raise ValueError("No valid samples were processed")

        return all_samples

    def _process_samples_parallel(self, json_files: List[Path], num_workers: int) -> List[Dict]:
        """Process samples in parallel using multiprocessing."""
        # Limit workers to available CPU cores
        max_workers = min(num_workers, cpu_count(), len(json_files))
        logger.info(f"🚀 Processing samples in parallel with {max_workers} workers")
        logger.info("ℹ️  Worker initialization logs suppressed (only warnings/errors shown)")

        # Process samples in parallel
        all_samples: List[Dict] = []
        processed_count = 0
        skipped_count = 0

        try:
            # Create pool with initializer to set up each worker once
            with Pool(
                processes=max_workers,
                initializer=_init_worker,
                initargs=(self.config, self.label_hierarchy)
            ) as pool:
                # Use ordered imap to preserve input ordering for deterministic outputs
                if tqdm is not None:
                    results = list(tqdm(
                        pool.imap(_process_sample_worker, json_files),
                        total=len(json_files),
                        desc=f"Processing samples ({max_workers} workers)",
                        unit="sample"
                    ))
                else:
                    logger.info(f"Processing {len(json_files)} samples with {max_workers} workers...")
                    results = list(pool.imap(_process_sample_worker, json_files))

            # Post-process results: validate and filter
            logger.info("📋 Validating processed samples...")
            for i, sample in enumerate(results):
                if not sample:
                    skipped_count += 1
                    continue

                # Validate full sample structure/content via ValidationManager
                sample_id = sample.get("images", [f"sample_{i}"])[0]
                image_width = sample.get("width")
                image_height = sample.get("height")

                is_valid, report = self.validation_manager.validate_sample(
                    sample,
                    sample_id=sample_id,
                    image_width=image_width,
                    image_height=image_height,
                )

                if not is_valid:
                    invalid_sample = {
                        "sample_id": sample_id,
                        "reason": "validation_failed",
                        "image_path": sample.get("images", [None])[0],
                        "validation_errors": [e.to_dict() for e in report.errors],
                    }
                    self.invalid_samples.append(invalid_sample)

                    message = (
                        f"Sample {sample_id} failed validation with "
                        f"{len(report.errors)} errors; skipping from outputs"
                    )

                    if getattr(self.config, "fail_fast", True):
                        raise ValueError(message)

                    logger.warning(message)
                    skipped_count += 1
                    continue

                all_samples.append(sample)
                processed_count += 1

        except Exception as e:
            logger.error(f"Error during parallel processing: {e}")
            raise

        logger.info(
            f"✅ Sample processing complete: {processed_count} processed, {skipped_count} skipped"
        )

        if not all_samples:
            raise ValueError("No valid samples were processed")

        return all_samples

    def split_into_sets(
        self, all_samples: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split samples into training and validation sets.

        Teacher pool selection has been removed; all samples are used for train/val.

        Args:
            all_samples: List of all processed samples

        Returns:
            Tuple of (train_samples, val_samples)
        """
        logger.info(f"📊 Splitting {len(all_samples)} samples into train/val...")

        # Split all samples into train and validation sets
        train_samples, val_samples = self.data_splitter.split(all_samples)

        logger.info(
            f"✅ Split complete: {len(train_samples)} train, {len(val_samples)} val samples"
        )

        return train_samples, val_samples

    def write_outputs(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
    ) -> None:
        """Write output files in flat format only.

        All files are written in the same flat format: {'images': [...], 'objects': [...]}
        No nested teacher-student structure is used.
        """
        logger.info("📾 Writing output files...")

        # Write all files in flat format (no teacher-student nesting)
        FileOperations.write_jsonl(train_samples, self.output_dir / "train.jsonl")
        FileOperations.write_jsonl(val_samples, self.output_dir / "val.jsonl")

        # Write all samples in flat format
        all_direct_samples = train_samples + val_samples
        FileOperations.write_jsonl(
            all_direct_samples, self.output_dir / "all_samples.jsonl"
        )

        # Extract and export unique labels from original samples
        self._export_label_vocabulary(all_direct_samples)

        # Export validation reports
        self._export_validation_reports()

        logger.info("📊 Output files written successfully")
        logger.info(
            f"   📚 All samples (flat format): all_samples.jsonl ({len(all_direct_samples)} samples)"
        )
        logger.info(
            f"   🎓 Training files (flat format): train.jsonl ({len(train_samples)} samples), val.jsonl ({len(val_samples)} samples)"
        )

    def _convert_to_teacher_student_format(
        self, student_samples: List[Dict], teacher_pool: List[Dict]
    ) -> List[Dict]:
        """DEPRECATED: No longer needed as we use flat format only.

        This method is kept for backward compatibility but now simply returns
        the student samples without any transformation to teacher-student format.

        Args:
            student_samples: List of samples to use as students
            teacher_pool: Pool of teacher samples (unused)

        Returns:
            List of samples in flat format (unchanged)
        """
        # Simply return the samples without any transformation
        # This ensures all files use the flat format
        return student_samples.copy()

    def _export_label_vocabulary(self, all_samples: List[Dict]) -> None:
        """Extract and export unique labels from all samples."""
        unique_labels = set()
        object_types = set()
        properties = set()
        full_descriptions = set()

        # Extract labels from all samples
        for sample in all_samples:
            for obj in sample.get("objects", []):
                desc = obj.get("desc", "")
                if desc:
                    full_descriptions.add(desc)

                    # Parse description to extract components
                    components = FormatConverter.parse_description_string(desc)

                    obj_type = components.get("object_type", "").strip()
                    prop = components.get("property", "").strip()
                    extra = components.get("extra_info", "").strip()

                    if obj_type:
                        object_types.add(obj_type)
                        unique_labels.add(obj_type)

                    if prop:
                        properties.add(prop)
                        unique_labels.add(prop)

                    if extra:
                        unique_labels.add(extra)

        # Create comprehensive label vocabulary
        label_vocabulary = {
            "metadata": {
                "total_samples": len(all_samples),
                "total_objects": sum(
                    len(sample.get("objects", [])) for sample in all_samples
                ),
                "language": "chinese",
                "extraction_date": self._get_current_timestamp(),
                "description": "Complete vocabulary of labels extracted from the dataset for training prompt enhancement",
            },
            "statistics": {
                "unique_labels_count": len(unique_labels),
                "object_types_count": len(object_types),
                "properties_count": len(properties),
                "full_descriptions_count": len(full_descriptions),
            },
            "vocabulary": {
                "all_unique_labels": sorted(list(unique_labels)),
                "object_types": sorted(list(object_types)),
                "properties": sorted(list(properties)),
                "full_descriptions": sorted(list(full_descriptions)),
            },
            "usage_notes": {
                "training_prompts": "Use 'all_unique_labels' for comprehensive label-aware training",
                "object_detection": "Use 'object_types' for class-specific detection tasks",
                "attribute_prediction": "Use 'properties' for attribute/property prediction",
                "full_context": "Use 'full_descriptions' for complete description generation",
            },
        }

        # Export to JSON file
        output_path = self.output_dir / "label_vocabulary.json"
        FileOperations.save_json_data(label_vocabulary, output_path, indent=2)

        logger.info(f"📋 Label vocabulary exported to {output_path}")
        logger.info(f"   📊 {len(unique_labels)} unique labels")
        logger.info(f"   🔖 {len(object_types)} object types")
        logger.info(f"   🏷️  {len(properties)} properties")
        logger.info(f"   📝 {len(full_descriptions)} complete descriptions")

    def _export_validation_reports(self) -> None:
        """Export validation reports and legacy invalid object/sample lists."""
        # Structured reports from ValidationManager (strict validation).
        if getattr(self, "validation_manager", None) is not None:
            if not self.enable_validation_reports:
                logger.info("📋 Validation report export disabled via configuration")
            elif self.validation_manager.validation_reports:
                report_files = self.validation_manager.export_validation_reports(
                    self.output_dir
                )
                summary = ", ".join(
                    f"{name}={path}" for name, path in report_files.items()
                )
                logger.info(f"📋 Validation reports exported: {summary}")
            else:
                logger.info(
                    "📋 No validation issues recorded; skipping structured reports"
                )
        else:
            logger.info(
                "📋 ValidationManager is not initialized; skipping structured reports"
            )

        # Legacy JSONL exports of invalid objects/samples for backward compatibility.
        if self.invalid_objects:
            invalid_objects_path = self.output_dir / "invalid_objects.jsonl"
            FileOperations.write_jsonl(self.invalid_objects, invalid_objects_path)
            logger.warning(
                f"📋 Exported {len(self.invalid_objects)} invalid objects to {invalid_objects_path}"
            )

        if self.invalid_samples:
            invalid_samples_path = self.output_dir / "invalid_samples.jsonl"
            FileOperations.write_jsonl(self.invalid_samples, invalid_samples_path)
            logger.warning(
                f"📋 Exported {len(self.invalid_samples)} invalid samples to {invalid_samples_path}"
            )

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()

    def process(self) -> Dict[str, int]:
        """
        Execute the complete unified processing pipeline.

        Returns:
            Dictionary with processing statistics
        """
        logger.info("🚀 Starting unified data processing pipeline")

        # Step 1: Process all samples
        all_samples = self.process_all_samples()

        # Step 2: Split into train/val sets
        train_samples, val_samples = self.split_into_sets(all_samples)

        # Step 3: Validate output structure
        StructureValidator.validate_pipeline_output(
            train_samples,
            val_samples,
        )

        # Step 4: Write output files
        self.write_outputs(train_samples, val_samples)

        # Final summary statistics
        result = {
            "train": len(train_samples),
            "val": len(val_samples),
            "total_processed": len(all_samples),
            "total_invalid_objects": len(self.invalid_objects),
            "total_invalid_samples": len(self.invalid_samples),
        }

        logger.info("🎉 Pipeline completed successfully!")
        logger.info("📊 Final Output:")
        logger.info(f"   Training: {result['train']} samples → train.jsonl")
        logger.info(f"   Validation: {result['val']} samples → val.jsonl")
        logger.info(
            f"   Combined: {result['total_processed']} samples → all_samples.jsonl"
        )
        logger.info(
            f"   Combined: {result['total_processed']} samples → all_samples.jsonl"
        )
        if self.invalid_objects or self.invalid_samples:
            logger.info("🔍 Processing Summary:")
            logger.info(
                f"   Invalid objects tracked: {result['total_invalid_objects']}"
            )
            logger.info(
                f"   Invalid samples tracked: {result['total_invalid_samples']}"
            )

        return result

    def _build_simple_summary(self, objects: List[Dict]) -> str:
        """
        RRU 汇总：按 desc 完整字符串聚合计数（包含组/备注等全部信息）。
        形式：desc×N，使用中文逗号分隔；N=1 时不附 ×N。
        """
        if not objects:
            return "无关图片"
        counts = {}
        for obj in objects:
            desc = obj.get("desc", "") or ""
            desc = desc.strip()
            if desc:
                counts[desc] = counts.get(desc, 0) + 1
        if not counts:
            return "无关图片"
        parts = []
        for desc, n in counts.items():
            if n == 1:
                parts.append(desc)
            else:
                parts.append(f"{desc}×{n}")
        return "，".join(parts)


# TeacherSelector has been extracted to data_conversion.teacher_selector


def main():
    """Main entry point with CLI argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Data Processor for Qwen2.5-VL")

    # Required arguments
    parser.add_argument(
        "--input_dir", required=True, help="Input directory with JSON/image files"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--dataset_name",
        help="Dataset name for organized output (auto-detected from input_dir if not provided)",
    )

    # Processing arguments - REQUIRED
    parser.add_argument(
        "--val_ratio",
        type=float,
        required=True,
        help="Validation split ratio (e.g., 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed (e.g., 42)"
    )

    # Image resize parameters - REQUIRED
    parser.add_argument(
        "--max_pixels",
        type=int,
        required=True,
        help="Maximum pixels for image resizing (e.g., 786432 for 768*32*32)",
    )
    parser.add_argument(
        "--image_factor",
        type=int,
        required=True,
        help="Factor for image dimensions (e.g., 32)",
    )

    # Processing options - OPTIONAL
    parser.add_argument("--resize", action="store_true", help="Enable image resizing")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--fail_fast",
        dest="fail_fast",
        action="store_true",
        help="Stop immediately when encountering invalid samples",
    )
    parser.add_argument(
        "--no_fail_fast",
        dest="fail_fast",
        action="store_false",
        help="Continue processing after invalid samples are detected (skip them)",
    )
    parser.set_defaults(fail_fast=True)


    # New filtering flag
    parser.add_argument(
        "--strip_occlusion",
        action="store_true",
        help="Remove tokens containing '遮挡' from descriptions",
    )
    parser.add_argument(
        "--sanitize_text",
        action="store_true",
        help="Apply text normalization sanitization (spaces, hyphens, fullwidth digits, circled numbers)",
    )
    parser.add_argument(
        "--standardize_label_desc",
        action="store_true",
        help="Standardize label descriptions: map '标签/*' empty-like values (空格/看不清/、 or empty) to '标签/无法识别'",
    )
    parser.add_argument(
        "--preserve_annotation_order",
        action="store_true",
        help="Skip TLBR reordering and keep the original annotation order (useful for regression diffs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit number of images to process for debugging (e.g., 10 for 10 images, -1 for all images)",
    )
    parser.add_argument(
        "--validation_mode",
        choices=["strict", "lenient", "warning_only"],
        required=True,
        help="Validation mode controlling strictness of geometry/text checks",
    )
    parser.add_argument(
        "--min_object_size",
        type=int,
        required=True,
        help="Minimum bbox width/height in pixels considered valid",
    )
    parser.add_argument(
        "--enable_validation_reports",
        dest="enable_validation_reports",
        action="store_true",
        help="Enable exporting validation reports to disk",
    )
    parser.add_argument(
        "--disable_validation_reports",
        dest="enable_validation_reports",
        action="store_false",
        help="Disable exporting validation reports to disk",
    )
    parser.set_defaults(enable_validation_reports=True)

    # Performance options
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for multiprocessing (1=sequential, >1=parallel). Default: 1",
    )

    args = parser.parse_args()

    # Create configuration from arguments
    config = DataConversionConfig.from_args(args)

    # Setup logging
    setup_logging(config)

    # Validate configuration
    validate_config(config)

    # Create and run unified processor
    processor = UnifiedProcessor(config)
    result = processor.process()

    # Print result for compatibility
    print(f"\n✅ Processing complete: {result}")


if __name__ == "__main__":
    main()
