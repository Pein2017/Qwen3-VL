#!/usr/bin/env python3
"""
Streamlined Unified Data Processor

Consolidates all data processing functionality with simplified architecture.
Merged SampleExtractor directly into UnifiedProcessor to eliminate redundancy.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# TeacherSelector now integrated as nested class
from data_conversion.config import DataConversionConfig
from data_conversion.constants import DEFAULT_LABEL_HIERARCHY
from data_conversion.coordinate_manager import (
    CoordinateManager,
    DataValidator,
    FormatConverter,
    StructureValidator,
)
from data_conversion.data_splitter import DataSplitter
from data_conversion.flexible_taxonomy_processor import HierarchicalProcessor
from data_conversion.summary_builder import build_summary_from_objects
from data_conversion.teacher_selector import TeacherSelector
from data_conversion.utils.file_ops import FileOperations
from data_conversion.utils.sanitizers import (
    sanitize_text,
    standardize_label_description,
    strip_occlusion_tokens,
)
from data_conversion.utils.sorting import sort_objects_tlbr
from data_conversion.vision_process import ImageProcessor


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


class UnifiedProcessor:
    """Streamlined orchestrator for the unified data processing pipeline."""

    def __init__(self, config: DataConversionConfig):
        """Initialize processor with configuration."""
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.output_dir = config.get_dataset_output_dir()

        # Initialize components - no token mapping needed for Chinese-only

        # Load label hierarchy or use default
        if config.hierarchy_path and Path(config.hierarchy_path).exists():
            self.label_hierarchy = FileOperations.load_label_hierarchy(
                Path(config.hierarchy_path)
            )
        else:
            # Default hierarchy matching actual v2 data structure
            self.label_hierarchy = DEFAULT_LABEL_HIERARCHY

        # Initialize hierarchical processor for v2 data support (Chinese only)
        self.hierarchical_processor = HierarchicalProcessor(
            object_types=set(config.object_types),
            label_hierarchy=self.label_hierarchy,
        )

        self.image_processor = ImageProcessor(config)
        self.teacher_selector = TeacherSelector(
            label_hierarchy=self.label_hierarchy,
            allowed_object_types=set(config.object_types),
            max_teachers=config.max_teachers,
            seed=config.seed,
        )
        self.data_splitter = DataSplitter(val_ratio=config.val_ratio, seed=config.seed)

        # Track invalid objects and samples for reporting (for legacy compatibility)
        self.invalid_objects = []
        self.invalid_samples = []

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
        # Apply built-in sanitizers based on flags
        s = desc
        if getattr(self.config, "sanitize_text", False):
            s = sanitize_text(s) or s
            # Also remove annotator notes like '框选范围*'
            try:
                from data_conversion.utils.sanitizers import strip_annotator_notes

                s = strip_annotator_notes(s) or s
            except Exception:
                # Best effort; do not fail pipeline on optional sanitization
                pass
        if getattr(self.config, "remove_occlusion_tokens", False):
            s = strip_occlusion_tokens(s) or s
        # Standardize 标签 descriptions to eliminate empty-like cases
        if getattr(self.config, "standardize_label_desc", False):
            try:
                s = standardize_label_description(s) or s
            except Exception:
                pass
        # Remove completeness attributes from screw/fiber connector objects
        # This is always applied (not behind a config flag) as it's a data quality fix
        try:
            from data_conversion.utils.sanitizers import remove_screw_completeness_attributes

            s = remove_screw_completeness_attributes(s) or s
        except Exception:
            # Best effort; do not fail pipeline on optional sanitization
            pass
        return s

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
            if obj.startswith("BBU设备") and any("机柜空间充足需要安装" in t for t in l1_tokens):
                structured_count = min(2, len(levels))
            elif obj.startswith("螺丝、光纤插头") and any("不符合要求" in t for t in l1_tokens):
                structured_count = min(2, len(levels))
            elif obj.startswith("光纤") and any("有保护措施" in t for t in l1_tokens):
                structured_count = min(2, len(levels))
            elif obj.startswith("挡风板") or obj.startswith("电线"):
                structured_count = min(1, len(levels))

            # 如果存在额外层，视为备注层（最后一层）
            if len(levels) > structured_count:
                remark = levels[-1]
                base = "/".join([obj] + levels[:structured_count]) if structured_count > 0 else obj
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

            desc = FormatConverter.format_description(
                content_dict, self.config.response_types, "chinese"
            )
            if desc:
                if getattr(self.config, "remove_occlusion_tokens", False) or getattr(self.config, "sanitize_text", False) or getattr(self.config, "standardize_label_desc", False):
                    desc = self._sanitize_description(desc)
                if desc:
                    desc = self._rewrite_desc_with_remark(desc)
                    objects.append({"bbox_2d": bbox, "desc": desc})

        return objects

    def extract_objects_from_markresult(self, features: List[Dict]) -> List[Dict]:
        """Extract objects from markResult features with native geometry types."""
        # Use hierarchical processor for V2 data support
        objects = self.hierarchical_processor.extract_objects_from_markresult(features)
        # Apply sanitizer if configured
        if getattr(self.config, "remove_occlusion_tokens", False) or getattr(self.config, "sanitize_text", False) or getattr(self.config, "standardize_label_desc", False):
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

            # Detect dimension mismatch (likely due to EXIF orientation)
            if json_width != actual_width or json_height != actual_height:
                logger.info(
                    f"Dimension mismatch for {image_path.name}: "
                    f"JSON says {json_width}x{json_height} but image is {actual_width}x{actual_height}. "
                    f"Will apply coordinate rescaling."
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
                objects = self.extract_objects_from_markresult(
                    json_data["markResult"]["features"]
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

            # Process objects (validation step has been removed)
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

            # Sort objects by position using first coordinate pair
            objects = sort_objects_tlbr(objects)

            # Build deterministic one-line summary from objects
            summary_text = build_summary_from_objects(objects)

            # Process image (copy/resize) to match coordinate transformations
            processed_image_path, _, _ = self.image_processor.process_image(
                image_path, json_width, json_height
            )

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
        """Filter objects using comprehensive validation with reporting."""
        if not objects:
            return objects

        # Re-sanitize descriptions right before validation/output, in case any slipped through
        if getattr(self.config, "remove_occlusion_tokens", False) or getattr(self.config, "sanitize_text", False) or getattr(self.config, "standardize_label_desc", False):
            for obj in objects:
                d = obj.get("desc", "")
                if d:
                    obj["desc"] = self._sanitize_description(d)

        # For trusted input, no validation needed - return all objects
        return objects

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
                from data_conversion.vision_process import (
                    MAX_PIXELS,
                    MIN_PIXELS,
                    smart_resize,
                )

                _, _, _, final_w, final_h = CoordinateManager.get_exif_transform_matrix(
                    image_path
                )
                resize_h, resize_w = smart_resize(
                    height=final_h,
                    width=final_w,
                    factor=28,
                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS,
                )
                return sample_data, resize_w, resize_h
            else:
                _, _, _, final_w, final_h = CoordinateManager.get_exif_transform_matrix(
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
            geometry_input, image_path, json_width, json_height, enable_smart_resize
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
                # Transform poly coordinates (x1,y1,x2,y2,x3,y3,x4,y4,...)
                # Note: poly format should be closed (first point repeated at end)
                coords = obj["poly"]
                # Check if already closed - if so, remove closing point before transformation
                is_closed = len(coords) >= 4 and coords[0] == coords[-2] and coords[1] == coords[-1]
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
                # Ensure polygon is closed
                if len(transformed_coords) >= 4:
                    transformed_coords.extend([transformed_coords[0], transformed_coords[1]])
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
                bbox, image_path, json_width, json_height, enable_smart_resize
            )
            return [transformed_bbox[0], transformed_bbox[1]]  # Return just x, y
        else:
            # For bbox format
            transformed_bbox, _, _, _ = CoordinateManager.transform_geometry_complete(
                coords, image_path, json_width, json_height, enable_smart_resize
            )
            return transformed_bbox

    def process_all_samples(self) -> List[Dict]:
        """Process all samples in the input directory."""
        logger.info("🚀 Starting sample processing")

        # Find all JSON files
        json_files = FileOperations.find_json_files(self.input_dir)
        logger.info(f"📁 Found {len(json_files)} JSON files")

        # Process all samples
        all_samples = []
        processed_count = 0
        skipped_count = 0

        for json_file in json_files:
            sample = self.process_single_sample(json_file)
            if sample:
                # Validate sample structure
                DataValidator.validate_sample_structure(sample)
                all_samples.append(sample)
                processed_count += 1
            else:
                skipped_count += 1

            if processed_count % 100 == 0 and processed_count > 0:
                logger.info(f"Processed {processed_count} samples...")

        logger.info(
            f"✅ Sample processing complete: {processed_count} processed, {skipped_count} skipped"
        )

        if not all_samples:
            raise ValueError("No valid samples were processed")

        return all_samples

    def split_into_sets(
        self, all_samples: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split samples into training, validation, and teacher sets.

        All samples remain in flat format without any teacher-student nesting.

        Args:
            all_samples: List of all processed samples

        Returns:
            Tuple of (train_samples, val_samples, teacher_samples)
        """
        logger.info(f"📊 Splitting {len(all_samples)} samples...")

        # Create data splitter
        data_splitter = DataSplitter(
            val_ratio=self.config.val_ratio, seed=self.config.seed
        )

        # Select teacher samples first (before train/val split)
        teacher_selector = self.teacher_selector
        teacher_samples, teacher_indices, teacher_stats = (
            teacher_selector.select_teachers(all_samples)
        )

        # Remove teacher samples from the pool
        remaining_samples = [
            s for i, s in enumerate(all_samples) if i not in teacher_indices
        ]

        # Split remaining samples into train and validation sets
        train_samples, val_samples = data_splitter.split(remaining_samples)

        logger.info(
            f"✅ Split complete: {len(train_samples)} train, {len(val_samples)} val, {len(teacher_samples)} teacher samples"
        )

        # Store teacher stats for later export
        self._teacher_pool_stats = teacher_stats

        return train_samples, val_samples, teacher_samples

    def write_outputs(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        teacher_samples: List[Dict],
    ) -> None:
        """Write output files in flat format only.

        All files are written in the same flat format: {'images': [...], 'objects': [...]}
        No nested teacher-student structure is used.
        """
        logger.info("📾 Writing output files...")

        # Write all files in flat format (no teacher-student nesting)
        FileOperations.write_jsonl(train_samples, self.output_dir / "train.jsonl")
        FileOperations.write_jsonl(val_samples, self.output_dir / "val.jsonl")
        FileOperations.write_jsonl(
            teacher_samples, self.output_dir / "teacher_pool.jsonl"
        )

        # Write all samples in flat format
        all_direct_samples = teacher_samples + train_samples + val_samples
        FileOperations.write_jsonl(
            all_direct_samples, self.output_dir / "all_samples.jsonl"
        )

        # Extract and export unique labels from original samples
        self._export_label_vocabulary(all_direct_samples)

        # Write teacher pool stats if available
        stats = getattr(self, "_teacher_pool_stats", None)
        if isinstance(stats, dict) and stats:
            stats_path = self.output_dir / "teacher_pool_stats.json"
            FileOperations.save_json_data(stats, stats_path, indent=2)
            logger.info(f"📈 Teacher pool stats written to {stats_path}")

        # Export validation reports
        self._export_validation_reports()

        logger.info("📊 Output files written successfully")
        logger.info(
            f"   📚 All samples (flat format): all_samples.jsonl ({len(all_direct_samples)} samples)"
        )
        logger.info(
            f"   🎓 Training files (flat format): train.jsonl ({len(train_samples)} samples), val.jsonl ({len(val_samples)} samples)"
        )
        logger.info(
            f"   📚 Teacher pool (flat format): teacher_pool.jsonl ({len(teacher_samples)} samples)"
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
        """Export validation reports (legacy compatibility - no validation for trusted input)."""
        logger.info("📋 Validation reports skipped (trusted input mode)")

        # Export invalid objects/samples if any were tracked (should be empty for trusted input)
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

        # Step 2: Split into train/val/teacher sets
        train_samples, val_samples, teacher_samples = self.split_into_sets(all_samples)

        # Step 3: Validate output structure
        StructureValidator.validate_pipeline_output(
            train_samples, val_samples, teacher_samples, max_teachers=self.config.max_teachers
        )

        # Step 4: Write output files
        self.write_outputs(train_samples, val_samples, teacher_samples)

        # Validation steps have been removed

        # Final summary statistics
        result = {
            "train": len(train_samples),
            "val": len(val_samples),
            "teacher": len(teacher_samples),
            "total_processed": len(all_samples),
            "total_invalid_objects": len(self.invalid_objects),
            "total_invalid_samples": len(self.invalid_samples),
        }

        logger.info("🎉 Pipeline completed successfully!")
        logger.info("📊 Final Output:")
        logger.info(f"   Training: {result['train']} samples → train.jsonl")
        logger.info(f"   Validation: {result['val']} samples → val.jsonl")
        logger.info(f"   Teacher: {result['teacher']} samples → teacher_pool.jsonl")
        logger.info(
            f"   Combined: {result['total_processed']} samples → all_samples.jsonl"
        )
        if self.invalid_objects or self.invalid_samples:
            logger.info("🔍 Processing Summary:")
            logger.info(f"   Invalid objects tracked: {result['total_invalid_objects']}")
            logger.info(f"   Invalid samples tracked: {result['total_invalid_samples']}")

        return result


# TeacherSelector has been extracted to data_conversion.teacher_selector

def main():
    """Main entry point with CLI argument parsing."""
    import argparse

    from config import setup_logging, validate_config

    parser = argparse.ArgumentParser(description="Data Processor for Qwen2.5-VL")

    # Required arguments
    parser.add_argument(
        "--input_dir", required=True, help="Input directory with JSON/image files"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--language",
        choices=["chinese", "english"],
        required=True,
        help="Language mode",
    )
    parser.add_argument(
        "--dataset_name",
        help="Dataset name for organized output (auto-detected from input_dir if not provided)",
    )

    # Processing arguments - REQUIRED
    parser.add_argument(
        "--object_types",
        nargs="+",
        required=True,
        help="Object types to include (e.g., bbu label fiber connect_point)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        required=True,
        help="Validation split ratio (e.g., 0.1)",
    )
    parser.add_argument(
        "--max_teachers",
        type=int,
        required=True,
        help="Maximum teacher samples (e.g., 10)",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed (e.g., 42)"
    )

    # Processing options - OPTIONAL
    parser.add_argument("--token_map_path", help="Path to token mapping file")
    parser.add_argument("--hierarchy_path", help="Path to label hierarchy file")
    parser.add_argument("--resize", action="store_true", help="Enable image resizing")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # Advanced processing options
    parser.add_argument(
        "--geometry_diversity_weight",
        type=float,
        default=4.0,
        help="Weight for geometry diversity in teacher selection",
    )

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
