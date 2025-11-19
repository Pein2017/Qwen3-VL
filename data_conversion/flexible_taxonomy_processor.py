#!/usr/bin/env python3
"""
Flexible Taxonomy Processor

A comprehensive system for processing V2 annotations using a flexible attribute taxonomy
without hardcoded stages. Groups information logically and creates hierarchical descriptions.

Includes HierarchicalProcessor compatibility layer for V2 data processing.
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
class AnnotationSample:
    """Represents a processed annotation sample with hierarchical information."""

    object_type: str
    geometry_format: str  # bbox_2d, poly, line
    coordinates: List[float]
    grouped_attributes: Dict[str, Dict[str, str]]  # group -> attribute -> value
    description: str
    original_geometry: Optional[Dict] = None

    def to_training_format(self) -> Dict:
        """Convert to training format with native geometry type."""
        return {self.geometry_format: self.coordinates, "desc": self.description}


class FlexibleTaxonomyProcessor:
    """Process V2 annotations using flexible attribute taxonomy."""

    def __init__(
        self,
        taxonomy_path: Optional[str] = None,
        hierarchical_mapping_path: Optional[str] = None,
    ):
        if taxonomy_path is None:
            taxonomy_path = str(Path(__file__).parent / "attribute_taxonomy.json")
        if hierarchical_mapping_path is None:
            hierarchical_mapping_path = str(
                Path(__file__).parent / "hierarchical_attribute_mapping.json"
            )

        self.taxonomy = self._load_taxonomy(taxonomy_path)
        self.hierarchical_mapping = self._load_taxonomy(hierarchical_mapping_path)
        self.object_types = self.taxonomy["object_types"]
        self.attribute_groups = self.taxonomy["attribute_groups"]
        self.geometry_mapping = self.taxonomy["geometry_format_mapping"]
        self.hierarchical_object_types = self.hierarchical_mapping["object_types"]

        logger.info(
            f"Loaded taxonomy with {len(self.object_types)} object types and {len(self.attribute_groups)} attribute groups"
        )
        logger.info(
            f"Loaded hierarchical mapping for {len(self.hierarchical_object_types)} object types"
        )

    def _load_taxonomy(self, taxonomy_path: str) -> Dict:
        """Load the attribute taxonomy."""
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def process_v2_feature(self, feature: Dict) -> Optional[AnnotationSample]:
        """
        Process a single V2 feature into structured annotation sample.

        Args:
            feature: V2 feature with geometry and properties

        Returns:
            AnnotationSample or None if processing fails
        """
        properties = feature.get("properties", {})
        content_zh = properties.get("contentZh", {})
        content = properties.get("content", {})
        geometry = feature.get("geometry", {})

        # Determine object type
        object_type = self._determine_object_type(content, content_zh)
        if not object_type:
            logger.warning("Could not determine object type")
            return None

        # Process geometry
        geometry_format, coordinates = self._process_geometry(geometry, object_type)

        # Group attributes by taxonomy
        grouped_attributes = self._group_attributes(content, content_zh, object_type)

        # Create hierarchical description using new method
        description = self._create_hierarchical_description(
            object_type, content_zh, content
        )
        # Standardize 标签 descriptions
        from data_conversion.utils.sanitizers import standardize_label_description

        description = standardize_label_description(description) or description

        return AnnotationSample(
            object_type=object_type,
            geometry_format=geometry_format,
            coordinates=coordinates,
            grouped_attributes=grouped_attributes,
            description=description,
            original_geometry=geometry,
        )

    def _determine_object_type(self, content: Dict, content_zh: Dict) -> Optional[str]:
        """Determine object type from content fields."""
        # Check content.label first
        content_label = content.get("label", "").strip()
        for obj_type, obj_info in self.object_types.items():
            if content_label == obj_info["content_key"]:
                return obj_type

        # Check contentZh.标签 as fallback
        zh_label = content_zh.get("标签", "").strip()
        for obj_type, obj_info in self.object_types.items():
            if zh_label == obj_info["chinese_label"] or zh_label in obj_info.get(
                "aliases", []
            ):
                return obj_type

        return None

    def _process_geometry(
        self, geometry: Dict, object_type: str
    ) -> Tuple[str, List[float]]:
        """Process geometry based on object type and geometry structure."""
        from data_conversion.coordinate_manager import CoordinateManager

        geometry_type = geometry.get("type", "")
        coordinates = geometry.get("coordinates", [])

        # Extract bbox for all cases
        bbox = CoordinateManager.extract_bbox_from_geometry(geometry)

        # Handle geometry types based on their native format
        if geometry_type == "LineString":
            # Extract line coordinates
            line_coords = []
            for coord in coordinates:
                if isinstance(coord, list) and len(coord) >= 2:
                    line_coords.extend([int(round(coord[0])), int(round(coord[1]))])
            return "line", line_coords

        elif geometry_type in ["Quad", "Square", "Polygon"]:
            # Extract poly coordinates using unified helper
            poly_coords = CoordinateManager._extract_poly_coordinates(geometry)
            if poly_coords:
                return "poly", poly_coords

        # ExtentPolygon -> bbox_2d
        return "bbox_2d", bbox

    def _group_attributes(
        self, content: Dict, content_zh: Dict, object_type: str
    ) -> Dict[str, Dict[str, str]]:
        """Group attributes according to taxonomy."""
        grouped = {}

        for group_name, group_info in self.attribute_groups.items():
            group_attributes = {}

            for attr_name, attr_info in group_info["attributes"].items():
                # Check if attribute applies to this object type
                applies_to = attr_info.get("applies_to", [])
                if applies_to != "all" and object_type not in applies_to:
                    continue

                # Extract attribute value
                value = self._extract_attribute_value(
                    content, content_zh, attr_info, object_type
                )
                if value:
                    group_attributes[attr_name] = value

            if group_attributes:
                grouped[group_name] = group_attributes

        return grouped

    def _extract_attribute_value(
        self, content: Dict, content_zh: Dict, attr_info: Dict, object_type: str
    ) -> Optional[str]:
        """Extract attribute value from contentZh (Chinese) fields only."""
        # For Chinese mode, extract directly from contentZh using Chinese questions
        chinese_questions = attr_info.get("chinese_questions", [])

        # Look for matching Chinese question in contentZh
        for question in chinese_questions:
            if question in content_zh:
                raw_value = content_zh[question]

                # Handle array values (like connect_point_check)
                if isinstance(raw_value, list) and raw_value:
                    raw_value = raw_value[0]

                # Return the Chinese value directly
                if raw_value and str(raw_value).strip():
                    return str(raw_value).strip()

        # Fallback: check for free text fields in contentZh
        values_mapping = attr_info.get("values")
        if values_mapping == "free_text":
            content_mapping = attr_info.get("content_mapping")
            if content_mapping and content_mapping in content_zh:
                return str(content_zh[content_mapping]).strip()

        return None

    def _create_hierarchical_description(
        self, object_type: str, content_zh: Dict, content: Dict
    ) -> str:
        """Create hierarchical description with correct separators: comma for same-level, slash for different levels."""

        # Get hierarchical mapping for this object type
        if object_type not in self.hierarchical_object_types:
            logger.warning(
                f"No hierarchical mapping found for object type: {object_type}"
            )
            return self.hierarchical_object_types.get(object_type, {}).get(
                "chinese_label", object_type
            )

        obj_mapping = self.hierarchical_object_types[object_type]

        # Process attributes and group by level
        attributes = obj_mapping["attributes"]
        attributes_by_level = sorted(attributes, key=lambda x: x["level"])

        # Track values for conditional logic and group by level
        extracted_values = {}
        values_by_level = {}  # level -> [values]

        # Start with object type at level 0
        values_by_level[0] = [obj_mapping["chinese_label"]]

        for attr in attributes_by_level:
            attr_name = attr["name"]
            level = attr["level"]

            # Check if this is a conditional attribute
            if attr.get("conditional"):
                condition = attr["conditional"]
                parent_attr = condition["parent_attribute"]
                required_parent_value = condition["parent_value"]

                # Skip if parent condition not met
                if (
                    parent_attr not in extracted_values
                    or extracted_values[parent_attr] != required_parent_value
                ):
                    continue

            # Extract value for this attribute
            attr_value = self._extract_hierarchical_attribute_value(
                attr, content_zh, content
            )

            if attr_value:
                extracted_values[attr_name] = attr_value

                # Group values by level
                if level not in values_by_level:
                    values_by_level[level] = []

                # Handle multiple values with separator (for specific issues like "未拧紧,生锈")
                if attr.get("multiple_values_separator") and "," in attr_value:
                    values_by_level[level].append(attr_value)  # Keep internal commas
                else:
                    values_by_level[level].append(attr_value)
            elif attr.get("required", False):
                # For required attributes, we might want to log missing values
                logger.debug(
                    f"Missing required attribute {attr_name} for {object_type}"
                )

        # Combine levels with proper separators
        level_parts = []
        for level in sorted(values_by_level.keys()):
            level_values = values_by_level[level]
            if level_values:
                # Join same-level values with comma
                level_part = ",".join(level_values)
                level_parts.append(level_part)

        # Join different levels with slash
        return "/".join(level_parts)

    def _extract_hierarchical_attribute_value(
        self, attr: Dict, content_zh: Dict, content: Dict
    ) -> Optional[str]:
        """Extract attribute value according to hierarchical mapping."""
        content_mapping = attr.get("content_mapping")
        if not content_mapping:
            return None

        # Handle free text attributes
        if attr.get("is_free_text", False):
            # For labels, extract from multiple possible fields
            if attr["name"] == "text_content":
                # Try to get actual text content from label fields
                for question in attr["chinese_questions"]:
                    if question in content_zh:
                        value = content_zh[question]
                        if isinstance(value, list) and value:
                            value = value[0]
                        if (
                            value
                            and str(value).strip()
                            and str(value).strip()
                            not in ["能", "不能", " ", "  ", "   "]
                        ):
                            return str(value).strip()

                # Fallback to content mapping
                if content_mapping in content_zh:
                    value = content_zh[content_mapping]
                    if isinstance(value, list) and value:
                        value = value[0]
                    if value and str(value).strip():
                        return str(value).strip()

            # For special circumstances
            elif attr["name"] == "special_circumstances":
                # 1) Try Chinese question key(s) directly
                for question in attr.get("chinese_questions", []):
                    if question in content_zh:
                        value = content_zh[question]
                        if isinstance(value, list) and value:
                            value = value[0]
                        if value and str(value).strip():
                            return str(value).strip()

                # 2) Try mapped key (e.g., special_situation)
                content_mapping = attr.get("content_mapping")
                if content_mapping and content_mapping in content_zh:
                    value = content_zh[content_mapping]
                    if isinstance(value, list) and value:
                        value = value[0]
                    if value and str(value).strip():
                        return str(value).strip()

                # 3) Legacy/fallback keys (ex_info) in Chinese and English content
                for legacy_key in ("ex_info", "exInfo"):
                    if legacy_key in content_zh:
                        value = content_zh[legacy_key]
                        if isinstance(value, list) and value:
                            value = value[0]
                        if value and str(value).strip():
                            return str(value).strip()
                    if legacy_key in content:
                        value = content[legacy_key]
                        if isinstance(value, list) and value:
                            value = value[0]
                        if value and str(value).strip():
                            return str(value).strip()
                return None
            return None

        # Handle structured attributes with defined values
        attr_values = attr.get("values", {})

        # Check in content_zh first
        for question in attr["chinese_questions"]:
            if question in content_zh:
                raw_value = content_zh[question]
                if isinstance(raw_value, list) and raw_value:
                    raw_value = raw_value[0]

                raw_value_str = str(raw_value).strip() if raw_value else ""

                # Try exact match first
                for key, mapped_value in attr_values.items():
                    if raw_value_str == key:
                        return mapped_value

                # Try partial match for complex values
                for key, mapped_value in attr_values.items():
                    if key in raw_value_str or raw_value_str in key:
                        return mapped_value

        # Check content mapping directly
        if content_mapping in content_zh:
            raw_value = content_zh[content_mapping]
            if isinstance(raw_value, list) and raw_value:
                raw_value = raw_value[0]

            raw_value_str = str(raw_value).strip() if raw_value else ""

            # Try to map the value
            for key, mapped_value in attr_values.items():
                if raw_value_str == key or key in raw_value_str or raw_value_str in key:
                    return mapped_value

        return None

    def _create_description(
        self, object_type: str, grouped_attributes: Dict[str, Dict[str, str]]
    ) -> str:
        """Legacy method - kept for compatibility. Use _create_hierarchical_description instead."""
        # Fallback to hierarchical description if possible
        return self.hierarchical_object_types.get(object_type, {}).get(
            "chinese_label", object_type
        )

    def process_v2_file(self, file_path: str) -> List[AnnotationSample]:
        """Process entire V2 JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("markResult", {}).get("features", [])
        samples = []

        for feature in features:
            sample = self.process_v2_feature(feature)
            if sample:
                samples.append(sample)

        logger.info(f"Processed {len(samples)} samples from {file_path}")
        return samples

    def batch_process(self, input_dir: str, output_file: str):
        """Batch process V2 files and save training format."""
        input_path = Path(input_dir)
        all_samples = []

        # Process all JSON files
        for json_file in input_path.glob("*.json"):
            samples = self.process_v2_file(str(json_file))
            all_samples.extend(samples)

        # Convert to training format and save
        training_data = []
        for sample in all_samples:
            training_data.append(sample.to_training_format())

        # Save as JSONL
        with open(output_file, "w", encoding="utf-8") as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(training_data)} samples to {output_file}")
        return len(training_data)

    def get_statistics(self, samples: List[AnnotationSample]) -> Dict[str, Any]:
        """Generate statistics about processed samples."""
        stats = {
            "total_samples": len(samples),
            "object_types": {},
            "geometry_formats": {},
            "attribute_groups": {},
        }

        for sample in samples:
            # Object type counts
            obj_type = sample.object_type
            stats["object_types"][obj_type] = stats["object_types"].get(obj_type, 0) + 1

            # Geometry format counts
            geo_format = sample.geometry_format
            stats["geometry_formats"][geo_format] = (
                stats["geometry_formats"].get(geo_format, 0) + 1
            )

            # Attribute group counts
            for group_name in sample.grouped_attributes:
                stats["attribute_groups"][group_name] = (
                    stats["attribute_groups"].get(group_name, 0) + 1
                )

        return stats


# Compatibility Layer - HierarchicalProcessor
class HierarchicalProcessor:
    """Clean processor for V2 data with native geometry output and object type filtering."""

    def __init__(self, object_types=None, label_hierarchy: Optional[Dict] = None):
        """Initialize processor for Chinese-only processing with object type filtering."""
        self.object_types = object_types or {"bbu", "label", "fiber", "connect_point"}
        self.label_hierarchy = label_hierarchy or {}

        # Create flexible processor
        self.flexible_processor = FlexibleTaxonomyProcessor()

        logger.info(
            f"Initialized HierarchicalProcessor for Chinese-only processing with object types: {self.object_types}"
        )

    def extract_objects_from_markresult(self, features: List[Dict]) -> List[Dict]:
        """
        Extract objects from markResult features with native geometry types.
        Filters by specified object types for training subject separation.

        Returns objects in format:
        [
            {'bbox_2d': [x1,y1,x2,y2], 'desc': '...'},
            {'poly': [x1,y1,x2,y2,x3,y3,x4,y4,...], 'desc': '...'},
            {'line': [x1,y1,x2,y2,...], 'desc': '...'}
        ]
        """
        objects = []

        for feature in features:
            # Process with flexible processor
            sample = self.flexible_processor.process_v2_feature(feature)
            if not sample:
                continue

            # Filter by object type - only include objects matching specified types
            if sample.object_type not in self.object_types:
                continue

            # Convert to clean training format (native geometry only)
            training_obj = sample.to_training_format()
            objects.append(training_obj)

        return objects


# For import compatibility
def create_hierarchical_processor(*args, **kwargs):
    """Factory function for compatibility."""
    return HierarchicalProcessor(*args, **kwargs)


if __name__ == "__main__":
    # Example usage
    processor = FlexibleTaxonomyProcessor()

    # Test single file
    test_file = "ds_v2/QC-20230216-0000244_377872.json"
    samples = processor.process_v2_file(test_file)

    print(f"Processed {len(samples)} samples")
    for sample in samples[:3]:  # Show first 3
        print(f"Object: {sample.object_type}")
        print(f"Format: {sample.geometry_format}")
        print(f"Description: {sample.description}")
        print(f"Attributes: {sample.grouped_attributes}")
        print("---")
