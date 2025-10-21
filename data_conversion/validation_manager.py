#!/usr/bin/env python3
"""
Centralized Validation Manager for Data Pipeline

This module provides comprehensive validation capabilities for the data processing pipeline,
consolidating all validation logic into a single, maintainable system with detailed reporting.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


class ValidationError:
    """Represents a single validation error with details and suggestions."""

    def __init__(
        self,
        error_type: str,
        severity: str,
        message: str,
        fix_suggestion: Optional[str] = None,
        object_index: Optional[int] = None,
        field_name: Optional[str] = None,
        invalid_value: Any = None,
    ):
        self.error_type = error_type
        self.severity = severity  # 'critical', 'warning', 'info'
        self.message = message
        self.fix_suggestion = fix_suggestion
        self.object_index = object_index
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON serialization."""
        return {
            "error_type": self.error_type,
            "severity": self.severity,
            "message": self.message,
            "fix_suggestion": self.fix_suggestion,
            "object_index": self.object_index,
            "field_name": self.field_name,
            "invalid_value": self.invalid_value,
            "timestamp": self.timestamp,
        }


class ValidationReport:
    """Comprehensive validation report with statistics and error details."""

    def __init__(self, sample_id: str):
        self.sample_id = sample_id
        self.errors: List[ValidationError] = []
        self.is_valid = True
        self.validation_timestamp = datetime.now().isoformat()

    def add_error(self, error: ValidationError) -> None:
        """Add a validation error to the report."""
        self.errors.append(error)
        if error.severity == "critical":
            self.is_valid = False

    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of errors by type and severity."""
        summary = {
            "total_errors": len(self.errors),
            "critical_errors": sum(1 for e in self.errors if e.severity == "critical"),
            "warnings": sum(1 for e in self.errors if e.severity == "warning"),
            "info": sum(1 for e in self.errors if e.severity == "info"),
        }

        # Count by error type
        error_types = {}
        for error in self.errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        summary["error_types"] = error_types

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "sample_id": self.sample_id,
            "is_valid": self.is_valid,
            "validation_timestamp": self.validation_timestamp,
            "error_summary": self.get_error_summary(),
            "errors": [error.to_dict() for error in self.errors],
        }


class ValidationManager:
    """
    Centralized validation system for data processing pipeline.

    Provides comprehensive validation with detailed error reporting,
    configurable validation rules, and invalid sample filtering capabilities.
    """

    def __init__(
        self,
        validation_mode: str = "strict",  # 'strict', 'lenient', 'warning_only'
        min_object_size: int = 10,  # minimum bbox width/height in pixels
        max_coordinate_value: int = 50000,  # maximum reasonable coordinate value
        require_non_empty_description: bool = True,
        check_coordinate_bounds: bool = True,
    ):
        self.validation_mode = validation_mode
        self.min_object_size = min_object_size
        self.max_coordinate_value = max_coordinate_value
        self.require_non_empty_description = require_non_empty_description
        self.check_coordinate_bounds = check_coordinate_bounds

        # Statistics tracking
        self.total_samples_processed = 0
        self.total_objects_processed = 0
        self.valid_samples = 0
        self.invalid_samples = 0
        self.validation_reports: List[ValidationReport] = []

        logger.info(f"ValidationManager initialized in {validation_mode} mode")

    def validate_sample(
        self,
        sample: Dict[str, Any],
        sample_id: str,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> Tuple[bool, ValidationReport]:
        """
        Validate a complete sample with comprehensive error reporting.

        Args:
            sample: Sample dictionary to validate
            sample_id: Unique identifier for the sample
            image_width: Image width for coordinate bounds checking
            image_height: Image height for coordinate bounds checking

        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = ValidationReport(sample_id)
        self.total_samples_processed += 1

        # Basic structure validation
        self._validate_sample_structure(sample, report)

        # Object-level validation
        if "objects" in sample and isinstance(sample["objects"], list):
            for i, obj in enumerate(sample["objects"]):
                self._validate_object(obj, i, report, image_width, image_height)
                self.total_objects_processed += 1

        # Update statistics
        if report.is_valid:
            self.valid_samples += 1
        else:
            self.invalid_samples += 1

        self.validation_reports.append(report)

        return report.is_valid, report

    def _validate_sample_structure(
        self, sample: Dict[str, Any], report: ValidationReport
    ) -> None:
        """Validate basic sample structure requirements."""
        required_fields = ["images", "objects"]

        for field in required_fields:
            if field not in sample:
                report.add_error(
                    ValidationError(
                        error_type="missing_field",
                        severity="critical",
                        message=f"Missing required field: {field}",
                        fix_suggestion=f"Add '{field}' field to sample",
                        field_name=field,
                    )
                )
                return

        # Validate images field
        images = sample.get("images")
        if not isinstance(images, list) or len(images) == 0:
            report.add_error(
                ValidationError(
                    error_type="invalid_images_field",
                    severity="critical",
                    message="Field 'images' must be a non-empty list",
                    fix_suggestion="Ensure 'images' contains at least one image path",
                    field_name="images",
                    invalid_value=images,
                )
            )

        # Validate objects field
        objects = sample.get("objects")
        if not isinstance(objects, list):
            report.add_error(
                ValidationError(
                    error_type="invalid_objects_field",
                    severity="critical",
                    message="Field 'objects' must be a list",
                    fix_suggestion="Ensure 'objects' is a list (can be empty)",
                    field_name="objects",
                    invalid_value=type(objects).__name__,
                )
            )
        elif len(objects) == 0:
            report.add_error(
                ValidationError(
                    error_type="empty_objects",
                    severity="warning",
                    message="Sample has no objects",
                    fix_suggestion="Consider removing samples with no annotations or add objects",
                    field_name="objects",
                )
            )

    def _validate_object(
        self,
        obj: Dict[str, Any],
        object_index: int,
        report: ValidationReport,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> None:
        """Validate a single object with comprehensive checks."""
        if not isinstance(obj, dict):
            report.add_error(
                ValidationError(
                    error_type="invalid_object_type",
                    severity="critical",
                    message=f"Object {object_index} must be a dictionary",
                    fix_suggestion="Ensure all objects are dictionary/JSON objects",
                    object_index=object_index,
                    invalid_value=type(obj).__name__,
                )
            )
            return

        # Check for description field
        if "desc" not in obj:
            report.add_error(
                ValidationError(
                    error_type="missing_description",
                    severity="critical",
                    message=f"Object {object_index} missing 'desc' field",
                    fix_suggestion="Add 'desc' field with meaningful description",
                    object_index=object_index,
                    field_name="desc",
                )
            )
        else:
            self._validate_description(obj["desc"], object_index, report)

        # Check for geometry fields
        geometry_types = ["bbox_2d", "quad", "line"]
        found_geometry = None

        for geom_type in geometry_types:
            if geom_type in obj:
                if found_geometry is not None:
                    report.add_error(
                        ValidationError(
                            error_type="multiple_geometries",
                            severity="warning",
                            message=f"Object {object_index} has multiple geometry types",
                            fix_suggestion="Use only one geometry type per object",
                            object_index=object_index,
                        )
                    )
                found_geometry = geom_type
                self._validate_geometry(
                    obj[geom_type],
                    geom_type,
                    object_index,
                    report,
                    image_width,
                    image_height,
                )

        if found_geometry is None:
            report.add_error(
                ValidationError(
                    error_type="missing_geometry",
                    severity="critical",
                    message=f"Object {object_index} missing geometry (bbox_2d, quad, or line)",
                    fix_suggestion="Add one of: bbox_2d, quad, or line geometry",
                    object_index=object_index,
                )
            )

    def _validate_description(
        self, desc: Any, object_index: int, report: ValidationReport
    ) -> None:
        """Validate object description field."""
        if not isinstance(desc, str):
            report.add_error(
                ValidationError(
                    error_type="invalid_description_type",
                    severity="critical",
                    message=f"Object {object_index} 'desc' must be string",
                    fix_suggestion="Convert description to string format",
                    object_index=object_index,
                    field_name="desc",
                    invalid_value=type(desc).__name__,
                )
            )
            return

        if self.require_non_empty_description and not desc.strip():
            report.add_error(
                ValidationError(
                    error_type="empty_description",
                    severity="warning"
                    if self.validation_mode == "lenient"
                    else "critical",
                    message=f"Object {object_index} has empty description",
                    fix_suggestion="Provide meaningful description for the object",
                    object_index=object_index,
                    field_name="desc",
                    invalid_value=desc,
                )
            )

    def _validate_geometry(
        self,
        geometry: Any,
        geom_type: str,
        object_index: int,
        report: ValidationReport,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> None:
        """Validate geometry coordinates based on type."""
        if geom_type == "bbox_2d":
            self._validate_bbox(
                geometry, object_index, report, image_width, image_height
            )
        elif geom_type == "quad":
            self._validate_quad(
                geometry, object_index, report, image_width, image_height
            )
        elif geom_type == "line":
            self._validate_line(
                geometry, object_index, report, image_width, image_height
            )

    def _validate_bbox(
        self,
        bbox: Any,
        object_index: int,
        report: ValidationReport,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> None:
        """Validate bbox_2d format and values."""
        if not isinstance(bbox, list) or len(bbox) != 4:
            report.add_error(
                ValidationError(
                    error_type="invalid_bbox_format",
                    severity="critical",
                    message=f"Object {object_index} bbox_2d must be [x1,y1,x2,y2]",
                    fix_suggestion="Ensure bbox_2d is a list of exactly 4 numbers",
                    object_index=object_index,
                    field_name="bbox_2d",
                    invalid_value=bbox,
                )
            )
            return

        # Check coordinate types
        for i, coord in enumerate(bbox):
            if not isinstance(coord, (int, float)):
                report.add_error(
                    ValidationError(
                        error_type="invalid_coordinate_type",
                        severity="critical",
                        message=f"Object {object_index} bbox coordinate {i} must be number",
                        fix_suggestion="Ensure all coordinates are numeric values",
                        object_index=object_index,
                        field_name=f"bbox_2d[{i}]",
                        invalid_value=coord,
                    )
                )
                return

        x1, y1, x2, y2 = bbox

        # Check for reasonable coordinate values
        coords = [x1, y1, x2, y2]
        for i, coord in enumerate(coords):
            if abs(coord) > self.max_coordinate_value:
                report.add_error(
                    ValidationError(
                        error_type="extreme_coordinate_value",
                        severity="warning",
                        message=f"Object {object_index} coordinate {i} seems extremely large: {coord}",
                        fix_suggestion="Check if coordinate values are reasonable for image size",
                        object_index=object_index,
                        field_name=f"bbox_2d[{i}]",
                        invalid_value=coord,
                    )
                )

        # Check coordinate ordering and bounds
        if x1 > x2 or y1 > y2:
            report.add_error(
                ValidationError(
                    error_type="invalid_bbox_ordering",
                    severity="warning",
                    message=f"Object {object_index} bbox coordinates not in min/max order",
                    fix_suggestion="Ensure x1 <= x2 and y1 <= y2",
                    object_index=object_index,
                    field_name="bbox_2d",
                    invalid_value=bbox,
                )
            )

        # Check minimum object size
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        if width < self.min_object_size or height < self.min_object_size:
            report.add_error(
                ValidationError(
                    error_type="object_too_small",
                    severity="warning",
                    message=f"Object {object_index} is very small ({width}x{height})",
                    fix_suggestion=f"Consider removing objects smaller than {self.min_object_size}px",
                    object_index=object_index,
                    field_name="bbox_2d",
                    invalid_value=f"{width}x{height}",
                )
            )

        # Check image bounds if available
        if self.check_coordinate_bounds and image_width and image_height:
            if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
                report.add_error(
                    ValidationError(
                        error_type="coordinates_out_of_bounds",
                        severity="critical",
                        message=f"Object {object_index} coordinates exceed image bounds",
                        fix_suggestion=f"Ensure coordinates are within image ({image_width}x{image_height})",
                        object_index=object_index,
                        field_name="bbox_2d",
                        invalid_value=f"bbox: {bbox}, image: {image_width}x{image_height}",
                    )
                )

    def _validate_quad(
        self,
        quad: Any,
        object_index: int,
        report: ValidationReport,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> None:
        """Validate quad format [x1,y1,x2,y2,x3,y3,x4,y4]."""
        if not isinstance(quad, list) or len(quad) != 8:
            report.add_error(
                ValidationError(
                    error_type="invalid_quad_format",
                    severity="critical",
                    message=f"Object {object_index} quad must be list of 8 coordinates",
                    fix_suggestion="Ensure quad is [x1,y1,x2,y2,x3,y3,x4,y4]",
                    object_index=object_index,
                    field_name="quad",
                    invalid_value=quad,
                )
            )
            return

        # Check coordinate types
        for i, coord in enumerate(quad):
            if not isinstance(coord, (int, float)):
                report.add_error(
                    ValidationError(
                        error_type="invalid_coordinate_type",
                        severity="critical",
                        message=f"Object {object_index} quad coordinate {i} must be number",
                        fix_suggestion="Ensure all coordinates are numeric values",
                        object_index=object_index,
                        field_name=f"quad[{i}]",
                        invalid_value=coord,
                    )
                )
                return

        # Check bounds if image dimensions available
        if self.check_coordinate_bounds and image_width and image_height:
            for i in range(0, 8, 2):
                x, y = quad[i], quad[i + 1]
                if x < 0 or y < 0 or x > image_width or y > image_height:
                    report.add_error(
                        ValidationError(
                            error_type="coordinates_out_of_bounds",
                            severity="critical",
                            message=f"Object {object_index} quad point {i // 2} out of bounds",
                            fix_suggestion=f"Ensure coordinates are within image ({image_width}x{image_height})",
                            object_index=object_index,
                            field_name="quad",
                            invalid_value=f"point: ({x},{y}), image: {image_width}x{image_height}",
                        )
                    )

    def _validate_line(
        self,
        line: Any,
        object_index: int,
        report: ValidationReport,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> None:
        """Validate line format [x1,y1,x2,y2,...,xn,yn]."""
        if not isinstance(line, list) or len(line) < 4 or len(line) % 2 != 0:
            report.add_error(
                ValidationError(
                    error_type="invalid_line_format",
                    severity="critical",
                    message=f"Object {object_index} line must have even number of coordinates (>=4)",
                    fix_suggestion="Ensure line has pairs of x,y coordinates",
                    object_index=object_index,
                    field_name="line",
                    invalid_value=line,
                )
            )
            return

        # Check coordinate types
        for i, coord in enumerate(line):
            if not isinstance(coord, (int, float)):
                report.add_error(
                    ValidationError(
                        error_type="invalid_coordinate_type",
                        severity="critical",
                        message=f"Object {object_index} line coordinate {i} must be number",
                        fix_suggestion="Ensure all coordinates are numeric values",
                        object_index=object_index,
                        field_name=f"line[{i}]",
                        invalid_value=coord,
                    )
                )
                return

        # Check bounds if image dimensions available
        if self.check_coordinate_bounds and image_width and image_height:
            for i in range(0, len(line), 2):
                x, y = line[i], line[i + 1]
                if x < 0 or y < 0 or x > image_width or y > image_height:
                    report.add_error(
                        ValidationError(
                            error_type="coordinates_out_of_bounds",
                            severity="critical",
                            message=f"Object {object_index} line point {i // 2} out of bounds",
                            fix_suggestion=f"Ensure coordinates are within image ({image_width}x{image_height})",
                            object_index=object_index,
                            field_name="line",
                            invalid_value=f"point: ({x},{y}), image: {image_width}x{image_height}",
                        )
                    )

    def filter_valid_objects(
        self,
        objects: List[Dict[str, Any]],
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        sample_id: str = "unknown",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter objects, returning valid objects and collecting invalid ones.

        Args:
            objects: List of objects to validate
            image_width: Image width for bounds checking
            image_height: Image height for bounds checking
            sample_id: Sample identifier for reporting

        Returns:
            Tuple of (valid_objects, invalid_objects_with_errors)
        """
        valid_objects = []
        invalid_objects = []

        for i, obj in enumerate(objects):
            # Create a temporary sample for validation
            temp_sample = {"images": [sample_id], "objects": [obj]}
            is_valid, report = self.validate_sample(
                temp_sample, f"{sample_id}_obj_{i}", image_width, image_height
            )

            if is_valid or self.validation_mode == "warning_only":
                valid_objects.append(obj)
            else:
                # Add error details to invalid object
                invalid_obj = obj.copy()
                invalid_obj["_validation_errors"] = [
                    error.to_dict() for error in report.errors
                ]
                invalid_obj["_sample_id"] = sample_id
                invalid_obj["_object_index"] = i
                invalid_objects.append(invalid_obj)

        return valid_objects, invalid_objects

    def generate_validation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive validation summary statistics."""
        if not self.validation_reports:
            return {"message": "No validation reports available"}

        total_errors = sum(len(report.errors) for report in self.validation_reports)
        error_type_counts = {}
        severity_counts = {"critical": 0, "warning": 0, "info": 0}

        for report in self.validation_reports:
            for error in report.errors:
                error_type_counts[error.error_type] = (
                    error_type_counts.get(error.error_type, 0) + 1
                )
                severity_counts[error.severity] += 1

        return {
            "validation_summary": {
                "total_samples_processed": self.total_samples_processed,
                "total_objects_processed": self.total_objects_processed,
                "valid_samples": self.valid_samples,
                "invalid_samples": self.invalid_samples,
                "validation_success_rate": self.valid_samples
                / max(1, self.total_samples_processed),
                "total_errors": total_errors,
                "error_type_counts": error_type_counts,
                "severity_counts": severity_counts,
            },
            "validation_config": {
                "validation_mode": self.validation_mode,
                "min_object_size": self.min_object_size,
                "max_coordinate_value": self.max_coordinate_value,
                "require_non_empty_description": self.require_non_empty_description,
                "check_coordinate_bounds": self.check_coordinate_bounds,
            },
            "timestamp": datetime.now().isoformat(),
        }

    def export_validation_reports(self, output_dir: Path) -> Dict[str, str]:
        """
        Export validation reports to JSON files.

        Args:
            output_dir: Directory to save validation reports

        Returns:
            Dictionary with paths to generated report files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_files = {}

        # Export detailed validation reports
        detailed_report_path = output_dir / "validation_detailed_report.json"
        detailed_reports = [report.to_dict() for report in self.validation_reports]
        with open(detailed_report_path, "w", encoding="utf-8") as f:
            json.dump(detailed_reports, f, indent=2, ensure_ascii=False)
        report_files["detailed_report"] = str(detailed_report_path)

        # Export validation summary
        summary_path = output_dir / "validation_summary.json"
        summary = self.generate_validation_summary()
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        report_files["summary"] = str(summary_path)

        # Export invalid samples only
        invalid_samples_path = output_dir / "invalid_samples_report.json"
        invalid_reports = [
            report.to_dict()
            for report in self.validation_reports
            if not report.is_valid
        ]
        with open(invalid_samples_path, "w", encoding="utf-8") as f:
            json.dump(invalid_reports, f, indent=2, ensure_ascii=False)
        report_files["invalid_samples"] = str(invalid_samples_path)

        logger.info(f"Validation reports exported to {output_dir}")
        logger.info(f"Generated files: {list(report_files.keys())}")

        return report_files
