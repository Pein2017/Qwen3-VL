#!/usr/bin/env python3
"""
Validate JSONL files for Qwen3-VL training.

Checks:
- JSON format validity
- Required fields presence
- Data types correctness
- Image file existence
- Bbox format and bounds
- Summary statistics
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class ValidationError:
    """Container for validation errors."""
    def __init__(self, line_num: int, field: str, message: str):
        self.line_num = line_num
        self.field = field
        self.message = message
    
    def __str__(self):
        return f"Line {self.line_num}, field '{self.field}': {self.message}"


class JSONLValidator:
    """Validator for Qwen3-VL JSONL format."""
    
    def __init__(self, check_images: bool = True, verbose: bool = False):
        self.check_images = check_images
        self.verbose = verbose
        self.errors: List[ValidationError] = []
        self.warnings: List[str] = []
        self.stats = {
            "total_lines": 0,
            "valid_samples": 0,
            "total_objects": 0,
            "missing_images": 0,
            "invalid_bboxes": 0,
            "categories": set()
        }
    
    def validate_file(self, jsonl_path: str) -> bool:
        """
        Validate entire JSONL file.
        
        Returns:
            True if all samples valid
        """
        if not os.path.exists(jsonl_path):
            print(f"✗ File not found: {jsonl_path}")
            return False
        
        print(f"Validating: {jsonl_path}")
        print("="*60)
        
        jsonl_dir = os.path.dirname(os.path.abspath(jsonl_path))
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                self.stats["total_lines"] += 1
                
                # Parse JSON
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    self.errors.append(ValidationError(
                        line_num, "json", f"Invalid JSON: {e}"
                    ))
                    continue
                
                # Validate sample
                if self.validate_sample(sample, line_num, jsonl_dir):
                    self.stats["valid_samples"] += 1
        
        return self.print_results()
    
    def validate_sample(
        self, 
        sample: Dict[str, Any], 
        line_num: int,
        base_dir: str
    ) -> bool:
        """
        Validate one sample.
        
        Expected format:
        {
          "images": [str],
          "objects": [{"bbox_2d": [x1,y1,x2,y2], "desc": str}],
          "width": int,
          "height": int,
          "summary": str  # optional
        }
        
        Returns:
            True if sample is valid
        """
        sample_valid = True
        
        # Check required fields
        required_fields = ["images", "objects", "width", "height"]
        for field in required_fields:
            if field not in sample:
                self.errors.append(ValidationError(
                    line_num, field, "Required field missing"
                ))
                sample_valid = False
        
        if not sample_valid:
            return False
        
        # Validate 'images' field
        images = sample["images"]
        if not isinstance(images, list):
            self.errors.append(ValidationError(
                line_num, "images", f"Must be list, got {type(images).__name__}"
            ))
            sample_valid = False
        elif len(images) != 1:
            self.warnings.append(
                f"Line {line_num}: Expected 1 image, got {len(images)}"
            )
        elif images:
            image_path = images[0]
            if not isinstance(image_path, str):
                self.errors.append(ValidationError(
                    line_num, "images[0]", f"Must be string, got {type(image_path).__name__}"
                ))
                sample_valid = False
            elif self.check_images:
                # Resolve relative path
                if not os.path.isabs(image_path):
                    image_path = os.path.join(base_dir, image_path)
                if not os.path.exists(image_path):
                    self.errors.append(ValidationError(
                        line_num, "images[0]", f"Image not found: {image_path}"
                    ))
                    self.stats["missing_images"] += 1
                    sample_valid = False
        
        # Validate 'width' and 'height'
        width = sample["width"]
        height = sample["height"]
        if not isinstance(width, int) or width <= 0:
            self.errors.append(ValidationError(
                line_num, "width", f"Must be positive int, got {width}"
            ))
            sample_valid = False
        if not isinstance(height, int) or height <= 0:
            self.errors.append(ValidationError(
                line_num, "height", f"Must be positive int, got {height}"
            ))
            sample_valid = False
        
        # Validate 'objects' field
        objects = sample["objects"]
        if not isinstance(objects, list):
            self.errors.append(ValidationError(
                line_num, "objects", f"Must be list, got {type(objects).__name__}"
            ))
            sample_valid = False
        else:
            self.stats["total_objects"] += len(objects)
            for obj_idx, obj in enumerate(objects):
                if not self.validate_object(obj, line_num, obj_idx, width, height):
                    sample_valid = False
        
        # Validate optional 'summary'
        if "summary" in sample:
            if not isinstance(sample["summary"], str):
                self.warnings.append(
                    f"Line {line_num}: 'summary' should be string, got {type(sample['summary']).__name__}"
                )
        
        return sample_valid
    
    def validate_object(
        self,
        obj: Dict[str, Any],
        line_num: int,
        obj_idx: int,
        img_width: int,
        img_height: int
    ) -> bool:
        """Validate one object annotation."""
        obj_valid = True
        prefix = f"objects[{obj_idx}]"
        
        # Check required fields
        if "bbox_2d" not in obj:
            self.errors.append(ValidationError(
                line_num, f"{prefix}.bbox_2d", "Required field missing"
            ))
            return False
        
        if "desc" not in obj:
            self.errors.append(ValidationError(
                line_num, f"{prefix}.desc", "Required field missing"
            ))
            return False
        
        # Validate bbox_2d
        bbox = obj["bbox_2d"]
        if not isinstance(bbox, list):
            self.errors.append(ValidationError(
                line_num, f"{prefix}.bbox_2d", f"Must be list, got {type(bbox).__name__}"
            ))
            obj_valid = False
        elif len(bbox) != 4:
            self.errors.append(ValidationError(
                line_num, f"{prefix}.bbox_2d", f"Must have 4 values, got {len(bbox)}"
            ))
            obj_valid = False
        else:
            x1, y1, x2, y2 = bbox
            
            # Check types (allow int or float)
            for i, coord in enumerate(bbox):
                if not isinstance(coord, (int, float)):
                    self.errors.append(ValidationError(
                        line_num, f"{prefix}.bbox_2d[{i}]", 
                        f"Must be numeric, got {type(coord).__name__}"
                    ))
                    obj_valid = False
            
            # Check bbox is well-formed
            if x2 <= x1:
                self.errors.append(ValidationError(
                    line_num, f"{prefix}.bbox_2d", f"Invalid: x2 ({x2}) <= x1 ({x1})"
                ))
                self.stats["invalid_bboxes"] += 1
                obj_valid = False
            
            if y2 <= y1:
                self.errors.append(ValidationError(
                    line_num, f"{prefix}.bbox_2d", f"Invalid: y2 ({y2}) <= y1 ({y1})"
                ))
                self.stats["invalid_bboxes"] += 1
                obj_valid = False
            
            # Check bounds (allow partial outside for robustness)
            if x2 < 0 or x1 > img_width:
                self.warnings.append(
                    f"Line {line_num}, {prefix}.bbox_2d: Box completely outside image width"
                )
            if y2 < 0 or y1 > img_height:
                self.warnings.append(
                    f"Line {line_num}, {prefix}.bbox_2d: Box completely outside image height"
                )
        
        # Validate desc
        desc = obj["desc"]
        if not isinstance(desc, str):
            self.errors.append(ValidationError(
                line_num, f"{prefix}.desc", f"Must be string, got {type(desc).__name__}"
            ))
            obj_valid = False
        elif not desc.strip():
            self.errors.append(ValidationError(
                line_num, f"{prefix}.desc", "Cannot be empty"
            ))
            obj_valid = False
        else:
            self.stats["categories"].add(desc)
        
        return obj_valid
    
    def print_results(self) -> bool:
        """Print validation results."""
        print("\n" + "="*60)
        print("Validation Results")
        print("="*60)
        
        # Statistics
        print(f"\n✓ Statistics:")
        print(f"  Total lines: {self.stats['total_lines']}")
        print(f"  Valid samples: {self.stats['valid_samples']}")
        print(f"  Total objects: {self.stats['total_objects']}")
        print(f"  Unique categories: {len(self.stats['categories'])}")
        
        if self.stats['valid_samples'] > 0:
            avg_obj = self.stats['total_objects'] / self.stats['valid_samples']
            print(f"  Avg objects/sample: {avg_obj:.2f}")
        
        # Errors
        if self.errors:
            print(f"\n✗ Errors ({len(self.errors)}):")
            for error in self.errors[:20]:  # Show first 20
                print(f"  • {error}")
            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more errors")
        else:
            print("\n✓ No errors found")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠ Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"  • {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
        
        # Summary
        print("\n" + "="*60)
        if not self.errors:
            print("✓ VALIDATION PASSED")
            print("="*60 + "\n")
            return True
        else:
            print("✗ VALIDATION FAILED")
            print("="*60 + "\n")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate JSONL files for Qwen3-VL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected JSONL format:

{
  "images": ["path/to/image.jpg"],
  "objects": [
    {"bbox_2d": [x1, y1, x2, y2], "desc": "person"},
    {"bbox_2d": [x1, y1, x2, y2], "desc": "car"}
  ],
  "width": 640,
  "height": 480,
  "summary": "A person standing next to a car"  # optional
}

Validation checks:
- JSON format correctness
- Required fields: images, objects, width, height
- Images list has exactly 1 element
- bbox_2d format: [x1, y1, x2, y2] with x2 > x1, y2 > y1
- Bboxes within image bounds (warning if outside)
- Category names are non-empty strings
- Image files exist (optional, use --skip-image-check to disable)

Examples:

  # Full validation
  python validate_jsonl.py lvis/processed/train.jsonl
  
  # Skip image existence check (faster)
  python validate_jsonl.py lvis/processed/train.jsonl --skip-image-check
  
  # Verbose output
  python validate_jsonl.py lvis/processed/train.jsonl --verbose
        """
    )
    
    parser.add_argument(
        "jsonl_file",
        type=str,
        help="JSONL file to validate"
    )
    
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip checking if image files exist (faster)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    validator = JSONLValidator(
        check_images=not args.skip_image_check,
        verbose=args.verbose
    )
    
    success = validator.validate_file(args.jsonl_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

