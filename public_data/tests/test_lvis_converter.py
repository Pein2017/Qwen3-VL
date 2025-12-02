#!/usr/bin/env python3
"""
Test suite for LVIS converter.

Tests annotation parsing and format conversion without requiring images.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from public_data.converters.base import ConversionConfig
from public_data.converters.lvis_converter import LVISConverter


def test_annotation_loading():
    """Test LVIS annotation file loading."""
    print("\n" + "=" * 60)
    print("Test 1: LVIS Annotation Loading")
    print("=" * 60)

    annotation_path = "./lvis/raw/annotations/lvis_v1_train.json"

    # Create minimal config (images don't need to exist for this test)
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ConversionConfig(
            input_path=annotation_path,
            output_path=os.path.join(tmpdir, "test.jsonl"),
            image_root=tmpdir,  # Mock directory
            max_samples=5,
        )

        converter = LVISConverter(config, use_polygon=False)
        annotations = converter.load_annotations()

        print(f"  ✓ Loaded {len(annotations['images'])} images")
        print(f"  ✓ Loaded {len(annotations['annotations'])} annotations")
        print(f"  ✓ Loaded {len(annotations['categories'])} categories")

        # Check category distribution
        freq_dist = {}
        for cat in annotations["categories"]:
            freq = cat.get("frequency", "unknown")
            freq_dist[freq] = freq_dist.get(freq, 0) + 1

        print("\n  Category frequency distribution:")
        for freq, count in sorted(freq_dist.items()):
            print(f"    {freq}: {count} categories")

        return True


def test_bbox_conversion():
    """Test bbox-only conversion (without image files)."""
    print("\n" + "=" * 60)
    print("Test 2: BBox Conversion (Mock Images)")
    print("=" * 60)

    annotation_path = "./lvis/raw/annotations/lvis_v1_train.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_bbox.jsonl")

        config = ConversionConfig(
            input_path=annotation_path,
            output_path=output_path,
            image_root=tmpdir,  # Mock, images won't be checked in this phase
            max_samples=10,
            clip_boxes=True,
            skip_crowd=True,
            relative_image_paths=True,
        )

        converter = LVISConverter(config, use_polygon=False)

        # Load annotations
        data = converter.load_annotations()

        # Test a few samples manually
        print("\n  Testing first 3 image annotations:")
        sample_count = 0

        for image_id in list(converter.annotations_by_image.keys())[:3]:
            image_info = converter.image_map[image_id]
            annotations = converter.annotations_by_image[image_id]

            print(f"\n  Image {image_id}:")
            print(f"    Size: {image_info['width']}x{image_info['height']}")
            print(f"    Annotations: {len(annotations)}")

            # Test first annotation
            if annotations:
                ann = annotations[0]
                result = converter._convert_annotation(ann, image_info)

                if result:
                    print("    ✓ Converted annotation:")
                    print(f"      bbox_2d: {result[0]['bbox_2d']}")
                    print(f"      desc: '{result[0]['desc']}'")
                    sample_count += 1

        print(f"\n  ✓ Successfully converted {sample_count}/3 samples")
        return True


def test_polygon_conversion():
    """Test polygon conversion."""
    print("\n" + "=" * 60)
    print("Test 3: Polygon Conversion (N-point → poly)")
    print("=" * 60)

    annotation_path = "./lvis/raw/annotations/lvis_v1_train.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ConversionConfig(
            input_path=annotation_path,
            output_path=os.path.join(tmpdir, "test_poly.jsonl"),
            image_root=tmpdir,
            max_samples=20,  # More samples to find polygons
            clip_boxes=True,
            skip_crowd=True,
        )

        converter = LVISConverter(config, use_polygon=True)
        data = converter.load_annotations()

        # Find samples with segmentation
        print("\n  Analyzing segmentation polygons:")

        polygon_stats = {"total": 0, "point_distribution": {}}

        for image_id in list(converter.annotations_by_image.keys())[:20]:
            image_info = converter.image_map[image_id]
            annotations = converter.annotations_by_image[image_id]

            for ann in annotations:
                if "segmentation" in ann and isinstance(ann["segmentation"], list):
                    polygon_stats["total"] += 1

                    for part in ann["segmentation"]:
                        if isinstance(part, list):
                            num_points = len(part) // 2
                            key = f"{num_points}_points"
                            polygon_stats["point_distribution"][key] = (
                                polygon_stats["point_distribution"].get(key, 0) + 1
                            )

                            # Test conversion
                            result = converter._extract_polygons(
                                ann, "test_category", image_info
                            )
                            if result:
                                print(
                                    f"    ✓ Converted {num_points}-point polygon → poly"
                                )
                            break

        print("\n  Polygon statistics (first 20 images):")
        print(f"    Total segmentations: {polygon_stats['total']}")
        print("    Point distribution:")

        for key in sorted(
            polygon_stats["point_distribution"].keys(),
            key=lambda x: int(x.split("_")[0]),
        ):
            count = polygon_stats["point_distribution"][key]
            print(f"      {key}: {count}")

        return True


def test_qwen3vl_format_compliance():
    """Test output format compliance with Qwen3-VL schema."""
    print("\n" + "=" * 60)
    print("Test 4: Qwen3-VL Format Compliance")
    print("=" * 60)

    # Create sample output
    sample_bbox = {
        "images": ["train2017/000000001234.jpg"],
        "objects": [
            {"bbox_2d": [100.0, 200.0, 300.0, 400.0], "desc": "person"},
            {"bbox_2d": [50.0, 50.0, 150.0, 150.0], "desc": "car"},
        ],
        "width": 640,
        "height": 480,
    }

    sample_polygon = {
        "images": ["train2017/000000001234.jpg"],
        "objects": [
            {
                "poly": [100.0, 200.0, 300.0, 200.0, 300.0, 400.0, 100.0, 400.0],
                "poly_points": 4,
                "desc": "sign",
            },
            {
                "poly": [
                    50.0,
                    50.0,
                    60.0,
                    55.0,
                    70.0,
                    60.0,
                    80.0,
                    50.0,
                    75.0,
                    40.0,
                    65.0,
                    35.0,
                ],
                "poly_points": 6,
                "desc": "polygon",
            },
        ],
        "width": 640,
        "height": 480,
    }

    def validate_sample(sample, name):
        """Validate sample against schema."""
        errors = []

        # Required fields
        for field in ["images", "objects", "width", "height"]:
            if field not in sample:
                errors.append(f"Missing required field: {field}")

        # images validation
        if not isinstance(sample.get("images"), list):
            errors.append("'images' must be list")
        elif len(sample["images"]) != 1:
            errors.append(
                f"'images' should have 1 element, got {len(sample['images'])}"
            )

        # objects validation
        for i, obj in enumerate(sample.get("objects", [])):
            # Must have exactly one geometry field
            geom_fields = [k for k in obj.keys() if k in ["bbox_2d", "poly", "line"]]
            if len(geom_fields) != 1:
                errors.append(
                    f"Object {i}: need exactly 1 geometry field, got {geom_fields}"
                )

            # Must have desc
            if "desc" not in obj:
                errors.append(f"Object {i}: missing 'desc'")

            # Validate bbox_2d
            if "bbox_2d" in obj:
                bbox = obj["bbox_2d"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    errors.append(f"Object {i}: bbox_2d must be [x1,y1,x2,y2]")

            # Validate poly (N-point polygon)
            if "poly" in obj:
                poly = obj["poly"]
                if not isinstance(poly, list):
                    errors.append(f"Object {i}: poly must be list")
                elif len(poly) < 6:
                    errors.append(f"Object {i}: poly needs >=6 values (3 points)")
                elif len(poly) % 2 != 0:
                    errors.append(f"Object {i}: poly must have even number of values")

                # Check poly_points consistency
                if "poly_points" in obj:
                    if obj["poly_points"] * 2 != len(poly):
                        errors.append(f"Object {i}: poly_points mismatch")

        if errors:
            print(f"  ✗ {name} validation failed:")
            for err in errors:
                print(f"      • {err}")
            return False
        else:
            print(f"  ✓ {name}: VALID")
            return True

    bbox_valid = validate_sample(sample_bbox, "bbox-only sample")
    poly_valid = validate_sample(sample_polygon, "polygon sample")

    return bbox_valid and poly_valid


def main():
    """Run all tests."""
    print("=" * 60)
    print("LVIS Converter Test Suite")
    print("=" * 60)
    print("Testing with downloaded annotations (images not required)")

    tests = [
        ("Annotation Loading", test_annotation_loading),
        ("BBox Conversion", test_bbox_conversion),
        ("Polygon Conversion", test_polygon_conversion),
        ("Format Compliance", test_qwen3vl_format_compliance),
    ]

    results = {}

    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = "✓ PASS" if result else "✗ FAIL"
        except Exception as e:
            print(f"\n  ✗ Test failed with exception: {e}")
            import traceback

            traceback.print_exc()
            results[name] = "✗ ERROR"

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results.items():
        print(f"  {result} - {name}")

    passed = sum(1 for r in results.values() if "PASS" in r)
    total = len(results)

    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
