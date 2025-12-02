#!/usr/bin/env python3
"""
Smoke test for polygon cap functionality.

Tests that polygons with more than poly_max_points vertices are converted to bbox_2d.
Verifies alignment with DATA_JSONL_CONTRACT.md (only bbox_2d and poly, no line/quad).
"""

import json
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from public_data.converters.base import ConversionConfig
from public_data.converters.lvis_converter import LVISConverter


def create_mock_annotation_with_polygons():
    """Create a mock LVIS annotation with polygons of varying vertex counts."""
    return {
        "images": [
            {"id": 1, "file_name": "test1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "test2.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            # Small polygon (4 points) - should stay as poly
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 100, 100],
                "segmentation": [[10, 10, 110, 10, 110, 110, 10, 110]],  # 4 points
            },
            # Medium polygon (8 points) - should stay as poly
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [120, 10, 220, 100],
                "segmentation": [
                    [
                        120,
                        10,
                        140,
                        10,
                        160,
                        20,
                        180,
                        30,
                        200,
                        40,
                        220,
                        50,
                        220,
                        110,
                        120,
                        110,
                    ]
                ],  # 8 points
            },
            # Large polygon (16 points) - should be converted to bbox_2d (cap=12)
            {
                "id": 3,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 120, 200, 220],
                "segmentation": [
                    [
                        10,
                        120,
                        30,
                        120,
                        50,
                        125,
                        70,
                        130,
                        90,
                        135,
                        110,
                        140,
                        130,
                        145,
                        150,
                        150,
                        170,
                        155,
                        190,
                        160,
                        200,
                        165,
                        200,
                        185,
                        200,
                        205,
                        200,
                        220,
                        10,
                        220,
                    ]
                ],  # 15 points (30 coords)
            },
            # Very large polygon (20 points) - should be converted to bbox_2d (cap=12)
            {
                "id": 4,
                "image_id": 2,
                "category_id": 1,
                "bbox": [10, 10, 300, 200],
                "segmentation": [
                    [
                        10,
                        10,
                        20,
                        10,
                        30,
                        15,
                        40,
                        20,
                        50,
                        25,
                        60,
                        30,
                        70,
                        35,
                        80,
                        40,
                        90,
                        45,
                        100,
                        50,
                        200,
                        60,
                        250,
                        70,
                        280,
                        80,
                        290,
                        90,
                        300,
                        100,
                        300,
                        150,
                        300,
                        200,
                        10,
                        200,
                    ]
                ],  # 18 points (36 coords)
            },
        ],
        "categories": [{"id": 1, "name": "test_object", "frequency": "common"}],
    }


def test_poly_cap_functionality():
    """Test that polygons exceeding cap are converted to bbox_2d."""
    print("=" * 60)
    print("Test 1: Polygon Cap Functionality")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock annotation file
        ann_path = Path(tmpdir) / "test_annotations.json"
        mock_data = create_mock_annotation_with_polygons()
        with ann_path.open("w") as f:
            json.dump(mock_data, f)

        # Create mock image directory
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        for img in mock_data["images"]:
            (img_dir / img["file_name"]).touch()

        # Test with cap=12
        config = ConversionConfig(
            input_path=str(ann_path),
            output_path=str(Path(tmpdir) / "output.jsonl"),
            image_root=str(img_dir),
            split="train",
        )

        converter = LVISConverter(config, use_polygon=True, poly_max_points=12)
        converter.load_annotations()

        results = {}
        for image_id in [1, 2]:
            image_info = converter.image_map[image_id]
            annotations = converter.annotations_by_image[image_id]

            for ann in annotations:
                ann_id = ann["id"]
                result = converter._convert_annotation(ann, image_info)

                if result:
                    obj = result[0] if isinstance(result, list) else result
                    has_poly = "poly" in obj
                    has_bbox = "bbox_2d" in obj
                    num_points = None

                    if has_poly:
                        num_points = obj.get("poly_points", len(obj["poly"]) // 2)

                    results[ann_id] = {
                        "has_poly": has_poly,
                        "has_bbox": has_bbox,
                        "num_points": num_points,
                        "desc": obj.get("desc", ""),
                    }

        # Verify results
        print("\n  Results:")
        for ann_id, res in sorted(results.items()):
            print(f"    Annotation {ann_id}:")
            print(f"      Has poly: {res['has_poly']}")
            print(f"      Has bbox: {res['has_bbox']}")
            if res["num_points"]:
                print(f"      Points: {res['num_points']}")

        # Expected: ann_id 1 (4 points) -> poly, ann_id 2 (8 points) -> poly,
        #           ann_id 3 (15 points) -> bbox_2d, ann_id 4 (18 points) -> bbox_2d
        expected = {
            1: {"has_poly": True, "has_bbox": False},
            2: {"has_poly": True, "has_bbox": False},
            3: {"has_poly": False, "has_bbox": True},
            4: {"has_poly": False, "has_bbox": True},
        }

        all_passed = True
        for ann_id, exp in expected.items():
            if ann_id not in results:
                print(f"  ✗ Annotation {ann_id} not found in results")
                all_passed = False
                continue

            res = results[ann_id]
            if res["has_poly"] != exp["has_poly"] or res["has_bbox"] != exp["has_bbox"]:
                print(f"  ✗ Annotation {ann_id} mismatch:")
                print(f"      Expected: poly={exp['has_poly']}, bbox={exp['has_bbox']}")
                print(f"      Got: poly={res['has_poly']}, bbox={res['has_bbox']}")
                all_passed = False
            else:
                print(f"  ✓ Annotation {ann_id}: correct")

        # Check stats
        print("\n  Stats:")
        print(f"    poly_converted: {converter.stats.get('poly_converted', 0)}")
        print(
            f"    poly_to_bbox_capped: {converter.stats.get('poly_to_bbox_capped', 0)}"
        )
        print(f"    polygon_skipped: {converter.stats.get('polygon_skipped', 0)}")

        if converter.stats.get("poly_to_bbox_capped", 0) != 2:
            print(
                f"  ✗ Expected 2 capped polygons, got {converter.stats.get('poly_to_bbox_capped', 0)}"
            )
            all_passed = False

        return all_passed


def test_data_contract_compliance():
    """Test that output conforms to DATA_JSONL_CONTRACT.md."""
    print("\n" + "=" * 60)
    print("Test 2: Data Contract Compliance")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        ann_path = Path(tmpdir) / "test_annotations.json"
        mock_data = create_mock_annotation_with_polygons()
        with ann_path.open("w") as f:
            json.dump(mock_data, f)

        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        for img in mock_data["images"]:
            (img_dir / img["file_name"]).touch()

        config = ConversionConfig(
            input_path=str(ann_path),
            output_path=str(Path(tmpdir) / "output.jsonl"),
            image_root=str(img_dir),
            split="train",
        )

        converter = LVISConverter(config, use_polygon=True, poly_max_points=12)
        converter.convert()

        # Read output
        output_path = Path(tmpdir) / "output.jsonl"
        if not output_path.exists():
            print("  ✗ Output file not created")
            return False

        errors = []
        with output_path.open("r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue

                # Check required fields
                for field in ["images", "objects", "width", "height"]:
                    if field not in record:
                        errors.append(
                            f"Line {line_num}: Missing required field '{field}'"
                        )

                # Check objects
                for obj_idx, obj in enumerate(record.get("objects", [])):
                    # Must have exactly one geometry field (bbox_2d or poly, no line/quad)
                    geom_fields = [
                        k
                        for k in obj.keys()
                        if k in ["bbox_2d", "poly", "line", "quad"]
                    ]
                    if len(geom_fields) != 1:
                        errors.append(
                            f"Line {line_num}, object {obj_idx}: Need exactly 1 geometry field, got {geom_fields}"
                        )

                    if "line" in obj:
                        errors.append(
                            f"Line {line_num}, object {obj_idx}: 'line' geometry not allowed for LVIS data"
                        )

                    if "quad" in obj:
                        errors.append(
                            f"Line {line_num}, object {obj_idx}: 'quad' geometry is deprecated (use 'poly')"
                        )

                    # Must have desc
                    if "desc" not in obj or not obj["desc"]:
                        errors.append(
                            f"Line {line_num}, object {obj_idx}: Missing or empty 'desc'"
                        )

                    # Validate poly format
                    if "poly" in obj:
                        poly = obj["poly"]
                        if not isinstance(poly, list):
                            errors.append(
                                f"Line {line_num}, object {obj_idx}: 'poly' must be list"
                            )
                        elif len(poly) < 6:
                            errors.append(
                                f"Line {line_num}, object {obj_idx}: 'poly' needs >=6 values (3 points)"
                            )
                        elif len(poly) % 2 != 0:
                            errors.append(
                                f"Line {line_num}, object {obj_idx}: 'poly' must have even number of values"
                            )

                        # Check poly_points consistency
                        if "poly_points" in obj:
                            expected_points = len(poly) // 2
                            if obj["poly_points"] != expected_points:
                                errors.append(
                                    f"Line {line_num}, object {obj_idx}: poly_points mismatch "
                                    f"(expected {expected_points}, got {obj['poly_points']})"
                                )

                    # Validate bbox_2d format
                    if "bbox_2d" in obj:
                        bbox = obj["bbox_2d"]
                        if not isinstance(bbox, list) or len(bbox) != 4:
                            errors.append(
                                f"Line {line_num}, object {obj_idx}: bbox_2d must be [x1,y1,x2,y2]"
                            )

        if errors:
            print("  ✗ Contract violations found:")
            for err in errors[:10]:  # Show first 10 errors
                print(f"      • {err}")
            if len(errors) > 10:
                print(f"      ... and {len(errors) - 10} more")
            return False
        else:
            print("  ✓ All records comply with DATA_JSONL_CONTRACT.md")
            return True


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Polygon Cap Smoke Tests")
    print("=" * 60)

    tests = [
        ("Polygon Cap Functionality", test_poly_cap_functionality),
        ("Data Contract Compliance", test_data_contract_compliance),
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
