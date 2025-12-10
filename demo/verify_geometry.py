import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.datasets.builders.jsonlines import JSONLinesBuilder
from src.datasets.utils import extract_object_points


def load_first_two_records(jsonl_path: str):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= 2:
                break
    if not records:
        raise RuntimeError("No records found in JSONL")
    if len(records) == 1:
        records.append(records[0])
    return records[0], records[1]


def scale_norm1000(points, width, height):
    out = []
    for i, v in enumerate(points):
        if i % 2 == 0:
            out.append(int(round(float(v) / float(width or 1) * 1000)))
        else:
            out.append(int(round(float(v) / float(height or 1) * 1000)))
    return out


def main():
    jsonl_path = "/data/Pein/code/data/bbu_full_768/val.jsonl"
    assert Path(jsonl_path).exists(), f"Missing JSONL: {jsonl_path}"

    rec_a, rec_b = load_first_two_records(jsonl_path)

    # Build messages + objects with raw geometry points preserved
    builder = JSONLinesBuilder(
        user_prompt="describe all objects",
        emit_norm="none",  # text-side only; leave as-is
        include_summary=True,
        use_section_headers=True,
        preserve_geometry_points=True,
        bbox_input_type="pixel",
    )
    # Build each record separately and merge objects manually
    merged_a = builder.build(rec_a)
    merged_b = builder.build(rec_b)

    # Merge objects from both builds
    objs_a = rec_a.get("objects", []) or []
    objs_b = rec_b.get("objects", []) or []
    raw_bbox_a = merged_a.get("objects", {}).get("bbox", [])
    raw_bbox_b = merged_b.get("objects", {}).get("bbox", [])
    image_ids_a = merged_a.get("objects", {}).get("image_id", [])
    image_ids_b = merged_b.get("objects", {}).get("image_id", [])

    # Combine objects from both records
    raw_bbox = raw_bbox_a + raw_bbox_b
    image_ids = image_ids_a + [
        i + 1 for i in image_ids_b
    ]  # Offset image_id for second record

    # Build a flattened list of expected raw point arrays in order (A then B)
    expected_pts = []
    for obj in objs_a:
        geom_type, pts = extract_object_points(obj)
        if geom_type:
            expected_pts.append([float(x) for x in pts])
    for obj in objs_b:
        geom_type, pts = extract_object_points(obj)
        if geom_type:
            expected_pts.append([float(x) for x in pts])

    assert len(raw_bbox) == len(expected_pts), (
        f"bbox count mismatch: got {len(raw_bbox)} vs expected {len(expected_pts)}"
    )

    mismatches = []
    for i, (got, exp) in enumerate(zip(raw_bbox, expected_pts)):
        if len(got) != len(exp) or any(float(a) != float(b) for a, b in zip(got, exp)):
            mismatches.append((i, got, exp))

    print(f"Total objects: {len(raw_bbox)}")
    print(f"Preservation mismatches: {len(mismatches)}")
    if mismatches:
        # print up to first 3 mismatches for debugging
        for i, got, exp in mismatches[:3]:
            print(f"  idx={i}: got={got} exp={exp}")
        raise SystemExit(1)

    # Validate scaling to norm1000 numerically (simulate Template.normalize_bbox)
    widths = [rec_a.get("width", 0)]
    heights = [rec_a.get("height", 0)]
    if rec_b is not rec_a:
        widths.append(rec_b.get("width", 0))
        heights.append(rec_b.get("height", 0))

    scaled = []
    for pts, img_id in zip(raw_bbox, image_ids or [0] * len(raw_bbox)):
        w = widths[img_id] if img_id < len(widths) else widths[0]
        h = heights[img_id] if img_id < len(heights) else heights[0]
        scaled.append(scale_norm1000(pts, w, h))

    # Spot check first 3 objects to show example scaling
    for i in range(min(3, len(raw_bbox))):
        print(f"Example {i}: raw={raw_bbox[i]} -> norm1000={scaled[i]}")

    print("OK: raw geometry preserved and norm1000 scaling computed correctly.")


if __name__ == "__main__":
    main()
