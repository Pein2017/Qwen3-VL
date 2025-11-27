from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
from PIL import Image

from data_conversion.utils.exif_utils import apply_exif_orientation  # noqa: E402
from vis_tools.vis_helper import (  # noqa: E402
    canonicalize_poly,
    draw_objects,
    generate_colors,
)

try:  # pragma: no cover
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover

    def _tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else []


# ============================================================================
# Configuration
# ============================================================================
input_path = "data/bbu_full_768_poly-need_review/all_samples.jsonl"  # Path to JSONL (e.g., data/bbu_full_768/all_samples.jsonl)
output_dir = "vis_out/debug"  # Directory to save visualizations
limit = 20  # Max number of samples to visualize (0 means all)
dpi = 120  # Matplotlib savefig DPI
color_by = "type"  # Color by object type or full desc prefix ("type" or "desc")
# Filter by image filename (set to None to visualize all, or a substring to match)
filter_image_name = "QC-20231218-0025165_4127784"  # Set to None to disable filtering
filter_image_name = None


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def to_vis_objects(
    sample_objects: List[Dict[str, Any]], color_by: str
) -> List[Dict[str, Any]]:
    vis_objects: List[Dict[str, Any]] = []
    for obj in sample_objects or []:
        if "bbox_2d" in obj:
            gtype = "bbox_2d"
            pts = obj["bbox_2d"]
        elif "poly" in obj:
            gtype = "poly"
            pts = obj.get("poly")
            if pts:
                pts = canonicalize_poly(pts)
        elif "line" in obj:
            gtype = "line"
            pts = obj["line"]
        else:
            raise ValueError(f"Object missing geometry: {obj}")

        # Derive a friendly label for coloring
        desc_full = obj.get("desc", "")
        if color_by == "type":
            label = desc_full.split("/")[0] if desc_full else gtype
        else:
            label = desc_full or gtype

        vis_objects.append({"type": gtype, "points": pts, "desc": label})
    return vis_objects


def visualize(
    jsonl_path: Path,
    out_dir: Path,
    limit: int,
    dpi: int,
    color_by: str,
    filter_image_name: Optional[str] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = jsonl_path.parent

    count = 0
    found_samples = 0
    for idx, rec in enumerate(
        _tqdm(read_jsonl(jsonl_path), desc=jsonl_path.name, unit="rec")
    ):
        if limit and count >= limit:
            break

        images = rec.get("images")
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(f"Record must have exactly one image path, got: {images}")
        image_rel = images[0]

        # Filter by image filename if specified
        if filter_image_name is not None:
            if filter_image_name not in image_rel:
                continue

        image_path = (base_dir / image_rel).resolve()

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        img = apply_exif_orientation(Image.open(image_path))
        obj_list = to_vis_objects(rec.get("objects", []), color_by=color_by)
        labels = [o["desc"] for o in obj_list]
        color_map = generate_colors(labels)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        try:
            draw_objects(ax, img, obj_list, color_map, scaled=True)
            # Add more info to title
            title = f"{image_rel}\nWidth: {rec.get('width', '?')} x Height: {rec.get('height', '?')} | Objects: {len(obj_list)}"
            ax.set_title(title, fontsize=10)
            fig.tight_layout()
            # Use image filename in output name for easier identification
            image_stem = Path(image_rel).stem
            out_path = out_dir / f"vis_{image_stem}.jpg"
            fig.savefig(out_path, dpi=dpi)
            print(f"Saved visualization: {out_path}")
        finally:
            plt.close(fig)

        count += 1
        found_samples += 1
        # If filtering and we found the sample, we can break early
        if filter_image_name is not None and found_samples > 0:
            break

    if filter_image_name is not None and found_samples == 0:
        print(f"Warning: No samples found matching '{filter_image_name}'")


def main() -> int:
    jsonl_path = Path(input_path).resolve()
    if not jsonl_path.exists():
        raise ValueError(f"JSONL not found: {jsonl_path}")

    out_dir = Path(output_dir).resolve()
    visualize(
        jsonl_path=jsonl_path,
        out_dir=out_dir,
        limit=limit,
        dpi=dpi,
        color_by=color_by,
        filter_image_name=filter_image_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
