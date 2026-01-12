from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Dict, List

from PIL import Image, ImageDraw

from src.datasets.augment import apply_augmentations
from src.datasets.augmentation import ops as _register_ops  # noqa: F401
from src.datasets.augmentation.builder import build_compose_from_config

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
GOLDEN_DIR = ROOT / "golden"


def _make_base_images() -> List[Path]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    size = (80, 64)

    # Image 0: gray background with colored shapes
    img0 = Image.new("RGB", size, (120, 120, 120))
    d0 = ImageDraw.Draw(img0)
    d0.rectangle([10, 12, 26, 30], outline=(255, 0, 0), width=2)
    d0.polygon([40, 15, 52, 14, 55, 25, 42, 26], outline=(0, 255, 0), fill=(0, 255, 0))
    d0.line([15, 40, 50, 45, 60, 55], fill=(0, 0, 255), width=2)
    path0 = INPUT_DIR / "scene0_img0.png"
    img0.save(path0)
    paths.append(path0)

    # Image 1: slight variation for multi-image handling
    img1 = Image.new("RGB", size, (110, 110, 110))
    d1 = ImageDraw.Draw(img1)
    d1.rectangle([12, 14, 28, 32], outline=(255, 64, 0), width=2)
    d1.polygon(
        [42, 17, 54, 16, 57, 27, 44, 28], outline=(0, 200, 80), fill=(0, 200, 80)
    )
    d1.line([17, 42, 52, 47, 62, 57], fill=(0, 0, 220), width=2)
    path1 = INPUT_DIR / "scene0_img1.png"
    img1.save(path1)
    paths.append(path1)

    return paths


def _build_record(img_paths: List[Path]) -> Dict[str, Any]:
    return {
        "images": [str(p.relative_to(ROOT)) for p in img_paths],
        "width": 80,
        "height": 64,
        "objects": [
            {
                "bbox_2d": [10.0, 12.0, 26.0, 30.0],
                "desc": "类别=紧固件,可见性=完整,备注=螺丝/fastener",
            },
            {
                "poly": [40.0, 15.0, 52.0, 14.0, 55.0, 25.0, 42.0, 26.0],
                # Use this as the ROI-crop anchor object.
                "desc": "类别=BBU设备,可见性=完整,备注=金属板/plate",
            },
            {
                "line": [15.0, 40.0, 50.0, 45.0, 60.0, 55.0],
                "desc": "类别=线缆,可见性=完整,备注=线缆/cable",
            },
        ],
    }


def _build_pipeline():
    cfg = {
        "ops": [
            {"name": "hflip", "params": {"prob": 0.9}},
            {
                "name": "roi_crop",
                "params": {
                    "anchor_classes": ["BBU设备"],
                    "scale_range": [1.2, 1.8],
                    "min_crop_size": 32,
                    "min_coverage": 0.25,
                    "completeness_threshold": 0.9,
                    "prob": 1.0,
                },
            },
            {"name": "rotate", "params": {"max_deg": 8.0, "prob": 1.0}},
            {"name": "scale", "params": {"lo": 0.9, "hi": 1.1, "prob": 1.0}},
            {
                "name": "resize_by_scale",
                "params": {"lo": 0.9, "hi": 1.1, "align_multiple": 8, "prob": 1.0},
            },
            {
                "name": "color_jitter",
                "params": {
                    "brightness": [0.8, 1.2],
                    "contrast": [0.85, 1.15],
                    "saturation": [0.8, 1.3],
                    "prob": 0.6,
                },
            },
            {"name": "gamma", "params": {"gamma": [0.8, 1.2], "prob": 0.5}},
            {
                "name": "expand_to_fit_affine",
                "params": {"multiple": 16, "max_pixels": 262144},
            },
        ]
    }
    return build_compose_from_config(cfg)


def _extract_geoms(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    geoms: List[Dict[str, Any]] = []
    for obj in record.get("objects", []):
        for key in ("bbox_2d", "poly", "line"):
            if key in obj:
                geoms.append({key: obj[key]})
                break
    return geoms


def _save_images(img_dicts: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, entry in enumerate(img_dicts):
        b = entry.get("bytes")
        if not isinstance(b, (bytes, bytearray)):
            raise TypeError("image bytes missing in augmented output")
        path = out_dir / f"image{idx}.png"
        with open(path, "wb") as f:
            f.write(b)


def _telemetry_to_dict(telemetry: Any) -> Dict[str, Any]:
    if telemetry is None:
        return {}
    return {
        "kept_indices": list(getattr(telemetry, "kept_indices", []) or []),
        "coverages": list(getattr(telemetry, "coverages", []) or []),
        "allows_geometry_drops": bool(
            getattr(telemetry, "allows_geometry_drops", False)
        ),
        "width": getattr(telemetry, "width", None),
        "height": getattr(telemetry, "height", None),
        "padding_ratio": getattr(telemetry, "padding_ratio", None),
        "skip_reason": getattr(telemetry, "skip_reason", None),
        "skip_counts": dict(getattr(telemetry, "skip_counts", {}) or {}),
    }


def generate(seeds: List[int] | None = None) -> None:
    seeds = seeds or [7, 42, 1337, 2024]
    img_paths = _make_base_images()
    record = _build_record(img_paths)
    geoms = _extract_geoms(record)

    # Persist input record for traceability
    with open(ROOT / "input_record.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    for seed in seeds:
        rng = random.Random(seed)
        pipeline = _build_pipeline()
        images = [Image.open(p).convert("RGB") for p in img_paths]
        out_imgs, out_geoms = apply_augmentations(images, geoms, pipeline, rng=rng)
        out_dir = GOLDEN_DIR / f"seed_{seed}"
        _save_images(out_imgs, out_dir)
        with open(out_dir / "geometries.json", "w", encoding="utf-8") as f:
            json.dump(out_geoms, f, ensure_ascii=False, indent=2)
        telemetry = getattr(pipeline, "last_summary", None)
        with open(out_dir / "telemetry.json", "w", encoding="utf-8") as f:
            json.dump(_telemetry_to_dict(telemetry), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    generate()
