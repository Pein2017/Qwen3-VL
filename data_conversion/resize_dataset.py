from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image

# Use the project's EXIF utility to materialize orientation
try:
    from data_conversion.utils.exif_utils import apply_exif_orientation
except Exception:  # pragma: no cover - fallback, but we still fail fast if missing
    apply_exif_orientation = None  # type: ignore


LOGGER = logging.getLogger("resize_dataset")

# Progress bar (safe import)
try:  # pragma: no cover
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    def _tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else []


# Defaults per request
DEFAULT_FACTOR = 32
DEFAULT_MAX_BLOCKS = 512
DEFAULT_MIN_BLOCKS = 4
MAX_RATIO = 200  # keep parity with vision_process.smart_resize


@dataclass(frozen=True)
class ResizeConfig:
    input_dir: Path
    output_dir: Path
    factor: int
    max_blocks: int
    min_blocks: int
    jpeg_quality: int
    fail_on_size_mismatch: bool

    @property
    def max_pixels(self) -> int:
        return int(self.max_blocks * self.factor * self.factor)

    @property
    def min_pixels(self) -> int:
        return int(self.min_blocks * self.factor * self.factor)


def parse_args(argv: Iterable[str]) -> ResizeConfig:
    parser = argparse.ArgumentParser(
        description="Resize BBU dataset images to multiples of factor and rewrite JSONLs with scaled geometry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--factor", type=int, default=DEFAULT_FACTOR)
    parser.add_argument("--max_blocks", type=int, default=DEFAULT_MAX_BLOCKS)
    parser.add_argument("--min_blocks", type=int, default=DEFAULT_MIN_BLOCKS)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument(
        "--fail_on_size_mismatch", action="store_true", default=True,
        help="If set, raises when PIL image size after EXIF doesn't match JSON width/height.",
    )
    parser.add_argument(
        "--no_fail_on_size_mismatch", dest="fail_on_size_mismatch", action="store_false",
        help="Disable strict size match validation (not recommended).",
    )

    args = parser.parse_args(list(argv))

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        raise ValueError(f"input_dir does not exist: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("output_dir must be different from input_dir to avoid overwriting")

    output_dir.mkdir(parents=True, exist_ok=True)

    return ResizeConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        factor=int(args.factor),
        max_blocks=int(args.max_blocks),
        min_blocks=int(args.min_blocks),
        jpeg_quality=int(args.jpeg_quality),
        fail_on_size_mismatch=bool(args.fail_on_size_mismatch),
    )


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def compute_target_size(h: int, w: int, factor: int, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    """Smart resize like vision_process.smart_resize but configurable.

    - h', w' divisible by factor
    - area in [min_pixels, max_pixels]
    - roughly preserve aspect ratio
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size, expected positive dims, got h={h}, w={w}")
    if max(h, w) / min(h, w) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(h, w) / min(h, w):.4f}"
        )

    h_bar = max(factor, round_by_factor(h, factor))
    w_bar = max(factor, round_by_factor(w, factor))

    area = h_bar * w_bar
    if area > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, floor_by_factor(int(h / beta), factor))
        w_bar = max(factor, floor_by_factor(int(w / beta), factor))
    elif area < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = ceil_by_factor(int(h * beta), factor)
        w_bar = ceil_by_factor(int(w * beta), factor)

    return int(h_bar), int(w_bar)


def _load_image_apply_exif(image_path: Path) -> Image.Image:
    if apply_exif_orientation is None:
        raise RuntimeError("EXIF orientation utility missing; cannot proceed")
    with Image.open(image_path) as img:
        img = apply_exif_orientation(img)
        # Convert to RGB to avoid mode surprises
        return img.convert("RGB")


def _clamp(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


def _scale_points(points: List[float | int], sx: float, sy: float, w: int, h: int) -> List[int]:
    out: List[int] = []
    for i, v in enumerate(points):
        if i % 2 == 0:
            # x
            nv = int(round(float(v) * sx))
            out.append(_clamp(nv, 0, w - 1))
        else:
            # y
            nv = int(round(float(v) * sy))
            out.append(_clamp(nv, 0, h - 1))
    return out


def _scale_bbox2d(bbox: List[float | int], sx: float, sy: float, w: int, h: int) -> List[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"bbox_2d must be length=4, got {bbox}")
    x1, y1, x2, y2 = _scale_points(list(bbox), sx, sy, w, h)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    # Ensure at least 1px extent if possible
    if x1 == x2 and x2 < w - 1:
        x2 = x1 + 1
    if y1 == y2 and y2 < h - 1:
        y2 = y1 + 1
    return [x1, y1, x2, y2]


def _canonicalize_quad(points8: List[int | float]) -> List[int]:
    # Ported from vis_tools/vis_helper.canonicalize_quad
    if not isinstance(points8, (list, tuple)) or len(points8) != 8:
        return [int(round(v)) for v in (points8 or [])]
    pts = [(float(points8[i]), float(points8[i + 1])) for i in range(0, 8, 2)]
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0

    def classify_corner(p: Tuple[float, float]) -> Tuple[int, float]:
        x, y = p
        if x <= cx and y <= cy:
            return (0, -(x + y))
        elif x >= cx and y <= cy:
            return (1, x - y)
        elif x >= cx and y >= cy:
            return (2, x + y)
        else:
            return (3, -x + y)

    sorted_pts = sorted(pts, key=classify_corner)
    if len({classify_corner(p)[0] for p in sorted_pts}) != 4:
        sorted_by_y = sorted(pts, key=lambda p: p[1])
        top = sorted(sorted_by_y[:2], key=lambda p: p[0])
        bottom = sorted(sorted_by_y[2:], key=lambda p: p[0])
        sorted_pts = [top[0], top[1], bottom[1], bottom[0]]
    return [int(round(v)) for xy in sorted_pts for v in xy]


def _scale_quad(points8: List[float | int], sx: float, sy: float, w: int, h: int) -> List[int]:
    if not isinstance(points8, (list, tuple)) or len(points8) != 8:
        raise ValueError(f"quad must be length=8, got {points8}")
    scaled = _scale_points(list(points8), sx, sy, w, h)
    return _canonicalize_quad(scaled)


def _scale_line(points: List[float | int], sx: float, sy: float, w: int, h: int) -> List[int]:
    if not isinstance(points, (list, tuple)) or len(points) < 4 or len(points) % 2 != 0:
        raise ValueError(f"line must be even length >= 4, got {points}")
    return _scale_points(list(points), sx, sy, w, h)


def _detect_geometry_key(obj: Dict[str, Any]) -> str:
    keys = [k for k in ("bbox_2d", "quad", "line") if k in obj]
    if len(keys) != 1:
        raise ValueError(f"object must contain exactly one geometry key, got {keys}")
    return keys[0]


def _resize_and_save_image(cfg: ResizeConfig, image_rel: str, json_w: int, json_h: int,
                           dims_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Resize image if not in cache; return {old_w,old_h,new_w,new_h,sx,sy}.

    Image is read from input_dir/images/... and written to output_dir/images/...
    """
    if image_rel in dims_cache:
        return dims_cache[image_rel]

    in_path = (cfg.input_dir / image_rel).resolve()
    out_path = (cfg.output_dir / image_rel).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = _load_image_apply_exif(in_path)
    ow, oh = img.size
    if cfg.fail_on_size_mismatch and (ow != json_w or oh != json_h):
        raise ValueError(
            f"Image size mismatch after EXIF for {image_rel}. Expected from JSON {json_w}x{json_h}, actual {ow}x{oh}."
        )

    nh, nw = compute_target_size(oh, ow, cfg.factor, cfg.min_pixels, cfg.max_pixels)
    sx = float(nw) / float(ow)
    sy = float(nh) / float(oh)

    # Save resized image
    if not out_path.exists():
        resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        resized.save(out_path, format="JPEG", quality=cfg.jpeg_quality)

    info = {"old_w": ow, "old_h": oh, "new_w": nw, "new_h": nh, "sx": sx, "sy": sy}
    dims_cache[image_rel] = info
    return info


def _scale_objects(objects: List[Dict[str, Any]], sx: float, sy: float, w: int, h: int) -> List[Dict[str, Any]]:
    scaled_objects: List[Dict[str, Any]] = []
    for obj in objects:
        gkey = _detect_geometry_key(obj)
        new_obj = dict(obj)
        if gkey == "bbox_2d":
            new_obj["bbox_2d"] = _scale_bbox2d(obj[gkey], sx, sy, w, h)
        elif gkey == "quad":
            new_obj["quad"] = _scale_quad(obj[gkey], sx, sy, w, h)
        elif gkey == "line":
            new_obj["line"] = _scale_line(obj[gkey], sx, sy, w, h)
        scaled_objects.append(new_obj)
    return scaled_objects


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _count_nonempty_lines(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def _process_jsonl(
    cfg: ResizeConfig,
    src_jsonl: Path,
    dst_jsonl: Path,
    dims_cache: Dict[str, Dict[str, Any]],
    geom_cache: Dict[str, List[Dict[str, Any]]],
    mutate_geom: bool,
) -> None:
    """Process a JSONL file.

    mutate_geom:
      - True: compute & write scaled geometry; populate geom_cache
      - False: reuse from geom_cache if available; otherwise compute on the fly
    """
    records_out: List[Dict[str, Any]] = []
    count = 0
    total = _count_nonempty_lines(src_jsonl)
    for rec in _tqdm(
        _read_jsonl(src_jsonl), total=total, desc=f"{src_jsonl.name}", unit="rec", leave=False
    ):
        images = rec.get("images")
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(f"Record must have exactly one image path, got: {images}")
        image_rel = images[0]
        json_w = int(rec.get("width"))
        json_h = int(rec.get("height"))

        dims = _resize_and_save_image(cfg, image_rel, json_w, json_h, dims_cache)
        sx, sy, nw, nh = dims["sx"], dims["sy"], dims["new_w"], dims["new_h"]

        new_rec = dict(rec)
        new_rec["width"] = nw
        new_rec["height"] = nh

        if not mutate_geom and image_rel in geom_cache:
            new_rec["objects"] = geom_cache[image_rel]
        else:
            objs = rec.get("objects") or []
            scaled_objs = _scale_objects(objs, sx, sy, nw, nh)
            new_rec["objects"] = scaled_objs
            # Populate geom cache to avoid re-scaling for duplicates across files
            geom_cache[image_rel] = scaled_objs

        records_out.append(new_rec)
        count += 1

    _write_jsonl(dst_jsonl, records_out)
    LOGGER.info(f"Processed {count} records: {src_jsonl} -> {dst_jsonl}")


def main(argv: Iterable[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = parse_args(argv)

    images_dir_in = (cfg.input_dir / "images").resolve()
    if not images_dir_in.exists():
        raise ValueError(f"Expected images directory under input_dir: {images_dir_in}")

    images_dir_out = (cfg.output_dir / "images").resolve()
    images_dir_out.mkdir(parents=True, exist_ok=True)

    dims_cache: Dict[str, Dict[str, Any]] = {}
    geom_cache: Dict[str, List[Dict[str, Any]]] = {}
    cache_path = cfg.output_dir / "_resize_cache.json"

    # File set to consider
    all_jsonl = cfg.input_dir / "all_samples.jsonl"
    train_jsonl = cfg.input_dir / "train.jsonl"
    val_jsonl = cfg.input_dir / "val.jsonl"
    teacher_jsonl = cfg.input_dir / "teacher_pool.jsonl"

    # 1) If all_samples exists, process it first and cache geometry
    if all_jsonl.exists():
        _process_jsonl(
            cfg,
            src_jsonl=all_jsonl,
            dst_jsonl=cfg.output_dir / all_jsonl.name,
            dims_cache=dims_cache,
            geom_cache=geom_cache,
            mutate_geom=True,
        )
        # Persist cache (dims + we persist which images were processed)
        to_save = {
            "dims_cache": dims_cache,
            "geom_images": list(geom_cache.keys()),
            "factor": cfg.factor,
            "max_blocks": cfg.max_blocks,
            "min_blocks": cfg.min_blocks,
        }
        cache_path.write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        LOGGER.warning("all_samples.jsonl not found; will compute geometry on each file.")

    # 2) Process other JSONLs, reusing geom cache
    for src in (train_jsonl, val_jsonl, teacher_jsonl):
        if src.exists():
            _process_jsonl(
                cfg,
                src_jsonl=src,
                dst_jsonl=cfg.output_dir / src.name,
                dims_cache=dims_cache,
                geom_cache=geom_cache,
                mutate_geom=False,
            )
        else:
            LOGGER.info(f"Skip absent file: {src}")

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as e:
        # Fail fast with actionable message
        LOGGER.error(f"Resize dataset failed: {e}")
        raise


