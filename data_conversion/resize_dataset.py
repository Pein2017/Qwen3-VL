from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from src.datasets.preprocessors.resize import Resizer, SmartResizeParams

# Use the project's EXIF utility to materialize orientation
try:  # pragma: no cover
    from data_conversion.utils.exif_utils import apply_exif_orientation
except Exception:  # pragma: no cover
    apply_exif_orientation = None  # type: ignore

# Progress bar (safe import)
try:  # pragma: no cover
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    def _tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else []


LOGGER = logging.getLogger("resize_dataset")

# Defaults per request
DEFAULT_FACTOR = 32
DEFAULT_MAX_PIXEL_BLOCKS = 768
DEFAULT_MIN_PIXEL_BLOCKS = 4


@dataclass(frozen=True)
class ResizeConfig:
    input_dir: Path
    output_dir: Path
    factor: int
    max_pixel_blocks: int
    min_pixel_blocks: int
    jpeg_quality: int
    fail_on_size_mismatch: bool
    images_dir_override: Path | None = None
    num_workers: int = 8

    @property
    def max_pixels(self) -> int:
        return int(self.max_pixel_blocks * self.factor * self.factor)

    @property
    def min_pixels(self) -> int:
        return int(self.min_pixel_blocks * self.factor * self.factor)


def parse_args(argv: Iterable[str]) -> ResizeConfig:
    parser = argparse.ArgumentParser(
        description="Resize dataset images to multiples of factor and rewrite JSONLs with scaled geometry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--images_dir",
        type=str,
        help="Override images directory (defaults to <input_dir>/images).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker threads for resizing and geometry scaling.",
    )
    parser.add_argument("--factor", type=int, default=DEFAULT_FACTOR)
    parser.add_argument("--max_pixel_blocks", type=int, default=DEFAULT_MAX_PIXEL_BLOCKS)
    parser.add_argument("--min_pixel_blocks", type=int, default=DEFAULT_MIN_PIXEL_BLOCKS)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument(
        "--fail_on_size_mismatch",
        action="store_true",
        default=True,
        help="If set, raises when PIL image size after EXIF doesn't match JSON width/height.",
    )
    parser.add_argument(
        "--no_fail_on_size_mismatch",
        dest="fail_on_size_mismatch",
        action="store_false",
        help="Disable strict size match validation (not recommended).",
    )

    args = parser.parse_args(list(argv))

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    images_dir = Path(args.images_dir).resolve() if args.images_dir else None

    if not input_dir.exists():
        raise ValueError(f"input_dir does not exist: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("output_dir must be different from input_dir to avoid overwriting")

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ResizeConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        factor=int(args.factor),
        max_pixel_blocks=int(args.max_pixel_blocks),
        min_pixel_blocks=int(args.min_pixel_blocks),
        jpeg_quality=int(args.jpeg_quality),
        fail_on_size_mismatch=bool(args.fail_on_size_mismatch),
        images_dir_override=images_dir,
        num_workers=int(args.num_workers),
    )
    return cfg


def _read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _process_jsonl(
    cfg: ResizeConfig,
    src_jsonl: Path,
    dst_jsonl: Path,
    resizer: Resizer,
) -> None:
    if not src_jsonl.exists():
        LOGGER.info("Skip absent file: %s", src_jsonl)
        return
    records = _read_jsonl(src_jsonl)
    out: List[dict] = []
    for rec in _tqdm(records, desc=src_jsonl.name, unit="rec"):
        images = rec.get("images") or []
        if cfg.fail_on_size_mismatch and images:
            resolved_first = resizer._resolve_image_paths([images[0]])  # type: ignore[attr-defined]
            actual_h, actual_w = resizer._probe_image_size(resolved_first[0])  # type: ignore[attr-defined]
            width = rec.get("width")
            height = rec.get("height")
            if isinstance(width, int) and isinstance(height, int):
                if (width, height) != (actual_w, actual_h):
                    raise ValueError(
                        f"Image size mismatch for {images[0]}: "
                        f"json={width}x{height}, exif={actual_w}x{actual_h}"
                    )
        out.append(resizer.resize_record(rec))
    _write_jsonl(dst_jsonl, out)
    LOGGER.info("Processed %d records: %s -> %s", len(out), src_jsonl, dst_jsonl)


def main(argv: Iterable[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = parse_args(argv)

    images_dir_in = (
        cfg.images_dir_override.resolve()
        if cfg.images_dir_override
        else (cfg.input_dir / "images").resolve()
    )
    output_images_dir = (cfg.output_dir / "images").resolve()
    output_images_dir.mkdir(parents=True, exist_ok=True)

    params = SmartResizeParams(
        max_pixels=cfg.max_pixels,
        image_factor=cfg.factor,
        min_pixels=cfg.min_pixels,
    )
    resizer = Resizer(
        params=params,
        jsonl_dir=None,  # set per-file when processing
        output_dir=output_images_dir,
        write_images=True,
        images_root_override=images_dir_in,
        relative_output_root=cfg.output_dir,
        exif_fn=apply_exif_orientation if apply_exif_orientation else None,
    )

    # File set to consider
    all_jsonl = cfg.input_dir / "all_samples.jsonl"
    train_jsonl = cfg.input_dir / "train.jsonl"
    val_jsonl = cfg.input_dir / "val.jsonl"
    teacher_jsonl = cfg.input_dir / "teacher_pool.jsonl"

    for src in (all_jsonl, train_jsonl, val_jsonl, teacher_jsonl):
        if not src.exists():
            LOGGER.info("Skip absent file: %s", src)
            continue
        # Update jsonl_dir per file so relative paths resolve correctly
        resizer.jsonl_dir = src.parent
        dst = cfg.output_dir / src.name
        _process_jsonl(cfg, src, dst, resizer)

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as e:  # pragma: no cover
        LOGGER.error("Resize dataset failed: %s", e)
        raise
