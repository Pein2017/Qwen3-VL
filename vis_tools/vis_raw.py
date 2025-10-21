from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
from PIL import Image

# Make repo root importable when executing this file directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vis_tools.vis_helper import draw_objects, generate_colors  # noqa: E402

try:  # pragma: no cover
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    def _tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else []


def parse_args(argv: Iterable[str]):
    parser = argparse.ArgumentParser(
        description="Visualize raw annotations over images from a JSONL dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jsonl", type=str, required=True, help="Path to JSONL (e.g., data/bbu_full_768/all_samples.jsonl)")
    parser.add_argument("--out_dir", type=str, default="vis_out/raw", help="Directory to save visualizations")
    parser.add_argument("--limit", type=int, default=0, help="Max number of samples to visualize (0 means all)")
    parser.add_argument("--dpi", type=int, default=120, help="Matplotlib savefig DPI")
    parser.add_argument(
        "--color_by",
        type=str,
        default="type",
        choices=["type", "desc"],
        help="Color by object type or full desc prefix",
    )
    return parser.parse_args(list(argv))


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def to_vis_objects(sample_objects: List[Dict[str, Any]], color_by: str) -> List[Dict[str, Any]]:
    vis_objects: List[Dict[str, Any]] = []
    for obj in sample_objects or []:
        if "bbox_2d" in obj:
            gtype = "bbox_2d"
            pts = obj["bbox_2d"]
        elif "quad" in obj:
            gtype = "quad"
            pts = obj["quad"]
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


def visualize(jsonl_path: Path, out_dir: Path, limit: int, dpi: int, color_by: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = jsonl_path.parent

    count = 0
    for idx, rec in enumerate(_tqdm(read_jsonl(jsonl_path), desc=jsonl_path.name, unit="rec")):
        if limit and count >= limit:
            break

        images = rec.get("images")
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(f"Record must have exactly one image path, got: {images}")
        image_rel = images[0]
        image_path = (base_dir / image_rel).resolve()

        img = Image.open(image_path).convert("RGB")
        obj_list = to_vis_objects(rec.get("objects", []), color_by=color_by)
        labels = [o["desc"] for o in obj_list]
        color_map = generate_colors(labels)

        fig, ax = plt.subplots(1, 1)
        try:
            draw_objects(ax, img, obj_list, color_map, scaled=True)
            ax.set_title(image_rel)
            fig.tight_layout()
            out_path = out_dir / f"vis_{idx:05d}.jpg"
            fig.savefig(out_path, dpi=dpi)
        finally:
            plt.close(fig)

        count += 1


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    jsonl_path = Path(args.jsonl).resolve()
    if not jsonl_path.exists():
        raise ValueError(f"JSONL not found: {jsonl_path}")

    out_dir = Path(args.out_dir).resolve()
    visualize(jsonl_path=jsonl_path, out_dir=out_dir, limit=int(args.limit), dpi=int(args.dpi), color_by=str(args.color_by))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


