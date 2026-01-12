from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from collections.abc import Mapping
from typing import Any, cast

from PIL import Image

from src.config import ConfigLoader
from src.datasets.augmentation.builder import build_compose_from_config
from src.datasets.augmentation.curriculum import AugmentationCurriculumScheduler
from src.datasets.augmentation.ops import RoiCrop


def _apply_curriculum_overrides(
    *, pipeline: object, overrides: Mapping[str, Mapping[str, object]]
) -> None:
    name_map = getattr(pipeline, "_augmentation_name_map", {}) or {}
    if not name_map:
        raise ValueError(
            "augmentation pipeline missing _augmentation_name_map metadata"
        )
    for op_name, param_overrides in overrides.items():
        targets = name_map.get(op_name, [])
        if not targets:
            raise ValueError(
                f"curriculum override references op '{op_name}' with no instances"
            )
        for op in targets:
            for param_name, raw_value in param_overrides.items():
                current = getattr(op, param_name, None)
                if isinstance(current, tuple):
                    if not isinstance(raw_value, (list, tuple)):
                        raise TypeError("tuple override must be list/tuple")
                    setattr(
                        op,
                        param_name,
                        tuple(
                            type(cast(tuple[Any, ...], current)[i])(raw_value[i])
                            for i in range(len(raw_value))
                        ),
                    )
                elif isinstance(current, list):
                    if not isinstance(raw_value, (list, tuple)):
                        raise TypeError("list override must be list/tuple")
                    setattr(
                        op,
                        param_name,
                        [type(current[i])(raw_value[i]) for i in range(len(raw_value))],
                    )
                elif isinstance(current, bool):
                    setattr(op, param_name, bool(raw_value))
                elif isinstance(current, int):
                    setattr(op, param_name, int(round(float(cast(Any, raw_value)))))
                elif isinstance(current, float):
                    setattr(op, param_name, float(cast(Any, raw_value)))
                else:
                    setattr(op, param_name, raw_value)


def analyze_dataset(
    *,
    jsonl_path: str,
    config_yaml: str,
    num_samples: int = 20,
    seeds_per_sample: int = 20,
    curriculum_percent: float | None = None,
    total_steps: int = 100,
) -> None:
    print(f"Analyzing {jsonl_path} with {config_yaml}...")

    conf = cast(Mapping[str, object], ConfigLoader.load_yaml_with_extends(config_yaml))
    custom = cast(Mapping[str, object], conf.get("custom") or {})
    aug_cfg = cast(Mapping[str, object], custom.get("augmentation") or {})
    curriculum_cfg = cast(
        Mapping[str, object] | None, custom.get("augmentation_curriculum")
    )

    if not aug_cfg.get("enabled", False):
        raise ValueError("augmentation disabled in config")

    pipeline = build_compose_from_config(aug_cfg)
    bypass_prob = float(aug_cfg.get("bypass_prob", 0.0) or 0.0)

    if curriculum_percent is not None:
        if curriculum_cfg is None:
            raise ValueError(
                "--curriculum-percent provided but config has no custom.augmentation_curriculum"
            )
        meta = getattr(pipeline, "_augmentation_meta", None) or []
        scheduler = AugmentationCurriculumScheduler.from_config(
            base_bypass=bypass_prob,
            op_meta=meta,
            curriculum_raw=curriculum_cfg,
        )
        scheduler.set_total_steps(total_steps)
        step = int(round((curriculum_percent / 100.0) * total_steps))
        state = scheduler.get_state(step)
        bypass_prob = float(state["bypass_prob"])
        _apply_curriculum_overrides(pipeline=pipeline, overrides=state["ops"])

    records: list[dict[str, object]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            records.append(cast(dict[str, object], json.loads(line)))

    stats: dict[str, object] = {
        "bypass_prob": bypass_prob,
        "curriculum_percent": curriculum_percent,
        "total_steps": total_steps,
        "roi_crop_triggered": 0,
        "roi_crop_not_triggered": 0,
        "roi_crop_skip_reasons": Counter(),
        "total_objects_before": 0,
        "total_objects_after": 0,
    }

    root_dir = os.path.dirname(os.path.abspath(jsonl_path))
    os.environ.setdefault("ROOT_IMAGE_DIR", root_dir)

    roi_ops = [op for op in getattr(pipeline, "ops", []) if isinstance(op, RoiCrop)]
    if not roi_ops:
        raise ValueError(
            "pipeline does not include RoiCrop; cannot audit roi_crop stats"
        )
    if len(roi_ops) > 1:
        raise ValueError(f"expected exactly 1 RoiCrop op, found {len(roi_ops)}")
    roi = roi_ops[0]

    for rec_idx, rec in enumerate(records):
        images = cast(list[object], rec.get("images") or [])
        objects = cast(list[object], rec.get("objects") or [])

        pil_images: list[Image.Image] = []
        for img_path_raw in images:
            if not isinstance(img_path_raw, str):
                continue
            img_path = img_path_raw
            if not os.path.isabs(img_path):
                img_path = os.path.join(root_dir, img_path)
            try:
                pil_images.append(Image.open(img_path).convert("RGB"))
            except Exception:
                continue

        if not pil_images:
            continue
        w, h = pil_images[0].size

        geoms: list[dict[str, object]] = []
        for obj_raw in objects:
            if not isinstance(obj_raw, dict):
                continue
            obj = cast(dict[str, object], obj_raw)
            g: dict[str, object] = {}
            if "bbox_2d" in obj:
                g["bbox_2d"] = obj["bbox_2d"]
            elif "poly" in obj:
                g["poly"] = obj["poly"]
            elif "line" in obj:
                g["line"] = obj["line"]
            desc = obj.get("desc")
            if isinstance(desc, str):
                g["desc"] = desc
            if g:
                geoms.append(g)

        stats["total_objects_before"] = (
            int(stats["total_objects_before"]) + len(geoms) * seeds_per_sample
        )

        for i in range(seeds_per_sample):
            rng = random.Random(2026 + rec_idx * 1000 + i)

            if rng.random() < bypass_prob:
                stats["total_objects_after"] = int(stats["total_objects_after"]) + len(
                    geoms
                )
                continue

            _, new_geoms = pipeline.apply(pil_images, geoms, width=w, height=h, rng=rng)
            stats["total_objects_after"] = int(stats["total_objects_after"]) + len(
                new_geoms
            )

            if roi.last_kept_indices is not None:
                stats["roi_crop_triggered"] = int(stats["roi_crop_triggered"]) + 1
            elif roi.last_crop_skip_reason is not None:
                cast(Counter[str], stats["roi_crop_skip_reasons"])[
                    roi.last_crop_skip_reason
                ] += 1
            else:
                stats["roi_crop_not_triggered"] = (
                    int(stats["roi_crop_not_triggered"]) + 1
                )

    # Make counters JSON-serializable
    stats["roi_crop_skip_reasons"] = dict(
        cast(Counter[str], stats["roi_crop_skip_reasons"])
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seeds-per-sample", type=int, default=20)
    parser.add_argument("--curriculum-percent", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=100)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    analyze_dataset(
        jsonl_path=args.jsonl,
        config_yaml=args.config,
        num_samples=args.num_samples,
        seeds_per_sample=args.seeds_per_sample,
        curriculum_percent=args.curriculum_percent,
        total_steps=args.total_steps,
    )
