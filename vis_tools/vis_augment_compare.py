from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
from random import Random

from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('/data/Qwen3-VL')
from src.datasets.utils import load_jsonl
from src.datasets.augmentation.base import Compose
from src.datasets.augmentation.ops import (
    HFlip,
    Rotate,
    Scale,
    ColorJitter,
    VFlip,
    Gamma,
    HueSaturationValue,
    CLAHE,
    AutoContrast,
    Equalize,
    Solarize,
    Posterize,
    Sharpness,
    AlbumentationsColor,
    PadToMultiple,
)
from src.datasets.augment import apply_augmentations
from vis_tools.vis_helper import draw_objects, generate_colors, create_legend


@dataclass
class VisConfig:
    jsonl_path: str
    out_dir: str
    num_samples: int = 8
    variants: int = 3  # number of random augmented variants per sample
    seed: int = 2025
    # randomization knobs
    rotate_p: float = 0.7
    max_deg: float = 8.0
    scale_p: float = 0.7
    scale_lo: float = 0.95
    scale_hi: float = 1.05
    hflip_p: float = 0.5
    vflip_p: float = 0.1
    color_p: float = 0.8
    # additional color ops
    gamma_p: float = 0.0
    hsv_p: float = 0.0
    clahe_p: float = 0.0
    auto_contrast_p: float = 0.0
    equalize_p: float = 0.0
    solarize_p: float = 0.0
    posterize_p: float = 0.0
    sharpness_p: float = 0.0
    albumentations_p: float = 0.0
    albumentations_preset: str = "strong"
    # enforce size multiple like training YAML
    pad_multiple: int = 32


def _geom_to_objects(geoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    for g in geoms:
        if 'bbox_2d' in g:
            objs.append({'type': 'bbox_2d', 'points': g['bbox_2d'], 'desc': 'bbox_2d'})
        elif 'quad' in g:
            objs.append({'type': 'quad', 'points': g['quad'], 'desc': 'quad'})
        elif 'line' in g:
            objs.append({'type': 'line', 'points': g['line'], 'desc': 'line'})
    return objs


def _load_pil(img_entry: Any, *, jsonl_path: str) -> Image.Image:
    if isinstance(img_entry, dict) and 'bytes' in img_entry:
        return Image.open(io.BytesIO(img_entry['bytes'])).convert('RGB')
    if isinstance(img_entry, str):
        if not os.path.isabs(img_entry):
            root_dir = os.environ.get('ROOT_IMAGE_DIR') or os.path.dirname(os.path.abspath(jsonl_path))
            path = os.path.join(root_dir, img_entry)
        else:
            path = img_entry
        return Image.open(path).convert('RGB')
    return img_entry


def _build_random_pipeline(rng: Random, cfg: VisConfig):
    ops = []
    labels: List[str] = []
    if rng.random() < cfg.hflip_p:
        ops.append(HFlip(1.0))
        labels.append("hflip")
    if rng.random() < cfg.vflip_p:
        ops.append(VFlip(1.0))
        labels.append("vflip")
    if rng.random() < cfg.rotate_p:
        # choose signed degree centered at 0
        deg = rng.uniform(-cfg.max_deg, cfg.max_deg)
        ops.append(Rotate(abs(deg), 1.0))  # Rotate expects max_deg; prob handled by inclusion
        labels.append(f"rot={deg:.1f}")
    if rng.random() < cfg.scale_p:
        s = rng.uniform(cfg.scale_lo, cfg.scale_hi)
        ops.append(Scale(s, s, 1.0))
        labels.append(f"scale={s:.3f}")
    if rng.random() < cfg.color_p:
        # Match YAML medium defaults: 0.8-1.2 ranges
        ops.append(ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), prob=1.0))
        labels.append("cj")
    if rng.random() < cfg.gamma_p:
        ops.append(Gamma(gamma=(0.8, 1.3), prob=1.0))
        labels.append("gamma")
    if rng.random() < cfg.hsv_p:
        ops.append(HueSaturationValue(hue_delta_deg=(-15, 15), sat=(0.8, 1.3), val=(0.8, 1.3), prob=1.0))
        labels.append("hsv")
    if rng.random() < cfg.clahe_p:
        ops.append(CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), prob=1.0))
        labels.append("clahe")
    if rng.random() < cfg.auto_contrast_p:
        ops.append(AutoContrast(cutoff=0, prob=1.0))
        labels.append("autoC")
    if rng.random() < cfg.equalize_p:
        ops.append(Equalize(prob=1.0))
        labels.append("eq")
    if rng.random() < cfg.solarize_p:
        ops.append(Solarize(threshold=128, prob=1.0))
        labels.append("solar")
    if rng.random() < cfg.posterize_p:
        ops.append(Posterize(bits=4, prob=1.0))
        labels.append("post")
    if rng.random() < cfg.sharpness_p:
        ops.append(Sharpness(factor=(0.4, 2.0), prob=1.0))
        labels.append("sharp")
    if rng.random() < cfg.albumentations_p:
        ops.append(AlbumentationsColor(preset=cfg.albumentations_preset, prob=1.0))
        labels.append(f"alb-{cfg.albumentations_preset}")
    if not ops:
        # ensure at least one op to visualize
        ops.append(ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), prob=1.0))
        labels.append("cj")
    # Always enforce pad-to-multiple to mirror training
    if cfg.pad_multiple and cfg.pad_multiple > 0:
        ops.append(PadToMultiple(cfg.pad_multiple))
        labels.append(f"pad{cfg.pad_multiple}")
    return Compose(ops), "|".join(labels)


def visualize_samples(cfg: VisConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    records = load_jsonl(cfg.jsonl_path)[: cfg.num_samples]

    base_rng = Random(cfg.seed)

    for idx, rec in enumerate(records):
        images = rec.get('images') or []
        objs = rec.get('objects') or []
        per_obj_geoms: List[Dict[str, Any]] = []
        for o in objs:
            g: Dict[str, Any] = {}
            if o.get('bbox_2d') is not None:
                g['bbox_2d'] = o['bbox_2d']
            if o.get('quad') is not None:
                g['quad'] = o['quad']
            if o.get('line') is not None:
                g['line'] = o['line']
            if g:
                per_obj_geoms.append(g)

        # Resolve images to PIL
        pil_images: List[Image.Image] = []
        for it in images:
            pil_images.append(_load_pil(it, jsonl_path=cfg.jsonl_path))

        # Load original image (first)
        im0 = pil_images[0]
        # Prepare objects for drawing
        objs0 = _geom_to_objects(per_obj_geoms)

        # Build N random pipelines and apply
        variants_imgs: List[Image.Image] = []
        variants_objs: List[List[Dict[str, Any]]] = []
        variant_titles: List[str] = []
        for j in range(int(cfg.variants)):
            # per-variant rng derived from base
            rng = Random(base_rng.random())
            pipe, title = _build_random_pipeline(rng, cfg)
            imgs_bytes, geoms_new = apply_augmentations(pil_images, per_obj_geoms, pipe, rng=rng)
            variants_imgs.append(_load_pil(imgs_bytes[0], jsonl_path=cfg.jsonl_path))
            variants_objs.append(_geom_to_objects(geoms_new))
            variant_titles.append(title)

        # Matplotlib side-by-side: original + variants
        cols = 1 + len(variants_imgs)
        fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))
        if cols == 1:
            axes = [axes]
        labels_all = [o['desc'] for o in objs0]
        for arr in variants_objs:
            labels_all.extend([o['desc'] for o in arr])
        color_map = generate_colors(labels_all)

        draw_objects(axes[0], im0, objs0, color_map, scaled=True)
        axes[0].set_title('Original (GT)')
        for c in range(1, cols):
            draw_objects(axes[c], variants_imgs[c - 1], variants_objs[c - 1], color_map, scaled=True)
            axes[c].set_title(f'Aug {c}: {variant_titles[c - 1]}')
        create_legend(fig, color_map, {l: [labels_all.count(l), labels_all.count(l)] for l in set(labels_all)})
        out_path = os.path.join(cfg.out_dir, f'vis_{idx:05d}.jpg')
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[INFO] Saved {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Defaults mirror configs/stage_3_vision_lora.yaml
    parser.add_argument('--jsonl', default='data/bbu_full_768/train.jsonl')
    parser.add_argument('--out', default='vis_out/augment_compare_default')
    parser.add_argument('--num', type=int, default=8)
    parser.add_argument('--variants', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--rotate_p', type=float, default=0.3)
    parser.add_argument('--max_deg', type=float, default=8.0)
    parser.add_argument('--scale_p', type=float, default=0.35)
    parser.add_argument('--scale_lo', type=float, default=0.95)
    parser.add_argument('--scale_hi', type=float, default=1.05)
    parser.add_argument('--hflip_p', type=float, default=0.5)
    parser.add_argument('--vflip_p', type=float, default=0.1)
    parser.add_argument('--color_p', type=float, default=0.5)
    parser.add_argument('--gamma_p', type=float, default=0.25)
    parser.add_argument('--hsv_p', type=float, default=0.3)
    parser.add_argument('--clahe_p', type=float, default=0.15)
    parser.add_argument('--auto_contrast_p', type=float, default=0.1)
    parser.add_argument('--equalize_p', type=float, default=0.05)
    parser.add_argument('--solarize_p', type=float, default=0.0)
    parser.add_argument('--posterize_p', type=float, default=0.0)
    parser.add_argument('--sharpness_p', type=float, default=0.25)
    parser.add_argument('--albumentations_p', type=float, default=0.0)
    parser.add_argument('--albumentations_preset', type=str, default='strong', choices=['strong', 'extreme'])
    parser.add_argument('--pad_multiple', type=int, default=32)
    args = parser.parse_args()
    cfg = VisConfig(
        jsonl_path=args.jsonl,
        out_dir=args.out,
        num_samples=args.num,
        variants=args.variants,
        seed=args.seed,
        rotate_p=args.rotate_p,
        max_deg=args.max_deg,
        scale_p=args.scale_p,
        scale_lo=args.scale_lo,
        scale_hi=args.scale_hi,
        hflip_p=args.hflip_p,
        vflip_p=args.vflip_p,
        color_p=args.color_p,
        gamma_p=args.gamma_p,
        hsv_p=args.hsv_p,
        clahe_p=args.clahe_p,
        auto_contrast_p=args.auto_contrast_p,
        equalize_p=args.equalize_p,
        solarize_p=args.solarize_p,
        posterize_p=args.posterize_p,
        sharpness_p=args.sharpness_p,
        albumentations_p=args.albumentations_p,
        albumentations_preset=args.albumentations_preset,
        pad_multiple=args.pad_multiple,
    )
    visualize_samples(cfg)


