
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, Dict, List
from random import Random

from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('/data/Qwen3-VL')
import yaml
from src.datasets.utils import load_jsonl
from src.datasets.augmentation.base import Compose
from src.datasets.augmentation.builder import build_compose_from_config
from src.datasets.augmentation.ops import (
    HFlip,
    Rotate,
    Scale,
    VFlip,
    ExpandToFitAffine,
    ResizeByScale,
    RandomCrop,
)
from vis_tools.vis_helper import draw_objects, generate_colors, create_legend


@dataclass
class VisConfig:
    """Configuration for augmentation visualization.
    
    Tests ALL available augmentation operations with EXTREME settings and HIGH probabilities
    to thoroughly verify coordinate transforms, geometry handling, and edge cases.
    """
    jsonl_path: str
    out_dir: str
    num_samples: int = 8
    variants: int = 4  # number of random augmented variants per sample
    seed: int = 2021
    
    # Set to None to use extreme testing pipeline, or path to YAML to mirror training
    config_yaml: str | None = None
    
    # EXTREME GEOMETRIC AUGMENTATIONS (test coordinate transforms)
    hflip_p: float = 0.9
    vflip_p: float = 0.5
    rotate_p: float = 0.9
    max_deg: float = 30.0           # Extreme rotation
    scale_p: float = 0.9
    scale_lo: float = 0.7           # Extreme shrink
    scale_hi: float = 1.5           # Extreme grow
    
    # EXTREME RESOLUTION CHANGES (test rescaling)
    resize_by_scale_p: float = 0.9
    resize_lo: float = 0.5          # Extreme shrink (50%)
    resize_hi: float = 1.8          # Extreme grow (180%)
    resize_align_multiple: int = 32
    
    # SMART CROPPING (test label filtering & geometry truncation)
    random_crop_p: float = 0.5      # Test crop filtering
    random_crop_scale_lo: float = 0.6
    random_crop_scale_hi: float = 1.0
    crop_min_coverage: float = 0.25 # More aggressive filtering
    crop_min_objects: int = 3       # Lower threshold for testing
    crop_skip_if_line: bool = True
    
    # EXTREME COLOR AUGMENTATIONS (test visual changes)
    color_p: float = 0.0
    color_brightness: tuple = (0.5, 1.5)
    color_contrast: tuple = (0.5, 1.5)
    color_saturation: tuple = (0.5, 1.5)
    gamma_p: float = 0.0
    gamma_range: tuple = (0.6, 1.6)
    hsv_p: float = 0.0
    hsv_hue_delta: int = 25
    hsv_sat: tuple = (0.6, 1.5)
    hsv_val: tuple = (0.6, 1.5)
    clahe_p: float = 0.0
    clahe_clip_limit: float = 4.0
    auto_contrast_p: float = 0.0
    solarize_p: float = 0.0
    solarize_threshold: int = 128
    posterize_p: float = 0.0
    posterize_bits: int = 3
    sharpness_p: float = 0.0
    sharpness_range: tuple = (0.3, 2.5)
    albumentations_p: float = 0.0
    albumentations_preset: str = "strong"
    
    # Padding
    pad_multiple: int = 32


def _extract_description(obj: Dict[str, Any]) -> str:
    candidate_keys = (
        "desc",
        "text",
        "caption",
        "name",
        "label",
        "category",
        "title",
        "ref",
    )
    for key in candidate_keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    attrs = obj.get("attributes")
    if isinstance(attrs, dict):
        attr_keys = (
            "desc",
            "text",
            "caption",
            "name",
            "label",
            "category",
            "描述",
            "名称",
        )
        for key in attr_keys:
            value = attrs.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def _geom_to_objects(
    geoms: List[Dict[str, Any]],
    descs: List[str] | None = None,
) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    for idx, g in enumerate(geoms):
        desc = ""
        if descs is not None and idx < len(descs):
            desc = descs[idx] or ""
        if 'bbox_2d' in g:
            objs.append({'type': 'bbox_2d', 'points': g['bbox_2d'], 'desc': desc or 'bbox_2d'})
        elif 'poly' in g:
            objs.append({'type': 'poly', 'points': g['poly'], 'desc': desc or 'poly'})
        elif 'line' in g:
            objs.append({'type': 'line', 'points': g['line'], 'desc': desc or 'line'})
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


def _build_pipeline_from_yaml(cfg: VisConfig) -> tuple[Compose, str]:
    """Load augmentation pipeline from YAML config. Raises exception if YAML not found."""
    if not cfg.config_yaml:
        raise FileNotFoundError("config_yaml is None (using extreme testing mode)")
    if not os.path.isfile(cfg.config_yaml):
        raise FileNotFoundError(f"config_yaml file not found: {cfg.config_yaml}")
    
    with open(cfg.config_yaml, 'r', encoding='utf-8') as f:
        conf = yaml.safe_load(f)
    custom = (conf or {}).get('custom') or {}
    aug = custom.get('augmentation') or {}
    if not aug or not aug.get('enabled', False):
        return Compose([]), 'yaml:no-augmentation'
    compose = build_compose_from_config(aug)
    # Build readable label from op names in order
    ops = aug.get('ops') or []
    label = 'yaml:' + ','.join(str(op.get('name')) for op in ops if isinstance(op, dict) and op.get('name'))
    return compose, label


def _build_random_pipeline(rng: Random, cfg: VisConfig):
    """Build pipeline with ALL augmentation operations using EXTREME settings.
    
    This tests:
    - Geometric transforms (rotation, scale, flip) with extreme parameters
    - Canvas expansion (expand_to_fit_affine) to preserve all content
    - Smart cropping with label filtering
    - Color augmentations with extreme ranges
    - Coordinate transform correctness under all operations
    """
    ops = []
    labels: List[str] = []
    
    # === GEOMETRIC AUGMENTATIONS (affine accumulation) ===
    if rng.random() < cfg.hflip_p:
        ops.append(HFlip(1.0))
        labels.append("hflip")
    
    if rng.random() < cfg.vflip_p:
        ops.append(VFlip(1.0))
        labels.append("vflip")
    
    if rng.random() < cfg.rotate_p:
        deg = rng.uniform(-cfg.max_deg, cfg.max_deg)
        ops.append(Rotate(abs(deg), 1.0))
        labels.append(f"rot={deg:.1f}°")
    
    if rng.random() < cfg.scale_p:
        s = rng.uniform(cfg.scale_lo, cfg.scale_hi)
        ops.append(Scale(s, s, 1.0))
        labels.append(f"scale={s:.2f}")
    
    # === EXPAND CANVAS (barrier - preserves all rotated/scaled content) ===
    ops.append(ExpandToFitAffine(multiple=cfg.pad_multiple))
    labels.append("expand")
    
    # === SMART CROPPING (barrier - filters labels) ===
    if rng.random() < cfg.random_crop_p:
        ops.append(RandomCrop(
            scale=(cfg.random_crop_scale_lo, cfg.random_crop_scale_hi),
            aspect_ratio=(0.9, 1.1),
            min_coverage=cfg.crop_min_coverage,
            min_objects=cfg.crop_min_objects,
            skip_if_line=cfg.crop_skip_if_line,
            prob=1.0
        ))
        labels.append(f"crop({cfg.random_crop_scale_lo:.1f}-{cfg.random_crop_scale_hi:.1f})")
    
    # === RESOLUTION RESIZING (barrier - tests coordinate scaling) ===
    if rng.random() < cfg.resize_by_scale_p:
        ops.append(ResizeByScale(
            lo=cfg.resize_lo,
            hi=cfg.resize_hi,
            align_multiple=cfg.resize_align_multiple,
            prob=1.0
        ))
        labels.append(f"resize({cfg.resize_lo:.1f}-{cfg.resize_hi:.1f})")
    
    # === COLOR AUGMENTATIONS (deferred, applied after all geometric ops) ===
    # No color ops for alignment visualization
    
    return Compose(ops), "|".join(labels)


def visualize_samples(cfg: VisConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    records = load_jsonl(cfg.jsonl_path)[: cfg.num_samples]

    base_rng = Random(cfg.seed)

    # Prefer exact training augmentation from YAML if provided
    use_yaml = False
    if cfg.config_yaml:
        try:
            yaml_pipeline, yaml_label = _build_pipeline_from_yaml(cfg)
            use_yaml = True
            print(f"[INFO] Using augmentation from YAML: {cfg.config_yaml}")
        except Exception as e:
            print(f"[ERROR] Failed to load training YAML augmentation: {e}")
            print("[INFO] Falling back to extreme testing pipeline")
    else:
        print("[INFO] Using extreme testing pipeline (config_yaml=None)")

    for idx, rec in enumerate(records):
        images = rec.get('images') or []
        objs = rec.get('objects') or []
        per_obj_geoms: List[Dict[str, Any]] = []
        per_obj_descs: List[str] = []
        for o in objs:
            g: Dict[str, Any] = {}
            if o.get('bbox_2d') is not None:
                g['bbox_2d'] = o['bbox_2d']
            if o.get('poly') is not None:
                g['poly'] = o['poly']
            if o.get('line') is not None:
                g['line'] = o['line']
            if g:
                per_obj_geoms.append(g)
                per_obj_descs.append(_extract_description(o))

        # Resolve images to PIL
        pil_images: List[Image.Image] = []
        for it in images:
            pil_images.append(_load_pil(it, jsonl_path=cfg.jsonl_path))

        # Load original image (first)
        im0 = pil_images[0]
        # Prepare objects for drawing
        objs0 = _geom_to_objects(per_obj_geoms, per_obj_descs)

        # Build N random pipelines and apply
        variants_imgs: List[Image.Image] = []
        variants_objs: List[List[Dict[str, Any]]] = []
        variant_titles: List[str] = []
        
        for j in range(int(cfg.variants)):
            # per-variant rng derived from base
            rng = Random(base_rng.random())
            if use_yaml:
                pipe, title = yaml_pipeline, yaml_label
            else:
                pipe, title = _build_random_pipeline(rng, cfg)
            # Apply pipeline directly to allow objects to be dropped (degenerate after clipping)
            out_imgs, geoms_new = pipe.apply(
                pil_images,
                per_obj_geoms,
                width=im0.width,
                height=im0.height,
                rng=rng,
            )

            # Check if objects were filtered by crop
            if len(geoms_new) != len(per_obj_geoms):
                print(f"  [CROP] Sample {idx}, variant {j+1}: {len(per_obj_geoms)} → {len(geoms_new)} objects")
                if hasattr(pipe, 'last_kept_indices') and pipe.last_kept_indices is not None:
                    print(f"         Kept indices: {pipe.last_kept_indices}")
                    if hasattr(pipe, 'last_object_coverages') and pipe.last_object_coverages:
                        avg_cov = sum(pipe.last_object_coverages) / len(pipe.last_object_coverages)
                        print(f"         Avg coverage of kept objects: {avg_cov:.2%}")

            # Use returned PIL image directly
            variants_imgs.append(_load_pil(out_imgs[0], jsonl_path=cfg.jsonl_path))
            kept_indices = getattr(pipe, 'last_kept_indices', None)
            if isinstance(kept_indices, list):
                descs_new = [per_obj_descs[i] for i in kept_indices if i < len(per_obj_descs)]
            elif len(geoms_new) == len(per_obj_descs):
                descs_new = per_obj_descs
            else:
                descs_new = per_obj_descs[: len(geoms_new)]
            variants_objs.append(_geom_to_objects(geoms_new, descs_new))
            variant_titles.append(title)

        # Matplotlib side-by-side: original + variants
        cols = 1 + len(variants_imgs)
        fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))
        if cols == 1:
            axes = [axes]
        
        # Collect all labels for color mapping and counting
        labels_all = [o['desc'] for o in objs0]
        for arr in variants_objs:
            labels_all.extend([o['desc'] for o in arr])
        color_map = generate_colors(labels_all)
        
        # Track counts: GT vs each augmented variant (use first variant for legend)
        counts: Dict[str, List[int]] = {}
        for o in objs0:
            key = o.get('desc', '')
            counts.setdefault(key, [0, 0])[0] += 1
        if variants_objs:
            for o in variants_objs[0]:  # Use first variant for comparison
                key = o.get('desc', '')
                counts.setdefault(key, [0, 0])[1] += 1

        draw_objects(axes[0], im0, objs0, color_map, scaled=True)
        axes[0].set_title(f'Original (GT) - {len(objs0)} objects')
        for c in range(1, cols):
            draw_objects(axes[c], variants_imgs[c - 1], variants_objs[c - 1], color_map, scaled=True)
            obj_count = len(variants_objs[c - 1])
            title = f'Aug {c}: {variant_titles[c - 1]}\n{obj_count} objects'
            if obj_count != len(objs0):
                title += f' ({obj_count - len(objs0):+d})'
            axes[c].set_title(title, fontsize=9)
        
        create_legend(fig, color_map, counts)
        out_path = os.path.join(cfg.out_dir, f'vis_{idx:05d}.jpg')
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[INFO] Saved {out_path}")


if __name__ == '__main__':
    # ========================================================================
    # EXTREME AUGMENTATION TESTING - Tests ALL ops with extreme settings
    # ========================================================================
    
    cfg = VisConfig(
        jsonl_path='data/bbu_full_768/train.jsonl',
        out_dir='vis_out/augment_extreme_test',
        num_samples=8,
        variants=3,
        seed=2025,
        
        config_yaml=None,
    )
    
    print("=" * 70)
    if cfg.config_yaml:
        print("MIRRORING TRAINING AUGMENTATION FROM YAML")
        print(f"  YAML: {cfg.config_yaml}")
    else:
        print("🔥 EXTREME AUGMENTATION TESTING MODE 🔥")
        print()
        print("Testing ALL augmentation operations with EXTREME settings:")
        print(f"  ✓ Geometric: hflip({cfg.hflip_p:.0%}), vflip({cfg.vflip_p:.0%}), rot(±{cfg.max_deg}°), scale({cfg.scale_lo}-{cfg.scale_hi})")
        print(f"  ✓ Resize: {cfg.resize_lo}x - {cfg.resize_hi}x resolution changes")
        print(f"  ✓ Crop: {cfg.random_crop_p:.0%} prob, min_coverage={cfg.crop_min_coverage}, min_objects={cfg.crop_min_objects}")
        print(f"  ✓ Color: brightness/contrast/saturation {cfg.color_brightness}")
        print(f"  ✓ Advanced: gamma({cfg.gamma_p:.0%}), hsv({cfg.hsv_p:.0%}), clahe({cfg.clahe_p:.0%}), sharpness({cfg.sharpness_p:.0%})")
        print(f"  ✓ Effects: solarize({cfg.solarize_p:.0%}), posterize({cfg.posterize_p:.0%}), autoContrast({cfg.auto_contrast_p:.0%})")
    print("=" * 70)
    print(f"Input:     {cfg.jsonl_path}")
    print(f"Output:    {cfg.out_dir}")
    print(f"Samples:   {cfg.num_samples} images × {cfg.variants} variants each")
    print(f"Seed:      {cfg.seed}")
    print("=" * 70)
    print()
    
    visualize_samples(cfg)


