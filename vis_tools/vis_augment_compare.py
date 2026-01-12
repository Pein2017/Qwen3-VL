from __future__ import annotations

import copy
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Dict, List, cast

import matplotlib.pyplot as plt
from PIL import Image

from src.config import ConfigLoader
from src.datasets.contracts import DatasetObject
from src.datasets.augmentation.base import Compose
from src.datasets.augmentation.builder import build_compose_from_config
from src.datasets.augmentation.curriculum import AugmentationCurriculumScheduler
from src.datasets.augmentation.ops import (
    ExpandToFitAffine,
    HFlip,
    ResizeByScale,
    RoiCrop,
    Rotate,
    Scale,
    VFlip,
)
from src.datasets.preprocessors.augmentation import AugmentationPreprocessor
from vis_tools.vis_helper import create_legend, draw_objects, generate_colors

# Matplotlib types are not fully available in the runtime environment; treat as Any for tooling.
plt = cast(Any, plt)


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

    # Curriculum visualization
    curriculum_marks: List[float] | None = (
        None  # percent checkpoints; None => auto from config + [0,30,100]
    )
    curriculum_total_steps: int = 1000  # steps used to resolve percent curriculum
    respect_bypass_prob: bool = True  # mirror training bypass chance
    save_state_summary: bool = True

    # Optional: focus on a single op (e.g., roi_crop) ignoring curriculum
    focus_op_name: str | None = None
    focus_force_prob: float | None = None  # if set, overrides that op's prob in vis

    # Set to None to use extreme testing pipeline, or path to YAML to mirror training
    config_yaml: str | None = None

    # EXTREME GEOMETRIC AUGMENTATIONS (test coordinate transforms)
    hflip_p: float = 0.9
    vflip_p: float = 0.5
    rotate_p: float = 0.9
    max_deg: float = 30.0  # Extreme rotation
    scale_p: float = 0.9
    scale_lo: float = 0.7  # Extreme shrink
    scale_hi: float = 1.5  # Extreme grow

    # EXTREME RESOLUTION CHANGES (test rescaling)
    resize_by_scale_p: float = 0.9
    resize_lo: float = 0.5  # Extreme shrink (50%)
    resize_hi: float = 1.8  # Extreme grow (180%)
    resize_align_multiple: int = 32

    # ROI CROPPING (tests filtering & geometry truncation)
    roi_crop_p: float = 0.5
    roi_crop_scale_lo: float = 1.2
    roi_crop_scale_hi: float = 2.0
    roi_crop_min_crop_size: int = 320
    roi_crop_anchor_classes: List[str] = field(
        default_factory=lambda: ["BBU设备", "RRU设备", "机柜"]
    )
    roi_crop_min_coverage: float = 0.25
    roi_crop_completeness_threshold: float = 0.95

    # EXTREME COLOR AUGMENTATIONS (test visual changes)
    color_p: float = 0.0
    color_brightness: tuple[float, float] = (0.5, 1.5)
    color_contrast: tuple[float, float] = (0.5, 1.5)
    color_saturation: tuple[float, float] = (0.5, 1.5)
    gamma_p: float = 0.0
    gamma_range: tuple[float, float] = (0.6, 1.6)
    hsv_p: float = 0.0
    hsv_hue_delta: int = 25
    hsv_sat: tuple[float, float] = (0.6, 1.5)
    hsv_val: tuple[float, float] = (0.6, 1.5)
    clahe_p: float = 0.0
    clahe_clip_limit: float = 4.0
    auto_contrast_p: float = 0.0
    solarize_p: float = 0.0
    solarize_threshold: int = 128
    posterize_p: float = 0.0
    posterize_bits: int = 3
    sharpness_p: float = 0.0
    sharpness_range: tuple[float, float] = (0.3, 2.5)
    albumentations_p: float = 0.0
    albumentations_preset: str = "strong"

    # Padding
    pad_multiple: int = 32


@dataclass
class PipelineSpec:
    compose: Compose
    label: str
    bypass_prob: float
    curriculum: AugmentationCurriculumScheduler | None = None
    curriculum_phase_percents: List[float] = field(default_factory=list)
    augmentation_cfg: Dict[str, Any] | None = None


@dataclass
class StageSpec:
    tag: str
    percent: float
    pipeline: Compose
    bypass_prob: float
    state: Dict[str, Any]
    label: str


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
        attrs_typed = cast(Dict[str, Any], attrs)
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
            value = attrs_typed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def _geom_to_objects(
    geoms: List[DatasetObject],
    descs: List[str] | None = None,
) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    for idx, g in enumerate(geoms):
        desc = ""
        if descs is not None and idx < len(descs):
            desc = descs[idx] or ""
        if "bbox_2d" in g:
            objs.append(
                {"type": "bbox_2d", "points": g["bbox_2d"], "desc": desc or "bbox_2d"}
            )
        elif "poly" in g:
            objs.append({"type": "poly", "points": g["poly"], "desc": desc or "poly"})
        elif "line" in g:
            objs.append({"type": "line", "points": g["line"], "desc": desc or "line"})
    return objs


def _load_pil(img_entry: Any, *, jsonl_path: str) -> Image.Image:
    if isinstance(img_entry, dict) and "bytes" in img_entry:
        raw = img_entry["bytes"]
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError(f"Unsupported image bytes type: {type(raw)}")
        return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    if isinstance(img_entry, str):
        if not os.path.isabs(img_entry):
            root_dir = os.environ.get("ROOT_IMAGE_DIR") or os.path.dirname(
                os.path.abspath(jsonl_path)
            )
            path = os.path.join(root_dir, img_entry)
        else:
            path = img_entry
        return Image.open(path).convert("RGB")
    if isinstance(img_entry, Image.Image):
        return img_entry
    raise TypeError(f"Unsupported image entry type: {type(img_entry)}")


def _build_pipeline_from_yaml(cfg: VisConfig) -> PipelineSpec:
    """Load augmentation pipeline + curriculum from YAML config."""
    if not cfg.config_yaml:
        raise FileNotFoundError("config_yaml is None (using extreme testing mode)")
    if not os.path.isfile(cfg.config_yaml):
        raise FileNotFoundError(f"config_yaml file not found: {cfg.config_yaml}")

    conf_raw = ConfigLoader.load_yaml_with_extends(cfg.config_yaml)
    conf: Dict[str, Any] = (
        cast(Dict[str, Any], conf_raw) if isinstance(conf_raw, dict) else {}
    )

    custom_raw = conf.get("custom")
    custom: Dict[str, Any] = (
        cast(Dict[str, Any], custom_raw) if isinstance(custom_raw, dict) else {}
    )

    aug_cfg_raw = custom.get("augmentation")
    aug_cfg: Dict[str, Any] = (
        cast(Dict[str, Any], aug_cfg_raw) if isinstance(aug_cfg_raw, dict) else {}
    )

    curriculum_cfg_raw = aug_cfg.get("curriculum") or custom.get(
        "augmentation_curriculum"
    )
    curriculum_cfg: Dict[str, Any] | None = (
        cast(Dict[str, Any], curriculum_cfg_raw)
        if isinstance(curriculum_cfg_raw, dict)
        else None
    )

    # Optional focus mode: keep only a single op, disable curriculum, and optionally override prob
    focus_name = cfg.focus_op_name
    if focus_name:
        ops_raw = aug_cfg.get("ops")
        ops_list: List[Dict[str, Any]] = (
            cast(List[Dict[str, Any]], ops_raw) if isinstance(ops_raw, list) else []
        )
        filtered_ops: List[Dict[str, Any]] = []
        for op in ops_list:
            if not isinstance(op, dict):
                continue
            if op.get("name") != focus_name:
                continue
            params = dict(op.get("params") or {})
            if cfg.focus_force_prob is not None:
                try:
                    params["prob"] = float(cfg.focus_force_prob)
                except Exception:
                    params["prob"] = cfg.focus_force_prob
            filtered_ops.append({"name": focus_name, "params": params})
        if not filtered_ops:
            available = [
                op.get("name") for op in ops_list if op.get("name") is not None
            ]
            raise ValueError(
                f"focus_op_name={focus_name!r} not found in augmentation.ops; "
                f"available ops: {available}"
            )
        aug_cfg = dict(aug_cfg)
        aug_cfg["ops"] = filtered_ops
        # For isolated op visualization we disable curriculum and bypass
        curriculum_cfg = None
        aug_cfg["bypass_prob"] = 0.0

    if not aug_cfg or not aug_cfg.get("enabled", False):
        return PipelineSpec(
            compose=Compose([]),
            label="yaml:no-augmentation",
            bypass_prob=float(
                aug_cfg.get("bypass_prob", custom.get("bypass_prob", 0.0))
            ),
            curriculum=None,
            curriculum_phase_percents=[],
            augmentation_cfg=aug_cfg,
        )

    compose = build_compose_from_config(aug_cfg)
    base_bypass = float(aug_cfg.get("bypass_prob", custom.get("bypass_prob", 0.0)))

    # Build readable label from op names in order
    ops = aug_cfg.get("ops") or []
    label = "yaml:" + ",".join(
        str(op.get("name")) for op in ops if isinstance(op, dict) and op.get("name")
    )

    curriculum_scheduler = None
    curriculum_phase_percents: List[float] = []
    if curriculum_cfg is not None:
        curriculum_scheduler = AugmentationCurriculumScheduler.from_config(
            base_bypass=base_bypass,
            op_meta=getattr(compose, "_augmentation_meta", []),
            curriculum_raw=curriculum_cfg,
        )
        for phase in curriculum_cfg.get("phases", []):
            if not isinstance(phase, dict):
                continue
            raw = phase.get("until_percent")
            if raw is None:
                continue
            try:
                up = float(raw)
                curriculum_phase_percents.append(up if up > 1 else up * 100.0)
            except Exception:
                continue

    return PipelineSpec(
        compose=compose,
        label=label,
        bypass_prob=base_bypass,
        curriculum=curriculum_scheduler,
        curriculum_phase_percents=curriculum_phase_percents,
        augmentation_cfg=aug_cfg,
    )


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

    # === ROI CROPPING (barrier - filters labels) ===
    if rng.random() < cfg.roi_crop_p:
        ops.append(
            RoiCrop(
                anchor_classes=cfg.roi_crop_anchor_classes,
                scale_range=(cfg.roi_crop_scale_lo, cfg.roi_crop_scale_hi),
                min_crop_size=cfg.roi_crop_min_crop_size,
                min_coverage=cfg.roi_crop_min_coverage,
                completeness_threshold=cfg.roi_crop_completeness_threshold,
                prob=1.0,
            )
        )
        labels.append(
            f"roi_crop({cfg.roi_crop_scale_lo:.1f}-{cfg.roi_crop_scale_hi:.1f})"
        )

    # === RESOLUTION RESIZING (barrier - tests coordinate scaling) ===
    if rng.random() < cfg.resize_by_scale_p:
        ops.append(
            ResizeByScale(
                lo=cfg.resize_lo,
                hi=cfg.resize_hi,
                align_multiple=cfg.resize_align_multiple,
                prob=1.0,
            )
        )
        labels.append(f"resize({cfg.resize_lo:.1f}-{cfg.resize_hi:.1f})")

    # === COLOR AUGMENTATIONS (deferred, applied after all geometric ops) ===
    # No color ops for alignment visualization

    return Compose(ops), "|".join(labels)


def _load_jsonl_head(jsonl_path: str, limit: int) -> List[Dict[str, Any]]:
    """Load only the first `limit` records to avoid pulling full datasets into memory."""
    out: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if limit and len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _coerce_bypass(value: Any, default: float) -> float:
    """Convert scheduler bypass value (scalar or 2-tuple) to float."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and value:
        try:
            return float(value[-1])
        except Exception:
            return default
    try:
        return float(value)
    except Exception:
        return default


def _apply_curriculum_overrides(pipeline: Compose, overrides: Dict[str, Any]) -> None:
    """Apply curriculum op overrides using the same coercion logic as training preprocessor."""
    pre = AugmentationPreprocessor(augmenter=pipeline)
    pre._apply_curriculum_overrides(overrides)


def _derive_stage_marks(cfg: VisConfig, spec: PipelineSpec) -> List[float]:
    """Merge user-provided marks, curriculum boundaries, and key checkpoints."""
    marks: List[float] = []
    if cfg.curriculum_marks:
        marks.extend(cfg.curriculum_marks)
    if spec.curriculum_phase_percents:
        marks.extend(spec.curriculum_phase_percents)
    # Always include start/end and 30% (user-observed overfitting)
    marks.extend([0.0, 30.0, 100.0])
    uniq = sorted({round(min(100.0, max(0.0, m)), 2) for m in marks})
    return uniq


def _build_stage_specs(spec: PipelineSpec, cfg: VisConfig) -> List[StageSpec]:
    """Create per-curriculum stages with cloned pipelines and resolved overrides."""
    if spec.curriculum is None:
        return [
            StageSpec(
                tag="p100",
                percent=100.0,
                pipeline=spec.compose,
                bypass_prob=spec.bypass_prob,
                state={"ops": {}, "bypass_prob": spec.bypass_prob},
                label=spec.label,
            )
        ]

    scheduler = copy.deepcopy(spec.curriculum)
    scheduler.set_total_steps(max(1, int(cfg.curriculum_total_steps)))
    marks = _derive_stage_marks(cfg, spec)

    stages: List[StageSpec] = []
    for pct in marks:
        step = int(round(cfg.curriculum_total_steps * (pct / 100.0)))
        state = scheduler.get_state(step)
        pipe = copy.deepcopy(spec.compose)
        _apply_curriculum_overrides(pipe, state.get("ops") or {})
        bypass_prob = _coerce_bypass(state.get("bypass_prob"), spec.bypass_prob)
        tag = f"p{int(round(pct)):03d}"
        stages.append(
            StageSpec(
                tag=tag,
                percent=pct,
                pipeline=pipe,
                bypass_prob=bypass_prob,
                state=state,
                label=f"{spec.label}|{tag}",
            )
        )
    return stages


def visualize_samples(cfg: VisConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Auto-set ROOT_IMAGE_DIR to match training loader behavior
    if os.environ.get("ROOT_IMAGE_DIR") in (None, ""):
        root_dir = str(Path(cfg.jsonl_path).resolve().parent)
        os.environ["ROOT_IMAGE_DIR"] = root_dir
        print(f"[INFO] ROOT_IMAGE_DIR set to {root_dir}")

    records = _load_jsonl_head(cfg.jsonl_path, cfg.num_samples)

    base_rng = Random(cfg.seed)

    # Prefer exact training augmentation from YAML if provided
    pipeline_spec: PipelineSpec | None = None
    if cfg.config_yaml:
        try:
            pipeline_spec = _build_pipeline_from_yaml(cfg)
            print(f"[INFO] Using augmentation from YAML: {cfg.config_yaml}")
        except Exception as e:
            print(f"[ERROR] Failed to load training YAML augmentation: {e}")
            print("[INFO] Falling back to extreme testing pipeline")

    use_yaml = pipeline_spec is not None
    stage_specs: List[StageSpec] = []
    if use_yaml and pipeline_spec is not None:
        stage_specs = _build_stage_specs(pipeline_spec, cfg)
        print(
            f"[INFO] Curriculum checkpoints: {[f'{s.percent:.1f}%' for s in stage_specs]}"
        )

    state_log: List[Dict[str, Any]] = []

    def _summarize_ops(state_ops: Dict[str, Any]) -> str:
        parts: List[str] = []
        for name, params in sorted(state_ops.items()):
            prob = params.get("prob")
            core = ""
            if prob is not None:
                try:
                    core = f"p={float(prob):.2f}"
                except Exception:
                    core = f"p={prob}"
            ranges = []
            for key in (
                "max_deg",
                "scale",
                "lo",
                "hi",
                "brightness",
                "contrast",
                "saturation",
                "hue_delta_deg",
            ):
                if key in params:
                    val = params[key]
                    if isinstance(val, (list, tuple)) and len(val) == 2:
                        ranges.append(f"{key}={val[0]:.2f}-{val[1]:.2f}")
                    else:
                        try:
                            ranges.append(f"{key}={float(val):.2f}")
                        except Exception:
                            ranges.append(f"{key}={val}")
            tail = ("," + ",".join(ranges)) if ranges else ""
            parts.append(f"{name}({core}{tail})" if core else f"{name}")
        return "; ".join(parts) if parts else "base"

    # === MAIN LOOP ===
    if use_yaml and stage_specs:
        for stage_idx, stage in enumerate(stage_specs):
            stage_dir = os.path.join(cfg.out_dir, stage.tag)
            os.makedirs(stage_dir, exist_ok=True)
            ops_summary = _summarize_ops(stage.state.get("ops", {}))
            print(
                f"[STAGE] {stage.tag} ({stage.percent:.1f}%): bypass={stage.bypass_prob:.3f} | {ops_summary}"
            )
            state_log.append(
                {
                    "tag": stage.tag,
                    "percent": stage.percent,
                    "bypass_prob": stage.bypass_prob,
                    "ops": stage.state.get("ops", {}),
                }
            )

            for idx, rec in enumerate(records):
                images = rec.get("images") or []
                objs = rec.get("objects") or []
                per_obj_geoms: List[DatasetObject] = []
                per_obj_descs: List[str] = []
                for o in objs:
                    g_raw: Dict[str, Any] = {}
                    if o.get("bbox_2d") is not None:
                        g_raw["bbox_2d"] = o["bbox_2d"]
                    if o.get("poly") is not None:
                        g_raw["poly"] = o["poly"]
                    if o.get("line") is not None:
                        g_raw["line"] = o["line"]
                    if g_raw:
                        per_obj_geoms.append(cast(DatasetObject, g_raw))
                        per_obj_descs.append(_extract_description(o))

                # Resolve images to PIL
                pil_images: List[Image.Image] = []
                for it in images:
                    pil_images.append(_load_pil(it, jsonl_path=cfg.jsonl_path))

                im0 = pil_images[0]
                objs0 = _geom_to_objects(per_obj_geoms, per_obj_descs)

                variants_imgs: List[Image.Image] = []
                variants_objs: List[List[Dict[str, Any]]] = []
                variant_titles: List[str] = []

                for j in range(int(cfg.variants)):
                    rng = Random(cfg.seed + idx * 1000 + stage_idx * 100 + j)
                    bypassed = False
                    if cfg.respect_bypass_prob and rng.random() < stage.bypass_prob:
                        out_imgs = pil_images
                        geoms_new = per_obj_geoms
                        bypassed = True
                    else:
                        out_imgs, geoms_new = stage.pipeline.apply(
                            pil_images,
                            per_obj_geoms,
                            width=im0.width,
                            height=im0.height,
                            rng=rng,
                        )

                    if not bypassed and len(geoms_new) != len(per_obj_geoms):
                        print(
                            f"  [CROP] {stage.tag} sample {idx}, variant {j + 1}: {len(per_obj_geoms)} → {len(geoms_new)} objects"
                        )
                        if (
                            hasattr(stage.pipeline, "last_kept_indices")
                            and stage.pipeline.last_kept_indices is not None
                        ):
                            print(
                                f"         Kept indices: {stage.pipeline.last_kept_indices}"
                            )
                            if (
                                hasattr(stage.pipeline, "last_object_coverages")
                                and stage.pipeline.last_object_coverages
                            ):
                                avg_cov = sum(
                                    stage.pipeline.last_object_coverages
                                ) / len(stage.pipeline.last_object_coverages)
                                print(
                                    f"         Avg coverage of kept objects: {avg_cov:.2%}"
                                )

                    variants_imgs.append(
                        _load_pil(out_imgs[0], jsonl_path=cfg.jsonl_path)
                    )
                    kept_indices = getattr(stage.pipeline, "last_kept_indices", None)
                    if isinstance(kept_indices, list):
                        descs_new = [
                            per_obj_descs[i]
                            for i in kept_indices
                            if i < len(per_obj_descs)
                        ]
                    elif len(geoms_new) == len(per_obj_descs):
                        descs_new = per_obj_descs
                    else:
                        descs_new = per_obj_descs[: len(geoms_new)]
                    variants_objs.append(_geom_to_objects(geoms_new, descs_new))

                    if bypassed:
                        variant_titles.append(
                            f"{stage.tag} v{j + 1}: bypass ({stage.bypass_prob:.2f})"
                        )
                    else:
                        variant_titles.append(f"{stage.tag} v{j + 1}: {ops_summary}")

                # Export as pairwise comparisons only (Original vs Aug) to preserve resolution.
                # One JPG per variant: vis_XXXXX_vYY.jpg (or vis_XXXXX.jpg if only 1 variant).
                if not variants_imgs:
                    variants_imgs = [im0]
                    variants_objs = [objs0]
                    variant_titles = [f"{stage.tag}: no_aug"]

                # Keep a stable color map across variants for the same sample/stage.
                labels_all = [o["desc"] for o in objs0]
                for arr in variants_objs:
                    labels_all.extend([o["desc"] for o in arr])
                color_map = generate_colors(labels_all)

                for v_idx, (aug_img, aug_objs, aug_title) in enumerate(
                    zip(variants_imgs, variants_objs, variant_titles), start=1
                ):
                    counts: Dict[str, List[int]] = {}
                    for o in objs0:
                        key = o.get("desc", "")
                        counts.setdefault(key, [0, 0])[0] += 1
                    for o in aug_objs:
                        key = o.get("desc", "")
                        counts.setdefault(key, [0, 0])[1] += 1

                    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                    draw_objects(axes[0], im0, objs0, color_map, scaled=True)
                    axes[0].set_title(f"Original (GT) - {len(objs0)} objects")

                    draw_objects(axes[1], aug_img, aug_objs, color_map, scaled=True)
                    obj_count = len(aug_objs)
                    title = f"{aug_title}\n{obj_count} objects"
                    if obj_count != len(objs0):
                        title += f" ({obj_count - len(objs0):+d})"
                    axes[1].set_title(title, fontsize=10)

                    create_legend(fig, color_map, counts)
                    suffix = "" if len(variants_imgs) == 1 else f"_v{v_idx:02d}"
                    out_path = os.path.join(stage_dir, f"vis_{idx:05d}{suffix}.jpg")
                    fig.tight_layout()
                    fig.savefig(out_path, dpi=160)
                    plt.close(fig)
                    print(f"[INFO] Saved {out_path}")
    else:
        # Fallback to extreme random testing (no YAML)
        print("[INFO] Using extreme testing pipeline (config_yaml=None)")
        for idx, rec in enumerate(records):
            images = rec.get("images") or []
            objs = rec.get("objects") or []
            per_obj_geoms: List[DatasetObject] = []
            per_obj_descs: List[str] = []
            for o in objs:
                g_raw: Dict[str, Any] = {}
                if o.get("bbox_2d") is not None:
                    g_raw["bbox_2d"] = o["bbox_2d"]
                if o.get("poly") is not None:
                    g_raw["poly"] = o["poly"]
                if o.get("line") is not None:
                    g_raw["line"] = o["line"]
                if g_raw:
                    per_obj_geoms.append(cast(DatasetObject, g_raw))
                    per_obj_descs.append(_extract_description(o))

            pil_images: List[Image.Image] = []
            for it in images:
                pil_images.append(_load_pil(it, jsonl_path=cfg.jsonl_path))

            im0 = pil_images[0]
            objs0 = _geom_to_objects(per_obj_geoms, per_obj_descs)

            variants_imgs: List[Image.Image] = []
            variants_objs: List[List[Dict[str, Any]]] = []
            variant_titles: List[str] = []

            for j in range(int(cfg.variants)):
                rng = Random(base_rng.random())
                pipe, title = _build_random_pipeline(rng, cfg)
                out_imgs, geoms_new = pipe.apply(
                    pil_images,
                    per_obj_geoms,
                    width=im0.width,
                    height=im0.height,
                    rng=rng,
                )

                if len(geoms_new) != len(per_obj_geoms):
                    print(
                        f"  [CROP] Sample {idx}, variant {j + 1}: {len(per_obj_geoms)} → {len(geoms_new)} objects"
                    )
                    if (
                        hasattr(pipe, "last_kept_indices")
                        and pipe.last_kept_indices is not None
                    ):
                        print(f"         Kept indices: {pipe.last_kept_indices}")
                        if (
                            hasattr(pipe, "last_object_coverages")
                            and pipe.last_object_coverages
                        ):
                            avg_cov = sum(pipe.last_object_coverages) / len(
                                pipe.last_object_coverages
                            )
                            print(
                                f"         Avg coverage of kept objects: {avg_cov:.2%}"
                            )

                variants_imgs.append(_load_pil(out_imgs[0], jsonl_path=cfg.jsonl_path))
                kept_indices = getattr(pipe, "last_kept_indices", None)
                if isinstance(kept_indices, list):
                    descs_new = [
                        per_obj_descs[i] for i in kept_indices if i < len(per_obj_descs)
                    ]
                elif len(geoms_new) == len(per_obj_descs):
                    descs_new = per_obj_descs
                else:
                    descs_new = per_obj_descs[: len(geoms_new)]
                variants_objs.append(_geom_to_objects(geoms_new, descs_new))
                variant_titles.append(title)

            # Export as pairwise comparisons only (Original vs Aug) to preserve resolution.
            # One JPG per variant: vis_XXXXX_vYY.jpg (or vis_XXXXX.jpg if only 1 variant).
            if not variants_imgs:
                variants_imgs = [im0]
                variants_objs = [objs0]
                variant_titles = ["no_aug"]

            # Keep a stable color map across variants for the same sample.
            labels_all = [o["desc"] for o in objs0]
            for arr in variants_objs:
                labels_all.extend([o["desc"] for o in arr])
            color_map = generate_colors(labels_all)

            for v_idx, (aug_img, aug_objs, aug_title) in enumerate(
                zip(variants_imgs, variants_objs, variant_titles), start=1
            ):
                counts: Dict[str, List[int]] = {}
                for o in objs0:
                    key = o.get("desc", "")
                    counts.setdefault(key, [0, 0])[0] += 1
                for o in aug_objs:
                    key = o.get("desc", "")
                    counts.setdefault(key, [0, 0])[1] += 1

                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                draw_objects(axes[0], im0, objs0, color_map, scaled=True)
                axes[0].set_title(f"Original (GT) - {len(objs0)} objects")

                draw_objects(axes[1], aug_img, aug_objs, color_map, scaled=True)
                obj_count = len(aug_objs)
                title = f"Aug v{v_idx}: {aug_title}\n{obj_count} objects"
                if obj_count != len(objs0):
                    title += f" ({obj_count - len(objs0):+d})"
                axes[1].set_title(title, fontsize=10)

                create_legend(fig, color_map, counts)
                suffix = "" if len(variants_imgs) == 1 else f"_v{v_idx:02d}"
                out_path = os.path.join(cfg.out_dir, f"vis_{idx:05d}{suffix}.jpg")
                fig.tight_layout()
                fig.savefig(out_path, dpi=160)
                plt.close(fig)
                print(f"[INFO] Saved {out_path}")

    if cfg.save_state_summary and state_log:
        summary_path = os.path.join(cfg.out_dir, "curriculum_states.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config_yaml": cfg.config_yaml,
                    "jsonl": cfg.jsonl_path,
                    "marks": [s["percent"] for s in state_log],
                    "states": state_log,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[INFO] Wrote curriculum summary → {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize data augmentation and curriculum effects."
    )
    parser.add_argument(
        "--mode",
        choices=["curriculum", "roi_crop"],
        default="curriculum",
        help="Visualization mode: 'curriculum' for full pipeline with curriculum, "
        "'roi_crop' for roi_crop only.",
    )
    parser.add_argument(
        "--config-yaml",
        default="configs/train/sft/dense_1024.yaml",
        help="Training YAML with custom.augmentation config.",
    )
    parser.add_argument(
        "--jsonl",
        default="data_new_schema/bbu_full_1024/train.jsonl",
        help="Input train JSONL with images/objects.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default depends on mode).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=12,
        help="Number of records to visualize.",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=3,
        help="Number of augmented variants per record.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Base RNG seed.",
    )
    args = parser.parse_args()

    if args.out_dir:
        out_dir = args.out_dir
    elif args.mode == "roi_crop":
        out_dir = "vis_out/augment_roi_crop_focus"
    else:
        out_dir = "vis_out/augment_curriculum_dense_1024"

    base_kwargs = dict(
        jsonl_path=args.jsonl,
        out_dir=out_dir,
        num_samples=args.num_samples,
        variants=args.variants,
        seed=args.seed,
        config_yaml=args.config_yaml,
    )

    if args.mode == "roi_crop":
        cfg = VisConfig(
            **base_kwargs,
            curriculum_marks=None,
            respect_bypass_prob=False,  # always apply op; bypass handled via focus_prob
            focus_op_name="roi_crop",
            focus_force_prob=1.0,
        )
        print("=" * 72)
        print("ROI_CROP VISUALIZATION")
        print(f"  YAML:          {cfg.config_yaml}")
        print(f"  JSONL:         {cfg.jsonl_path}")
        print(f"  Output Dir:    {cfg.out_dir}")
        print(f"  Samples:       {cfg.num_samples} images × {cfg.variants} variants")
        print(f"  Focus op:      {cfg.focus_op_name} (prob={cfg.focus_force_prob})")
        print(f"  Seed:          {cfg.seed}")
        print("=" * 72)
    else:
        cfg = VisConfig(
            **base_kwargs,
            curriculum_marks=[0, 5, 20, 30, 45, 70, 100],
            curriculum_total_steps=1000,
        )
        print("=" * 72)
        print("CURRICULUM AUGMENTATION VISUALIZATION")
        print(f"  YAML:          {cfg.config_yaml}")
        print(f"  JSONL:         {cfg.jsonl_path}")
        print(f"  Output Dir:    {cfg.out_dir}")
        print(f"  Samples:       {cfg.num_samples} images × {cfg.variants} variants")
        print(f"  Curriculum %:  {cfg.curriculum_marks}")
        print(f"  Total steps:   {cfg.curriculum_total_steps}")
        print(f"  Seed:          {cfg.seed}")
        print(f"  Respect bypass:{cfg.respect_bypass_prob}")
        print("=" * 72)

    visualize_samples(cfg)
