from __future__ import annotations

from copy import deepcopy
from typing import cast

from .base import AugmentationMeta, Compose, ImageAugmenter
from .curriculum import NumericParam
from .registry import get as get_augmenter
from . import ops as _register_builtin_ops  # noqa: F401
from src.utils import require_mapping
from src.utils.unstructured import UnstructuredMapping


def build_compose_from_config(cfg: UnstructuredMapping) -> Compose:
    """Build a Compose pipeline from a simple dict schema.

    Schema example (list of ops):
      ops:
        - name: rotate
          params: { max_deg: 8.0, prob: 0.3 }
        - name: hflip
          params: { prob: 0.5 }
        - name: color_jitter
          params: { brightness: [0.8, 1.2], contrast: [0.8, 1.2], saturation: [0.8, 1.2], prob: 0.5 }
        - name: expand_to_fit_affine
          params: { multiple: 32 }

    Returns:
      Compose([...])
    """
    if cfg is None:
        raise ValueError("augmentation config is required to build pipeline")
    cfg = require_mapping(cfg, context="augmentation.config")
    ops_cfg: list[UnstructuredMapping] = cast(
        list[UnstructuredMapping], cfg.get("ops") or []
    )
    if not isinstance(ops_cfg, list):
        raise TypeError("augmentation.ops must be a list of operations")
    ops: list[ImageAugmenter] = []
    ops_meta: list[AugmentationMeta] = []
    curriculum_base: dict[str, dict[str, NumericParam]] = {}

    def _is_prob_field(name: str) -> bool:
        n = name.lower()
        return n == "prob" or n.endswith("_prob")

    for op in ops_cfg:
        if not isinstance(op, dict):
            raise TypeError(
                "each augmentation op must be a dict with 'name' and optional 'params'"
            )
        name = op.get("name")
        params = op.get("params") or {}
        if not name:
            raise ValueError("augmentation op missing 'name'")
        cls = get_augmenter(str(name))
        if not isinstance(params, dict):
            raise TypeError("augmentation op 'params' must be a dict")
        params_copy = dict(params)
        augmenter_instance = cls(**params_copy)
        setattr(augmenter_instance, "_aug_name", name)
        ops.append(augmenter_instance)
        # Capture curriculum-exposed numeric params (typed) from the instance
        curr_params: dict[str, NumericParam] = {}
        raw_curr = getattr(augmenter_instance, "curriculum_params", None)
        if callable(raw_curr):
            raw_curr = raw_curr()
        if isinstance(raw_curr, dict):
            for param_name, value in raw_curr.items():
                numeric = (
                    value
                    if isinstance(value, NumericParam)
                    else NumericParam.from_raw(value)
                )
                if _is_prob_field(param_name):
                    for v in numeric.values:
                        if v < 0.0 or v > 1.0:
                            raise ValueError(
                                f"augmentation op '{name}' param '{param_name}' must be within [0, 1]; got {v}"
                            )
                curr_params[param_name] = numeric
                curriculum_base.setdefault(name, {}).update({param_name: numeric})

        meta_entry: AugmentationMeta = {"name": name, "params": deepcopy(params_copy)}
        if curr_params:
            meta_entry["curriculum_params"] = {
                k: v.to_python_value() for k, v in curr_params.items()
            }
        ops_meta.append(meta_entry)
    pipeline = Compose(ops)
    pipeline._augmentation_meta = ops_meta  # type: ignore[attr-defined]
    name_map: dict[str, list[ImageAugmenter]] = {}
    for meta, instance in zip(ops_meta, ops):
        name_map.setdefault(meta["name"], []).append(instance)
    pipeline._augmentation_name_map = name_map  # type: ignore[attr-defined]
    pipeline._curriculum_base_ops = curriculum_base  # type: ignore[attr-defined]
    return pipeline


__all__ = ["build_compose_from_config"]
