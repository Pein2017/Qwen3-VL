from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from .base import Compose, ImageAugmenter
from .registry import get as get_augmenter


def build_compose_from_config(cfg: Dict[str, Any]) -> Compose:
    """Build a Compose pipeline from a simple dict schema.

    Schema example (list of ops):
      ops:
        - name: rotate
          params: { max_deg: 8.0, prob: 0.3 }
        - name: hflip
          params: { prob: 0.5 }
        - name: color_jitter
          params: { brightness: [0.8, 1.2], contrast: [0.8, 1.2], saturation: [0.8, 1.2], prob: 0.5 }
        - name: pad_to_multiple
          params: { multiple: 32 }

    Returns:
      Compose([...])
    """
    if cfg is None:
        raise ValueError("augmentation config is required to build pipeline")
    ops_cfg: List[Dict[str, Any]] = cfg.get("ops") or []
    if not isinstance(ops_cfg, list):
        raise TypeError("augmentation.ops must be a list of operations")
    ops: List[ImageAugmenter] = []
    ops_meta: List[Dict[str, Any]] = []
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
        ops_meta.append({"name": name, "params": deepcopy(params_copy)})
    pipeline = Compose(ops)
    pipeline._augmentation_meta = ops_meta  # type: ignore[attr-defined]
    name_map: Dict[str, List[ImageAugmenter]] = {}
    for meta, instance in zip(ops_meta, ops):
        name_map.setdefault(meta["name"], []).append(instance)
    pipeline._augmentation_name_map = name_map  # type: ignore[attr-defined]
    return pipeline


__all__ = ["build_compose_from_config"]
