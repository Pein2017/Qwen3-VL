"""Augmentation preprocessor - decoupled from dataset logic"""

import copy
import random
import re
from collections.abc import Mapping, MutableMapping
from typing import Any, cast

from PIL import Image as PILImage

from ...utils.logger import get_logger
from ..augmentation.curriculum import NumericParam, _build_base_ops
from ..contracts import AugmentationTelemetry, ConversationRecord
from ..utils import extract_geometry
from .base import BasePreprocessor


class AugmentationPreprocessor(BasePreprocessor):
    """Preprocessor that applies augmentations to records.

    Decouples augmentation logic from dataset __getitem__, making it:
    - Reusable across different datasets
    - Testable independently
    - Composable with other preprocessors
    """

    def __init__(
        self,
        *,
        augmenter: object | None = None,
        rng: random.Random | None = None,
        bypass_prob: float = 0.0,
        curriculum_state: MutableMapping[str, object] | None = None,
        **kwargs: object,
    ):
        """Initialize augmentation preprocessor.

        Args:
            augmenter: AugmentationConfig instance
            rng: Random number generator for reproducibility
            bypass_prob: Probability (0-1) of bypassing augmentation for clean samples
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.augmenter = augmenter
        self.rng = rng if rng is not None else random.Random()
        self.bypass_prob = float(bypass_prob)
        self.curriculum_state = curriculum_state
        self._curriculum_last_step: int | None = None

    def preprocess(self, row: ConversationRecord) -> ConversationRecord | None:
        """Apply augmentations to a record.

        Args:
            row: Input record with images and objects

        Returns:
            Augmented record
        """
        if self.augmenter is None:
            return row

        if self.curriculum_state is not None:
            self._sync_curriculum()

        # Randomly bypass augmentation for clean samples
        if self.rng.random() < self.bypass_prob:
            return row

        row = self._augment_record(row)

        return row

    def _augment_record(self, rec: ConversationRecord) -> ConversationRecord:
        """Apply augmentation to a single record.

        Args:
            rec: Record to augment

        Returns:
            Augmented record
        """
        from ..augment import apply_augmentations

        # Only plugin registry path is supported
        try:
            from ..augmentation.base import Compose
        except Exception:
            Compose = None  # type: ignore
            _get = None  # type: ignore

        rec_map = cast(MutableMapping[str, object], cast(object, rec))
        images: list[str | PILImage.Image] = cast(
            list[str | PILImage.Image], rec_map.get("images") or []
        )
        objs: list[dict[str, object]] = cast(
            list[dict[str, object]], rec_map.get("objects") or []
        )

        # Extract geometries and keep an index mapping back to the object list.
        # This lets us append new duplicated objects when PatchOps increase geometry count
        # (e.g., small_object_zoom_paste).
        per_obj_geoms: list[dict[str, object]] = []
        obj_idx_with_geom: list[int] = []
        for idx, obj in enumerate(objs):
            g = cast(dict[str, object], extract_geometry(obj))
            if g:
                # Attach desc metadata for class-aware PatchOps (e.g., small_object_zoom_paste).
                # This enables whitelist matching against the full desc string, including 备注/文本.
                desc = obj.get("desc")
                if desc is not None:
                    g["desc"] = str(desc)
                per_obj_geoms.append(g)
                obj_idx_with_geom.append(idx)

        # Apply augmentations using unified pipeline API only
        if Compose is None or not isinstance(self.augmenter, Compose):
            raise TypeError(
                "AugmentationPreprocessor requires 'augmenter' to be a Compose pipeline."
            )

        pipeline = self.augmenter
        images_bytes, per_obj_geoms_new = apply_augmentations(
            images, per_obj_geoms, pipeline, rng=self.rng
        )

        telemetry: AugmentationTelemetry | None = getattr(
            pipeline, "last_summary", None
        )

        if telemetry is None or not telemetry.kept_indices:
            # No crop applied (or crop was skipped) - update geometries only
            if len(per_obj_geoms_new) < len(obj_idx_with_geom):
                raise ValueError(
                    "augmentation returned fewer geometries than input objects without crop telemetry: "
                    f"got {len(per_obj_geoms_new)}, expected >= {len(obj_idx_with_geom)}"
                )

            # 1) Update existing objects in-place.
            for j, obj_idx in enumerate(obj_idx_with_geom):
                self._update_geometry_field(objs[obj_idx], per_obj_geoms_new[j])

            # 2) Append any extra geometries as duplicated objects (labeled).
            extra_geoms = per_obj_geoms_new[len(obj_idx_with_geom) :]
            if extra_geoms and obj_idx_with_geom:
                appended_objs: list[dict[str, object]] = []
                for geom in extra_geoms:
                    src_idx = geom.get("__src_geom_idx")
                    src_idx_int = 0
                    if isinstance(src_idx, (int, float, str)):
                        try:
                            src_idx_int = int(src_idx)
                        except (TypeError, ValueError):
                            src_idx_int = 0
                    if src_idx_int < 0 or src_idx_int >= len(obj_idx_with_geom):
                        src_idx_int = 0

                    src_obj = objs[obj_idx_with_geom[src_idx_int]]
                    dup_obj = copy.deepcopy(src_obj)
                    self._update_geometry_field(dup_obj, geom)
                    appended_objs.append(dup_obj)
                objs.extend(appended_objs)
                rec_map["objects"] = objs
        else:
            # Crop was applied - filter objects and update completeness
            filtered_objects: list[dict[str, object]] = []
            kept_indices = list(telemetry.kept_indices)
            coverages = list(telemetry.coverages)

            # Get completeness threshold from the pipeline's last crop operator
            # (For now, we assume the crop operator is accessible; alternatively,
            # we could store completeness_threshold as pipeline metadata)
            completeness_threshold = 0.95  # Default
            # Try to get from pipeline's crop operators
            for op in pipeline.ops if hasattr(pipeline, "ops") else []:
                if hasattr(op, "completeness_threshold"):
                    completeness_threshold = getattr(op, "completeness_threshold", 0.95)
                    break

            if len(per_obj_geoms_new) < len(kept_indices):
                raise ValueError(
                    "augmentation returned fewer geometries than kept crop indices: "
                    f"got {len(per_obj_geoms_new)}, expected >= {len(kept_indices)}"
                )

            # Filter and update objects
            completeness_updates = 0
            structured_updates = 0
            for idx_in_kept, orig_idx in enumerate(kept_indices):
                # orig_idx refers to the index in the original geometries list
                # Map back to the object index
                obj_idx = obj_idx_with_geom[orig_idx]
                obj = objs[obj_idx]
                new_geom = per_obj_geoms_new[idx_in_kept]
                cov = coverages[idx_in_kept] if idx_in_kept < len(coverages) else 1.0

                # Update geometry field (single field only)
                self._update_geometry_field(obj, new_geom)

                # Update completeness field if below threshold
                if cov < completeness_threshold:
                    desc = str(obj.get("desc", ""))
                    if "可见性=完整" in desc:
                        obj["desc"] = desc.replace("可见性=完整", "可见性=部分")
                        completeness_updates += 1
                    elif "可见性=显示完整" in desc:
                        # Legacy key=value tokens.
                        obj["desc"] = desc.replace(
                            "可见性=显示完整", "可见性=只显示部分"
                        )
                        completeness_updates += 1
                    elif "显示完整" in desc:
                        # Legacy (slash-delimited) completeness token fallback.
                        obj["desc"] = desc.replace("显示完整", "只显示部分")
                        completeness_updates += 1
                    elif "完整" in desc:
                        obj["desc"] = desc.replace("完整", "部分")
                        completeness_updates += 1
                    # Structured completeness metadata support
                    attrs = obj.get("attributes")
                    if isinstance(attrs, dict):
                        completeness_key = None
                        for key in ("completeness", "完整性", "complete"):
                            if key in attrs:
                                completeness_key = key
                                break
                        if completeness_key is not None:
                            value = attrs.get(completeness_key)
                            if value in {"显示完整", "完整"}:
                                attrs[completeness_key] = (
                                    "只显示部分" if value == "显示完整" else "部分"
                                )
                                structured_updates += 1
                        else:
                            logger = get_logger("augmentation.preprocessor")
                            logger.debug(
                                "Completeness metadata missing expected key on object; desc_updated=%s",
                                ("可见性=显示完整" in desc) or ("显示完整" in desc),
                            )

                filtered_objects.append(obj)

            # Append any extra geometries as duplicated objects (labeled).
            extra_geoms = per_obj_geoms_new[len(kept_indices) :]
            if extra_geoms and filtered_objects:
                appended_objs_crop: list[dict[str, object]] = []
                for geom in extra_geoms:
                    src_idx = geom.get("__src_geom_idx")
                    src_idx_int = 0
                    if isinstance(src_idx, (int, float, str)):
                        try:
                            src_idx_int = int(src_idx)
                        except (TypeError, ValueError):
                            src_idx_int = 0
                    if src_idx_int < 0 or src_idx_int >= len(filtered_objects):
                        src_idx_int = 0

                    src_obj = filtered_objects[src_idx_int]
                    dup_obj = copy.deepcopy(src_obj)
                    self._update_geometry_field(dup_obj, geom)
                    appended_objs_crop.append(dup_obj)
                filtered_objects.extend(appended_objs_crop)

            # Replace objects list with filtered objects
            rec_map["objects"] = filtered_objects

            # Log crop filtering results (debug level)
            logger = get_logger("augmentation.preprocessor")
            logger.debug(
                f"Crop filter: {len(objs)} → {len(filtered_objects)} objects "
                f"({completeness_updates} desc updates, {structured_updates} attribute updates)"
            )

        rec_map["images"] = images_bytes

        # Update record width/height to reflect any resize/pad ops in augmentation
        try:
            if images_bytes:
                import io

                from PIL import Image  # type: ignore

                b0 = images_bytes[0].get("bytes")
                if isinstance(b0, (bytes, bytearray)):
                    with Image.open(io.BytesIO(b0)) as im0:
                        im0 = im0.convert("RGB")
                        rec_map["width"] = int(im0.width)
                        rec_map["height"] = int(im0.height)
        except Exception:
            # Non-fatal: leave original width/height
            pass

        return rec

    def _sync_curriculum(self) -> None:
        state = self.curriculum_state
        if state is None:
            return
        step = state.get("step")
        try:
            step = int(cast(Any, step)) if step is not None else 0
        except (TypeError, ValueError):
            step = 0
        if self._curriculum_last_step == step:
            return
        bypass = state.get("bypass_prob")
        if bypass is not None:
            try:
                value = float(cast(Any, bypass))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Curriculum bypass_prob must be numeric; got {bypass}"
                ) from exc
            if value < 0.0 or value > 1.0:
                raise ValueError(
                    f"Curriculum bypass_prob must be within [0, 1]; got {value}"
                )
            self.bypass_prob = value
        ops = state.get("ops") or {}
        if isinstance(ops, Mapping):
            self._apply_curriculum_overrides(ops)
        self._curriculum_last_step = step

    def _apply_curriculum_overrides(
        self, overrides: Mapping[str, Mapping[str, object]]
    ) -> None:
        if self.augmenter is None:
            return

        def _is_prob_field(name: str) -> bool:
            n = str(name).lower()
            return n == "prob" or n.endswith("_prob")

        def _coerce_value(current: object, new_value: object) -> object:
            """Preserve operator parameter types when applying overrides."""
            if current is None:
                return new_value
            if isinstance(current, bool):
                return bool(new_value)
            if isinstance(current, int):
                return int(round(cast(Any, new_value)))
            if isinstance(current, float):
                return float(cast(Any, new_value))
            if isinstance(current, tuple):
                if not isinstance(new_value, (list, tuple)):
                    raise TypeError("tuple override must be list/tuple")
                return tuple(
                    _coerce_value(current[i], new_value[i])
                    for i in range(len(new_value))
                )
            if isinstance(current, list):
                if not isinstance(new_value, (list, tuple)):
                    raise TypeError("list override must be list/tuple")
                return [
                    _coerce_value(current[i], new_value[i])
                    for i in range(len(new_value))
                ]
            return new_value

        name_map = getattr(self.augmenter, "_augmentation_name_map", {}) or {}
        if not name_map:
            for op in getattr(self.augmenter, "ops", []):
                n = (
                    getattr(op, "_aug_name", None)
                    or re.sub(r"(?<!^)(?=[A-Z])", "_", op.__class__.__name__).lower()
                )
                name_map.setdefault(n, []).append(op)
        base_map: dict[str, dict[str, NumericParam]] = (
            getattr(self.augmenter, "_curriculum_base_ops", {}) or {}
        )
        if not base_map:
            meta = getattr(self.augmenter, "_augmentation_meta", [])
            base_map = _build_base_ops(meta)
        if not base_map:
            for op in getattr(self.augmenter, "ops", []):
                n = (
                    getattr(op, "_aug_name", None)
                    or re.sub(r"(?<!^)(?=[A-Z])", "_", op.__class__.__name__).lower()
                )
                raw_curr = getattr(op, "curriculum_params", None)
                if callable(raw_curr):
                    raw_curr = raw_curr()
                if isinstance(raw_curr, Mapping):
                    for param_name, value in raw_curr.items():
                        numeric = (
                            value
                            if isinstance(value, NumericParam)
                            else NumericParam.from_raw(value)
                        )
                        base_map.setdefault(n, {})[param_name] = numeric

        for op_name, param_overrides in overrides.items():
            if op_name not in base_map:
                raise ValueError(
                    f"Curriculum override references unknown op '{op_name}'"
                )
            base_params = base_map[op_name]
            if not isinstance(param_overrides, Mapping):
                raise TypeError(
                    f"Curriculum override for '{op_name}' must be a mapping"
                )
            targets = name_map.get(op_name, [])
            if not targets:
                raise ValueError(
                    f"Curriculum override references op '{op_name}' with no instances"
                )
            for param_name, raw_value in param_overrides.items():
                if param_name not in base_params:
                    raise ValueError(
                        f"Curriculum override references unknown param '{op_name}.{param_name}'"
                    )
                base_numeric = base_params[param_name]
                override_numeric = NumericParam.from_raw(raw_value)
                if len(base_numeric.values) != len(override_numeric.values):
                    raise ValueError(
                        f"Curriculum override for '{op_name}.{param_name}' has dimension {len(override_numeric.values)}, "
                        f"expected {len(base_numeric.values)}"
                    )
                if _is_prob_field(param_name):
                    for v in override_numeric.values:
                        if v < 0.0 or v > 1.0:
                            raise ValueError(
                                f"Curriculum override for '{op_name}.{param_name}' must be within [0, 1]; got {v}"
                            )
                override_value = override_numeric.to_python_value()
                for op in targets:
                    current = getattr(op, param_name, None)
                    coerced = _coerce_value(current, override_value)
                    setattr(op, param_name, coerced)

    def _update_geometry_field(
        self, obj: dict[str, object], new_geom: dict[str, object]
    ) -> None:
        """
        Update object's geometry field, ensuring only ONE geometry type exists.

        Critical for downstream consistency - builders expect exactly one of:
        bbox_2d, poly, or line per object.

        Args:
            obj: Object dict to update
            new_geom: New geometry dict with bbox_2d, poly, or line field
        """
        if "bbox_2d" in new_geom:
            obj["bbox_2d"] = new_geom["bbox_2d"]
            obj.pop("poly", None)
            obj.pop("line", None)
        elif "poly" in new_geom:
            obj["poly"] = new_geom["poly"]
            obj.pop("bbox_2d", None)
            obj.pop("line", None)
        elif "line" in new_geom:
            obj["line"] = new_geom["line"]
            obj.pop("bbox_2d", None)
            obj.pop("poly", None)


__all__ = ["AugmentationPreprocessor"]
