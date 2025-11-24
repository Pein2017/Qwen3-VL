"""Augmentation preprocessor - decoupled from dataset logic"""

import random
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from .base import BasePreprocessor
from ..utils import extract_geometry
from ...utils.logger import get_logger
from ..contracts import ConversationRecord, AugmentationTelemetry


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
        augmenter: Optional[Any] = None,
        rng: Optional[random.Random] = None,
        bypass_prob: float = 0.0,
        curriculum_state: Optional[MutableMapping[str, Any]] = None,
        **kwargs: Any,
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

    def preprocess(self, row: ConversationRecord) -> Optional[ConversationRecord]:
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
            from ..augmentation import (
                ops as _builtin_ops,
            )  # ensure registration side-effects
            from ..augmentation.registry import get as _get
        except Exception:
            Compose = None  # type: ignore
            _get = None  # type: ignore

        images = rec.get("images") or []
        objs = rec.get("objects") or []

        # Extract geometries
        per_obj_geoms: List[Dict[str, Any]] = []
        for obj in objs:
            g = extract_geometry(obj)
            if g:
                per_obj_geoms.append(g)

        # Apply augmentations using unified pipeline API only
        if Compose is None or not isinstance(self.augmenter, Compose):
            raise TypeError(
                "AugmentationPreprocessor requires 'augmenter' to be a Compose pipeline."
            )

        pipeline = self.augmenter
        images_bytes, per_obj_geoms_new = apply_augmentations(
            images, per_obj_geoms, pipeline, rng=self.rng
        )

        telemetry: Optional[AugmentationTelemetry] = getattr(
            pipeline, "last_summary", None
        )

        if telemetry is None or not telemetry.kept_indices:
            # No crop applied (or crop was skipped) - update geometries only
            j = 0
            for i, obj in enumerate(objs):
                if (
                    obj.get("bbox_2d") is not None
                    or obj.get("poly") is not None
                    or obj.get("line") is not None
                ):
                    g = per_obj_geoms_new[j]
                    j += 1
                    # Update with new geometry, clear others
                    self._update_geometry_field(obj, g)
        else:
            # Crop was applied - filter objects and update completeness
            filtered_objects: List[Dict[str, Any]] = []
            kept_indices = list(telemetry.kept_indices)
            coverages = list(telemetry.coverages)

            # Get completeness threshold from the pipeline's last crop operator
            # (For now, we assume the crop operator is accessible; alternatively,
            # we could store completeness_threshold as pipeline metadata)
            completeness_threshold = 0.95  # Default
            # Try to get from pipeline's crop operators
            for op in pipeline.ops if hasattr(pipeline, "ops") else []:
                if hasattr(op, "completeness_threshold"):
                    completeness_threshold = op.completeness_threshold
                    break

            # Build mapping from original obj indices to new geometries
            obj_idx_with_geom = []
            for i, obj in enumerate(objs):
                if (
                    obj.get("bbox_2d") is not None
                    or obj.get("poly") is not None
                    or obj.get("line") is not None
                ):
                    obj_idx_with_geom.append(i)

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
                    desc = obj.get("desc", "")
                    if "显示完整" in desc:
                        obj["desc"] = desc.replace("显示完整", "只显示部分")
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
                            if value == "显示完整":
                                attrs[completeness_key] = "只显示部分"
                                structured_updates += 1
                        else:
                            logger = get_logger("augmentation.preprocessor")
                            logger.debug(
                                "Completeness metadata missing expected key on object; desc_updated=%s",
                                "显示完整" in desc,
                            )

                filtered_objects.append(obj)

            # Replace objects list with filtered objects
            rec["objects"] = filtered_objects

            # Log crop filtering results (debug level)
            logger = get_logger("augmentation.preprocessor")
            logger.debug(
                f"Crop filter: {len(objs)} → {len(filtered_objects)} objects "
                f"({completeness_updates} desc updates, {structured_updates} attribute updates)"
            )

        rec["images"] = images_bytes

        # Update record width/height to reflect any resize/pad ops in augmentation
        try:
            if images_bytes:
                import io
                from PIL import Image  # type: ignore

                b0 = images_bytes[0].get("bytes")
                if isinstance(b0, (bytes, bytearray)):
                    with Image.open(io.BytesIO(b0)) as im0:
                        im0 = im0.convert("RGB")
                        rec["width"] = int(im0.width)
                        rec["height"] = int(im0.height)
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
            step = int(step) if step is not None else 0
        except (TypeError, ValueError):
            step = 0
        if self._curriculum_last_step == step:
            return
        bypass = state.get("bypass_prob")
        if bypass is not None:
            try:
                self.bypass_prob = float(bypass)
            except (TypeError, ValueError):
                pass
        ops = state.get("ops") or {}
        if isinstance(ops, Mapping):
            self._apply_curriculum_overrides(ops)
        self._curriculum_last_step = step

    def _apply_curriculum_overrides(
        self, overrides: Mapping[str, Mapping[str, Any]]
    ) -> None:
        if self.augmenter is None:
            return

        def _coerce_value(current: Any, new_value: Any) -> Any:
            """Preserve operator parameter types when applying overrides."""
            if current is None:
                return new_value
            if isinstance(current, bool):
                return bool(new_value)
            if isinstance(current, int):
                try:
                    return int(round(new_value))
                except Exception:
                    return current
            if isinstance(current, float):
                try:
                    return float(new_value)
                except Exception:
                    return current
            if isinstance(current, tuple):
                if isinstance(new_value, (list, tuple)):
                    coerced = []
                    for i, item in enumerate(new_value):
                        base = current[i] if i < len(current) else (current[-1] if current else None)
                        coerced.append(_coerce_value(base, item) if base is not None else item)
                    return tuple(coerced)
                return current
            if isinstance(current, list):
                if isinstance(new_value, (list, tuple)):
                    coerced = []
                    for i, item in enumerate(new_value):
                        base = current[i] if i < len(current) else None
                        coerced.append(_coerce_value(base, item) if base is not None else item)
                    return coerced
            return new_value

        for op in getattr(self.augmenter, "ops", []):
            name = getattr(op, "_aug_name", None)
            if not name:
                continue
            params = overrides.get(name)
            if not isinstance(params, Mapping):
                continue
            for param_name, value in params.items():
                try:
                    current = getattr(op, param_name, None)
                    coerced = _coerce_value(current, value)
                    setattr(op, param_name, coerced)
                except Exception:
                    continue

    def _update_geometry_field(
        self, obj: Dict[str, Any], new_geom: Dict[str, Any]
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
