"""Augmentation preprocessor - decoupled from dataset logic"""
import random
from typing import Any, Dict, List, Optional

from .base import BasePreprocessor
from ..utils import extract_geometry


class AugmentationPreprocessor(BasePreprocessor):
    """Preprocessor that applies augmentations to records.
    
    Decouples augmentation logic from dataset __getitem__, making it:
    - Reusable across different datasets
    - Testable independently
    - Composable with other preprocessors
    """
    
    def __init__(self, *, augmenter: Optional[Any] = None, rng: Optional[random.Random] = None, **kwargs):
        """Initialize augmentation preprocessor.
        
        Args:
            augmenter: AugmentationConfig instance
            rng: Random number generator for reproducibility
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.augmenter = augmenter
        self.rng = rng if rng is not None else random.Random()
    
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply augmentations to a record.
        
        Args:
            row: Input record with images and objects
            
        Returns:
            Augmented record
        """
        if self.augmenter is None:
            return row
        
        try:
            row = self._augment_record(row)
        except Exception:
            # If augmentation fails, return original record
            pass
        
        return row
    
    def _augment_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentation to a single record.
        
        Args:
            rec: Record to augment
            
        Returns:
            Augmented record
        """
        from ..augment import apply_augmentations
        
        images = rec.get("images") or []
        objs = rec.get("objects") or []
        
        # Extract geometries
        per_obj_geoms: List[Dict[str, Any]] = []
        for obj in objs:
            g = extract_geometry(obj)
            if g:
                per_obj_geoms.append(g)
        
        # Apply augmentations
        images_bytes, _objects_dummy, per_obj_geoms_new = apply_augmentations(
            images, {}, per_obj_geoms, cfg=self.augmenter, rng=self.rng
        )
        
        # Update geometries in objects
        j = 0
        for i, obj in enumerate(objs):
            if obj.get("bbox_2d") is not None or obj.get("quad") is not None or obj.get("line") is not None:
                g = per_obj_geoms_new[j]
                j += 1
                # Update with new geometry, clear others
                if "bbox_2d" in g:
                    obj["bbox_2d"] = g["bbox_2d"]
                    obj.pop("quad", None)
                    obj.pop("line", None)
                elif "quad" in g:
                    obj["quad"] = g["quad"]
                    obj.pop("bbox_2d", None)
                    obj.pop("line", None)
                elif "line" in g:
                    obj["line"] = g["line"]
                    obj.pop("bbox_2d", None)
                    obj.pop("quad", None)
        
        rec["images"] = images_bytes
        return rec


__all__ = ["AugmentationPreprocessor"]

