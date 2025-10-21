#!/usr/bin/env python3
"""
Sorting utilities for objects in the data conversion pipeline.
"""

from typing import Dict, List, Tuple


def _first_xy(obj: Dict) -> Tuple[int, int]:
    if "bbox_2d" in obj:
        return obj["bbox_2d"][0], obj["bbox_2d"][1]
    if "quad" in obj:
        return obj["quad"][0], obj["quad"][1]
    if "line" in obj:
        return obj["line"][0], obj["line"][1]
    return 0, 0


def sort_objects_tlbr(objects: List[Dict]) -> List[Dict]:
    """Sort top-to-bottom, then left-to-right by first coordinate pair."""
    return sorted(objects, key=lambda o: (_first_xy(o)[1], _first_xy(o)[0]))
