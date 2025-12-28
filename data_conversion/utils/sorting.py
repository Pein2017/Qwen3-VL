#!/usr/bin/env python3
"""
Sorting utilities for objects in the data conversion pipeline.
"""

from typing import Any, Dict, List, Tuple


def _first_xy(obj: Dict[str, Any]) -> Tuple[int, int]:
    """
    Get sorting reference point according to prompt specification:
    - bbox_2d: top-left corner (x1, y1)
    - poly: first vertex (x1, y1)
    - line: leftmost point (min X, then min Y if tie)
    """
    if "bbox_2d" in obj:
        # Prompt: "使用左上角坐标 (x1, y1)"
        return obj["bbox_2d"][0], obj["bbox_2d"][1]
    if "poly" in obj:
        # Prompt: "使用第一个顶点 (x1, y1)"
        return obj["poly"][0], obj["poly"][1]
    if "line" in obj:
        # Prompt: "使用最左端点（X 坐标最小的点）作为排序参考；若多个点的 X 坐标相同，则取其中 Y 坐标最小的点"
        coords = obj["line"]
        # Extract all points as (x, y) pairs
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        # Find leftmost point (min X, then min Y if tie)
        leftmost = min(points, key=lambda p: (p[0], p[1]))
        return leftmost[0], leftmost[1]
    return 0, 0


def sort_objects_tlbr(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort objects top-to-bottom, then left-to-right.
    
    Matches prompt specification exactly:
    - First by Y coordinate (top to bottom, smallest to largest)
    - Then by X coordinate (left to right, smallest to largest)
    - Uses reference points per geometry type as specified in prompts.py
    """
    return sorted(objects, key=lambda o: (_first_xy(o)[1], _first_xy(o)[0]))
