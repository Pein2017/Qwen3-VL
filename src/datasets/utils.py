"""Shared utilities for dataset operations"""
import json
from typing import Any, Dict, List, Tuple


def load_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load records from a JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_object_points(obj: Dict[str, Any]) -> Tuple[str, List[float]]:
    """Extract geometry type and points from an object.
    
    Args:
        obj: Object dictionary containing geometry (bbox_2d, quad, or line)
        
    Returns:
        Tuple of (geometry_type, points_list)
    """
    if "bbox_2d" in obj:
        return "bbox_2d", list(map(float, obj["bbox_2d"]))
    if "quad" in obj:
        return "quad", list(map(float, obj["quad"]))
    if "line" in obj:
        return "line", list(map(float, obj["line"]))
    return "", []


def extract_geometry(obj: Dict[str, Any]) -> Dict[str, List[float]]:
    """Extract geometry dictionary from object.
    
    Useful for augmentation and processing pipelines.
    
    Args:
        obj: Object dictionary
        
    Returns:
        Dictionary with geometry key and points
    """
    geom: Dict[str, List[float]] = {}
    if obj.get("bbox_2d") is not None:
        geom["bbox_2d"] = obj["bbox_2d"]
    if obj.get("quad") is not None:
        geom["quad"] = obj["quad"]
    if obj.get("line") is not None:
        geom["line"] = obj["line"]
    return geom


def is_same_record(record_a: Dict[str, Any], record_b: Dict[str, Any]) -> bool:
    """Check if two records are the same (identity check).
    
    Args:
        record_a: First record
        record_b: Second record
        
    Returns:
        True if records are the same object
    """
    return record_a is record_b


__all__ = [
    "load_jsonl",
    "extract_object_points",
    "extract_geometry",
    "is_same_record",
]

