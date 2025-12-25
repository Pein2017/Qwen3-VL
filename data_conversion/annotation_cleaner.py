#!/usr/bin/env python3
"""
Shared helpers for cleaning raw annotation JSON structures.

Used by both the standalone cleaning script and FormatConverter to ensure we
only retain the fields needed by the pipeline while keeping language-specific
content under user control.
"""

from __future__ import annotations

from typing import Any, Dict, List


ESSENTIAL_KEYS: List[str] = ["info", "tagInfo", "version"]


def _normalize_geometry(geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of geometry with legacy types normalized."""
    if not geometry:
        return geometry

    if geometry.get("type") == "Square":
        geometry = geometry.copy()
        geometry["type"] = "Quad"

    return geometry


def clean_annotation_content(
    data: Dict[str, Any], lang: str = "both"
) -> Dict[str, Any]:
    """
    Clean annotation content preserving core metadata and requested languages.

    Args:
        data: Raw annotation JSON data.
        lang: One of {"zh", "en", "both"} describing which language blocks to keep.

    Returns:
        Cleaned annotation data with consistent geometry types.
    """
    normalized_lang = lang.lower()
    if normalized_lang not in {"zh", "en", "both"}:
        raise ValueError(
            f"Unsupported lang '{lang}'. Expected one of: zh, en, both"
        )

    cleaned: Dict[str, Any] = {}
    for key in ESSENTIAL_KEYS:
        if key in data:
            cleaned[key] = data[key]

    cleaned_features = []
    mark_result = data.get("markResult", {})
    for feature in mark_result.get("features", []) or []:
        properties: Dict[str, Any] = {}
        original_properties: Dict[str, Any] = feature.get("properties", {}) or {}

        if normalized_lang in {"zh", "both"}:
            properties["contentZh"] = original_properties.get("contentZh", {})
        if normalized_lang in {"en", "both"}:
            properties["content"] = original_properties.get("content", {})

        cleaned_features.append(
            {
                "type": feature.get("type", "Feature"),
                "geometry": _normalize_geometry(feature.get("geometry", {})),
                "properties": properties,
            }
        )

    if "markResult" in data:
        cleaned["markResult"] = {
            "features": cleaned_features,
            "type": mark_result.get("type", "FeatureCollection"),
        }
        for key, value in mark_result.items():
            if key not in {"features", "type"}:
                cleaned["markResult"][key] = value

    return cleaned


__all__ = ["clean_annotation_content"]
