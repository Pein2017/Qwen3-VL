#!/usr/bin/env python3
"""
Sanitizers for text/description cleanup in the data conversion pipeline.
"""

import re
import unicodedata
from typing import Optional


def strip_occlusion_tokens(desc: Optional[str]) -> Optional[str]:
    """
    Remove tokens containing '遮挡' per level/token while preserving separators.

    - Split by '/' into levels, by ',' within levels
    - Drop any token containing '遮挡'
    - Rejoin; drop empty levels
    """
    if not desc or not isinstance(desc, str):
        return desc
    levels = [lvl.strip() for lvl in desc.split("/")]
    kept_levels = []
    for lvl in levels:
        if not lvl:
            continue
        tokens = [t.strip() for t in lvl.split(",")]
        kept_tokens = [t for t in tokens if t and ("遮挡" not in t)]
        if kept_tokens:
            kept_levels.append(",".join(kept_tokens))
    return "/".join(kept_levels)


def strip_annotator_notes(desc: Optional[str]) -> Optional[str]:
    """
    Remove annotator guidance tokens containing '框选范围'.
    Preserves separators and drops empty levels.
    """
    if not desc or not isinstance(desc, str):
        return desc
    levels = [lvl.strip() for lvl in desc.split("/")]
    kept_levels = []
    for lvl in levels:
        if not lvl:
            continue
        tokens = [t.strip() for t in lvl.split(",")]
        kept_tokens = [t for t in tokens if t and ("框选范围" not in t)]
        if kept_tokens:
            kept_levels.append(",".join(kept_tokens))
    return "/".join(kept_levels)


def sanitize_text(desc: Optional[str]) -> Optional[str]:
    """
    Normalize and sanitize description text.

    Rules:
    - Remove all ASCII and fullwidth spaces
    - Collapse repeated hyphens: "--" → "-" (also for 3+)
    - Remove trailing hyphens at token boundaries (end of whole string and after '/',' ,')
    - Convert fullwidth digits to ASCII (e.g., ９００ → 900)
    - Normalize circled/parenthesized digits (e.g., ①②… → 1 2 …)
    - Normalize dash-like characters to ASCII hyphen '-'
    """
    if not desc or not isinstance(desc, str):
        return desc

    s = desc

    # Unicode compatibility normalization (handles many fullwidth forms)
    s = unicodedata.normalize("NFKC", s)

    # Normalize dash-like characters to '-' (covers hyphen/minus/en/em dashes, etc.)
    dash_like_chars = "‐‑‒–—−﹣－"
    s = s.translate({ord(c): "-" for c in dash_like_chars})

    # Convert circled/parenthesized digits to plain ASCII digits
    circled_map = {ord("\u24EA"): "0"}  # ⓪
    for i in range(1, 21):
        circled_map[ord(chr(0x2460 + i - 1))] = str(i)  # ①..⑳
        circled_map[ord(chr(0x2474 + i - 1))] = str(i)  # ⑴..⒇
    for i in range(1, 11):
        circled_map[ord(chr(0x2776 + i - 1))] = str(i)  # ❶..❿
    s = s.translate(circled_map)

    # Remove ASCII space and fullwidth space
    s = s.replace("\u3000", "").replace(" ", "")

    # Collapse multiple hyphens
    s = re.sub(r"-{2,}", "-", s)

    # Remove dangling trailing '-' at end and at token boundaries
    # First handle per '/'-separated tokens
    parts = s.split("/")
    parts = [p.rstrip("-") for p in parts]
    s = "/".join(parts)

    # Also handle any ','-separated remnants just in case
    comma_parts = [p.rstrip("-") for p in s.split(",")]
    s = ",".join(comma_parts)

    # Finally, strip any trailing hyphen at end of string
    s = s.rstrip("-")

    return s


def standardize_label_description(desc: Optional[str]) -> Optional[str]:
    """
    Standardize descriptions for 标签 objects.

    - If description is exactly '标签' or '标签/' (empty content), normalize to '标签/无法识别'
    - If second-level content is one of {'空格', '看不清', '、'} (non-informative), normalize to '标签/无法识别'
    - Otherwise, return the original description unchanged
    """
    if not desc or not isinstance(desc, str):
        return desc

    s = desc.strip()
    if not s:
        return s

    parts = [p.strip() for p in s.split("/")]
    if not parts:
        return s

    # Only standardize 标签 objects
    if parts[0] != "标签":
        return s

    # Determine property/content part (second level)
    prop = parts[1] if len(parts) >= 2 else ""

    # Treat explicitly non-informative values as unrecognizable
    invalid_tokens = {"空格", "看不清", "、"}

    if (prop is None) or (prop == "") or (prop in invalid_tokens):
        return "标签/无法识别"

    return s


def remove_screw_completeness_attributes(desc: Optional[str]) -> Optional[str]:
    """
    Remove completeness attributes from screw objects only within the attribute slot.

    We expect descriptions of the form:
        螺丝、光纤插头/{type},{completeness},...
    Completeness tokens ('只显示部分', '显示完整') should only be dropped from the comma
    list immediately after the object type. Text appearing in later sections such as
    '备注' must remain untouched.
    """
    if not desc or not isinstance(desc, str):
        return desc

    s = desc.strip()
    if not s:
        return s

    parts = s.split("/", 2)
    if not parts or parts[0].strip() != "螺丝、光纤插头":
        return s

    if len(parts) == 1:
        return s  # No attributes to clean

    attribute_segment = parts[1]
    remainder = parts[2] if len(parts) > 2 else None

    tokens = [tok.strip() for tok in attribute_segment.split(",")]
    filtered = [
        tok for tok in tokens if tok and tok not in {"只显示部分", "显示完整"}
    ]
    cleaned_segment = ",".join(filtered)

    rebuilt = [parts[0]]
    rebuilt.append(cleaned_segment)
    if remainder is not None:
        rebuilt.append(remainder)

    return "/".join(rebuilt)
