#!/usr/bin/env python3
"""
Sanitizers for text/description cleanup in the data conversion pipeline.
"""

import re
import unicodedata
from typing import Optional


def strip_occlusion_tokens(desc: Optional[str]) -> Optional[str]:
    """
    Remove tokens containing '遮挡' while preserving key=value pairs.

    - Split by ',' into tokens
    - Drop any token/value containing '遮挡'
    """
    if not desc or not isinstance(desc, str):
        return desc
    tokens = [t.strip() for t in desc.split(",") if t.strip()]
    kept = []
    for token in tokens:
        if "遮挡" in token:
            continue
        if "=" in token:
            _, value = token.split("=", 1)
            if "遮挡" in value:
                continue
        kept.append(token)
    return ",".join(kept)


def strip_annotator_notes(desc: Optional[str]) -> Optional[str]:
    """
    Remove annotator guidance tokens containing '框选范围'.
    Preserves key=value separators and drops empty tokens.
    """
    if not desc or not isinstance(desc, str):
        return desc
    tokens = [t.strip() for t in desc.split(",") if t.strip()]
    kept = []
    for token in tokens:
        if "框选范围" in token:
            continue
        if "=" in token:
            _, value = token.split("=", 1)
            if "框选范围" in value:
                continue
        kept.append(token)
    return ",".join(kept)


def sanitize_text(desc: Optional[str]) -> Optional[str]:
    """
    Normalize and sanitize description text (whitespace only).

    Rules:
    - Remove all ASCII and fullwidth spaces
    """
    if not desc or not isinstance(desc, str):
        return desc

    s = re.sub(r"\s+", "", desc.replace("\u3000", ""))

    return s


def sanitize_desc_value(value: Optional[str]) -> Optional[str]:
    """Normalize a single key=value value for structured fields (escape separators)."""
    if value is None or not isinstance(value, str):
        return value
    s = unicodedata.normalize("NFKC", value)
    s = re.sub(r"\s+", "", s.replace("\u3000", ""))
    s = s.replace(",", "，").replace("|", "｜").replace("=", "＝")
    return s


_LEARNING_NOTE_PATTERN = re.compile(
    r"(?:[，,])?这里已经帮助修改,请注意参考学习(?:[，,])?"
)

_NOISE_REMARK_TOKENS = (
    "请参考学习",
    "建议看下操作手册中螺丝、插头的标注规范",
)
_NOISE_REMARK_PATTERN = re.compile(
    "|".join(re.escape(tok) for tok in _NOISE_REMARK_TOKENS)
)


def sanitize_free_text_value(value: Optional[str]) -> Optional[str]:
    """Preserve free text (OCR/备注) while removing whitespace and known notes."""
    if value is None or not isinstance(value, str):
        return value
    s = re.sub(r"\s+", "", value.replace("\u3000", ""))
    s = _LEARNING_NOTE_PATTERN.sub(",", s)
    s = _NOISE_REMARK_PATTERN.sub("", s)
    s = re.sub(r"[，,]{2,}", ",", s)
    s = s.strip("，,")
    return s


_DISTANCE_RE = re.compile(r"(\d+)")


def sanitize_station_distance_value(value: Optional[str]) -> Optional[str]:
    """Normalize station distance to an integer token (digits only)."""
    if value is None or not isinstance(value, str):
        return None
    s = unicodedata.normalize("NFKC", value)
    s = re.sub(r"\s+", "", s.replace("\u3000", ""))
    if not s:
        return None
    match = _DISTANCE_RE.search(s)
    if match:
        return match.group(1)
    return None


def standardize_label_description(desc: Optional[str]) -> Optional[str]:
    """Legacy no-op for key=value label descriptions."""
    return desc


def remove_screw_completeness_attributes(desc: Optional[str]) -> Optional[str]:
    """
    Remove completeness attributes from screw objects only within the attribute slot.

    We expect descriptions of the form:
        螺丝、光纤插头/{type},{completeness},...
    Completeness tokens ('部分', '完整') should only be dropped from the comma
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
        tok
        for tok in tokens
        if tok and tok not in {"只显示部分", "显示完整", "部分", "完整"}
    ]
    cleaned_segment = ",".join(filtered)

    rebuilt = [parts[0]]
    rebuilt.append(cleaned_segment)
    if remainder is not None:
        rebuilt.append(remainder)

    return "/".join(rebuilt)


def remove_specific_annotation_remark(desc: Optional[str]) -> Optional[str]:
    """
    Remove specific annotation remark that contains '这里已经帮助修改,请注意参考学习'.

    This function removes the remark pattern ',备注:这里已经帮助修改,请注意参考学习'
    from the description string.
    """
    if not desc or not isinstance(desc, str):
        return desc

    s = desc.strip()
    if not s:
        return s

    pattern = r"(?:[，,]备注[:=：])?这里已经帮助修改,请注意参考学习"
    s = re.sub(pattern, "", s)

    return s


def fold_free_text_into_remark(desc: Optional[str]) -> Optional[str]:
    """Fold stray comma tokens into 备注, preserving OCR/备注 free text."""
    if not desc or not isinstance(desc, str):
        return desc

    tokens = [t.strip() for t in desc.split(",") if t.strip()]
    if not tokens:
        return desc

    pairs = []
    current_key = None
    current_value = ""
    stray_tokens = []

    for token in tokens:
        if "=" in token:
            if current_key is not None:
                pairs.append((current_key, current_value))
            key, value = token.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                stray_tokens.append(token)
                current_key = None
                current_value = ""
                continue
            current_key = key
            current_value = value
        else:
            if current_key in {"备注", "文本"}:
                current_value = f"{current_value},{token}" if current_value else token
            else:
                stray_tokens.append(token)

    if current_key is not None:
        pairs.append((current_key, current_value))

    if stray_tokens:
        remark_value = ",".join(stray_tokens)
        appended = False
        for idx in range(len(pairs) - 1, -1, -1):
            if pairs[idx][0] == "备注":
                existing = pairs[idx][1] or ""
                pairs[idx] = (
                    pairs[idx][0],
                    f"{existing},{remark_value}" if existing else remark_value,
                )
                appended = True
                break
        if not appended:
            pairs.append(("备注", remark_value))

    cleaned = []
    for key, value in pairs:
        if value is None:
            continue
        if key in {"备注", "文本"}:
            value = sanitize_free_text_value(value)
            if not value:
                continue
        cleaned.append((key, value))

    return ",".join(f"{k}={v}" for k, v in cleaned if v)
