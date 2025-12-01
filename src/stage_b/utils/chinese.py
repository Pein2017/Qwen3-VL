#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chinese text conversion utilities for Stage-B."""

from __future__ import annotations

import re

try:
    import zhconv
except ImportError:
    zhconv = None  # type: ignore


def to_simplified(text: str) -> str:
    """Convert traditional Chinese to simplified Chinese.

    Args:
        text: Input text that may contain traditional Chinese characters.

    Returns:
        Text with traditional Chinese characters converted to simplified.
        If zhconv is not available, returns the original text unchanged.
    """
    if zhconv is None:
        return text
    return zhconv.convert(text, "zh-cn")


def normalize_spaces(text: str) -> str:
    """Normalize spaces in Chinese text.

    Removes extra spaces, especially those between Chinese characters.
    Keeps single spaces around punctuation and between phrases.
    Handles mixed Chinese/English text.

    Args:
        text: Input text that may contain extra spaces.

    Returns:
        Text with normalized spaces.
    """
    if not text:
        return text

    # First, replace multiple consecutive spaces with a single space
    text = re.sub(r" +", " ", text)

    # Remove spaces between Chinese characters (CJK Unified Ideographs)
    # Pattern: Chinese char + space + Chinese char
    text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)

    # Remove spaces between Chinese char and Chinese punctuation
    text = re.sub(r"([\u4e00-\u9fff])\s+([，。；：！？、])", r"\1\2", text)
    text = re.sub(r"([，。；：！？、])\s+([\u4e00-\u9fff])", r"\1\2", text)

    # Remove spaces around English words that are between Chinese characters
    # Pattern: Chinese char + space + English word + space + Chinese char
    # This handles cases like "挡 wind 板" -> "挡wind板"
    text = re.sub(
        r"([\u4e00-\u9fff])\s+([a-zA-Z]+)\s+([\u4e00-\u9fff])", r"\1\2\3", text
    )

    # Remove spaces between English letters/numbers when followed/preceded by Chinese
    # Pattern: Chinese + English letters + space + English letters/numbers + Chinese
    # This handles cases like "BB U设备" -> "BBU设备"
    text = re.sub(
        r"([\u4e00-\u9fff])([A-Z0-9]+)\s+([A-Z0-9]+)([\u4e00-\u9fff])",
        r"\1\2\3\4",
        text,
    )
    # Also handle when English is at start: "BB U设备" -> "BBU设备"
    text = re.sub(r"^([A-Z0-9]+)\s+([A-Z0-9]+)([\u4e00-\u9fff])", r"\1\2\3", text)
    # And when English is in middle: "设备BB U设备" -> "设备BBU设备"
    text = re.sub(
        r"([\u4e00-\u9fff])([A-Z0-9]+)\s+([A-Z0-9]+)([\u4e00-\u9fff])",
        r"\1\2\3\4",
        text,
    )

    # Remove leading/trailing spaces
    text = text.strip()

    # Normalize multiple spaces again (in case the above created new ones)
    text = re.sub(r" +", " ", text)

    return text
