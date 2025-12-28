#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verdict normalization utilities for Stage-B."""

from __future__ import annotations

from ..types import GroupLabel


def normalize_verdict(verdict: str | GroupLabel | None) -> GroupLabel | None:
    """
    Normalize verdict strings to canonical GroupLabel format.

    Handles both English ("pass", "fail") and Chinese ("通过", "不通过") variants.
    Returns None if the verdict cannot be normalized.

    Args:
        verdict: Verdict string to normalize, or already-normalized GroupLabel

    Returns:
        Normalized GroupLabel ("pass" or "fail") or None if unrecognized
    """
    if verdict is None:
        return None

    # Already normalized
    if verdict in ("pass", "fail"):
        return verdict  # type: ignore[return-value]

    # Normalize string input
    cleaned = str(verdict).strip().replace(" ", "").lower()

    # Chinese variants
    if cleaned in {"通过", "通过。"}:
        return "pass"
    if cleaned in {"不通过", "未通过", "不通过。"}:
        return "fail"
    # Third-state / pending phrases are forbidden in Stage-B inference outputs; treat as unrecognized.
    if cleaned in {"需复核", "需要复核", "无法判断", "无法判定", "待复核"}:
        return None
    if cleaned in {"通过需复核", "通过需要复核", "通过需要复核。", "通过需复核。"}:
        return None

    # English variants
    if cleaned in {"pass", "pass."}:
        return "pass"
    if cleaned in {"fail", "fail."}:
        return "fail"

    return None


__all__ = ["normalize_verdict"]
