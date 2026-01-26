#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared preprocessing helpers for generation backends."""

from __future__ import annotations

from collections.abc import Mapping

from src.utils import get_logger

from .contracts import VlmPreprocessOptions

logger = get_logger(__name__)


def normalize_tokenizer(tokenizer: object, *, options: VlmPreprocessOptions) -> None:
    """Normalize tokenizer padding/truncation and pad-token fallback."""

    if tokenizer is None:
        return
    try:
        padding_side = getattr(options, "padding_side", None)
        truncation_side = getattr(options, "truncation_side", None)
        if padding_side is not None:
            setattr(tokenizer, "padding_side", padding_side)
        if truncation_side is not None:
            setattr(tokenizer, "truncation_side", truncation_side)
        logger.info(
            "Tokenizer padding_side=%s truncation_side=%s",
            getattr(tokenizer, "padding_side", None),
            getattr(tokenizer, "truncation_side", None),
        )
    except Exception:
        logger.warning(
            "Failed to set tokenizer padding/truncation side", exc_info=False
        )

    if options.pad_token_fallback:
        try:
            pad_token = getattr(tokenizer, "pad_token", None)
            eos_token = getattr(tokenizer, "eos_token", None)
            if pad_token is None and eos_token is not None:
                setattr(tokenizer, "pad_token", eos_token)
        except Exception:
            logger.warning("Failed to set pad_token fallback", exc_info=False)


def configure_vlm_processor(
    processor: object, *, options: VlmPreprocessOptions
) -> None:
    """Normalize VLM processor pixel budgets and resizing options."""

    if processor is None:
        return
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        logger.warning("Image processor unavailable; skipping pixel budget config")
        return

    if options.do_resize is not None:
        try:
            setattr(ip, "do_resize", bool(options.do_resize))
        except Exception:
            logger.warning("Failed to set image_processor.do_resize", exc_info=False)

    if options.max_pixels is None and options.min_pixels is None:
        return

    try:
        if not hasattr(ip, "size") or not isinstance(ip.size, Mapping):
            ip.size = {}
        size_map = ip.size

        min_pix = options.min_pixels
        if min_pix is None:
            min_pix = getattr(ip, "min_pixels", None)
        if min_pix is None:
            min_pix = size_map.get("shortest_edge", 56 * 56)
        max_pix = options.max_pixels
        if max_pix is None:
            max_pix = getattr(ip, "max_pixels", None)

        if min_pix is not None:
            size_map["shortest_edge"] = int(min_pix)
        if max_pix is not None:
            size_map["longest_edge"] = int(max_pix)
        try:
            if min_pix is not None:
                ip.min_pixels = int(min_pix)
        except Exception:
            pass
        try:
            if max_pix is not None:
                ip.max_pixels = int(max_pix)
        except Exception:
            pass

        logger.info(
            "Set pixel budget: min_pixels=%s max_pixels=%s (size=%s)",
            size_map.get("shortest_edge"),
            size_map.get("longest_edge"),
            size_map,
        )
    except Exception:
        logger.warning(
            "Failed to set pixel budget on image_processor; continuing with defaults",
            exc_info=False,
        )

    try:
        logger.info(
            "ImageProcessor settings: do_resize=%s, patch_size=%s, merge_size=%s, min_pixels=%s, max_pixels=%s, size=%s",
            getattr(ip, "do_resize", None),
            getattr(ip, "patch_size", None),
            getattr(ip, "merge_size", None),
            getattr(ip, "min_pixels", None),
            getattr(ip, "max_pixels", None),
            getattr(ip, "size", None),
        )
    except Exception:
        pass


def build_image_kwargs(processor: object) -> dict[str, int]:
    """Best-effort image kwargs for VLM processor calls."""

    kwargs: dict[str, int] = {}
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return kwargs
    try:
        if isinstance(getattr(ip, "size", None), Mapping):
            min_pix = ip.size.get("shortest_edge")
            max_pix = ip.size.get("longest_edge")
        else:
            min_pix = getattr(ip, "min_pixels", None)
            max_pix = getattr(ip, "max_pixels", None)
        if min_pix is not None:
            kwargs["min_pixels"] = int(min_pix)
        if max_pix is not None:
            kwargs["max_pixels"] = int(max_pix)
    except Exception:
        return {}
    return kwargs


__all__ = ["build_image_kwargs", "configure_vlm_processor", "normalize_tokenizer"]
