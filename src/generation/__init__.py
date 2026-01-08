#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Centralized generation engine package."""

from .contracts import (
    ChatTemplateOptions,
    DecodeOptions,
    GenerationOptions,
    GenerationResult,
    ModelLoadConfig,
    StopOptions,
    TextGenerationRequest,
    VlmGenerationRequest,
    VlmPreprocessOptions,
    VllmEngineOptions,
)
from .engine import GenerationEngine, build_hf_engine, build_vllm_engine
from .stop_policy import QWEN_STOP_TOKENS

__all__ = [
    "ChatTemplateOptions",
    "DecodeOptions",
    "GenerationEngine",
    "GenerationOptions",
    "GenerationResult",
    "ModelLoadConfig",
    "QWEN_STOP_TOKENS",
    "StopOptions",
    "TextGenerationRequest",
    "VlmGenerationRequest",
    "VlmPreprocessOptions",
    "VllmEngineOptions",
    "build_hf_engine",
    "build_vllm_engine",
]
