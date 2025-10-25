#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-A per-image inference for GRPO group reasoning.

This module provides lightweight inference for generating Chinese single-line
summaries per image, grouped by mission and work order ID.
"""
from __future__ import annotations

from .inference import run_stage_a_inference
from .prompts import MISSION_FOCUS, SUPPORTED_MISSIONS, build_user_prompt

__all__ = [
    "run_stage_a_inference",
    "MISSION_FOCUS",
    "SUPPORTED_MISSIONS",
    "build_user_prompt",
]

