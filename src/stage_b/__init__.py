#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-B: Group-level judgment inference and GRPO training."""

from .dataset import load_stage_a_for_grpo, prepare_grpo_batch
from .prompts import MISSION_FOCUS_MAP, build_stage_b_messages
from .rewards import label_reward, format_reward, get_reward_function

__all__ = [
    "MISSION_FOCUS_MAP",
    "build_stage_b_messages",
    "load_stage_a_for_grpo",
    "prepare_grpo_batch",
    "label_reward",
    "format_reward",
    "get_reward_function",
]

