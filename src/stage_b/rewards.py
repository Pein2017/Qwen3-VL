#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reward functions for Stage-B GRPO training.

Implements binary label reward and two-line format reward for group-level judgment.
Compatible with ms-swift GRPO reward function interface.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Valid verdicts (first line options)
VALID_VERDICTS = {"通过", "不通过"}


def extract_verdict_from_response(response: str) -> Optional[str]:
    """Extract verdict from first line of model response.
    
    Args:
        response: Full model output text
        
    Returns:
        Extracted verdict ("通过" or "不通过"), or None if invalid
        
    Examples:
        >>> extract_verdict_from_response("通过\\n理由: 符合要求")
        '通过'
        >>> extract_verdict_from_response("不通过\\n理由: 缺少螺丝")
        '不通过'
        >>> extract_verdict_from_response("可能通过\\n...")
        None
    """
    lines = response.strip().split("\n")
    if not lines:
        return None
    
    first_line = lines[0].strip()
    
    # Exact match only (no extra tokens)
    if first_line in VALID_VERDICTS:
        return first_line
    
    return None


def label_reward(
    responses: List[str],
    row: Dict[str, Any],
    **kwargs
) -> List[float]:
    """Binary reward for verdict matching ground-truth label.
    
    Args:
        responses: List of model responses (length = num_generations)
        row: Dataset row containing 'group_label' field ("通过" | "不通过")
        **kwargs: Additional args (ignored)
        
    Returns:
        List of rewards (1.0 if verdict matches label, 0.0 otherwise)
        
    Behavior:
        - Extract verdict from first line
        - Compare with row['group_label']
        - Return 1.0 for exact match, 0.0 for mismatch or invalid format
    """
    gt_label = row.get("group_label")
    if not gt_label or gt_label not in VALID_VERDICTS:
        # Invalid GT label → return 0.0 for all responses
        return [0.0] * len(responses)
    
    rewards = []
    for response in responses:
        verdict = extract_verdict_from_response(response)
        if verdict == gt_label:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards


def format_reward(
    responses: List[str],
    row: Dict[str, Any],
    **kwargs
) -> List[float]:
    """Reward for two-line format compliance.
    
    Args:
        responses: List of model responses (length = num_generations)
        row: Dataset row (not used, but required by interface)
        **kwargs: Additional args (ignored)
        
    Returns:
        List of rewards (1.0 if valid format, 0.0 otherwise)
        
    Format requirements:
        - Exactly 2 lines (允许末尾空白字符)
        - Line 1: exactly "通过" or "不通过" (no extra tokens)
        - Line 2: starts with "理由:" or "理由：" and has content after it
        
    Examples:
        >>> format_reward(["通过\\n理由: 符合要求"], {})
        [1.0]
        >>> format_reward(["通过\\n\\n理由: 符合要求"], {})
        [0.0]
        >>> format_reward(["通过"], {})
        [0.0]
    """
    rewards = []
    
    for response in responses:
        # Split and strip trailing whitespace
        lines = response.rstrip().split("\n")
        
        # Must have exactly 2 lines
        if len(lines) != 2:
            rewards.append(0.0)
            continue
        
        first_line = lines[0].strip()
        second_line = lines[1].strip()
        
        # Line 1: exact match required
        if first_line not in VALID_VERDICTS:
            rewards.append(0.0)
            continue
        
        # Line 2: must start with "理由:" or "理由：" (full-width colon also allowed)
        # and have non-empty content after it
        if not re.match(r"^理由[:：]\s*.+", second_line):
            rewards.append(0.0)
            continue
        
        # All checks passed
        rewards.append(1.0)
    
    return rewards


def consistency_reward(
    responses: List[str],
    row: Dict[str, Any],
    **kwargs
) -> List[float]:
    """Placeholder for consistency reward (v2).
    
    This reward will check that the reasoning in line 2 aligns with
    Stage-A summaries and doesn't contradict or hallucinate.
    
    Args:
        responses: List of model responses
        row: Dataset row with 'stage_a_summaries' field
        **kwargs: Additional args
        
    Returns:
        List of zeros (not implemented yet)
        
    Notes:
        Deferred to v2. Implementation will require:
        - Parse reasoning from line 2
        - Check for references to 图片_i
        - Verify claims against stage_a_summaries
        - Penalize hallucinations or contradictions
    """
    # Placeholder: return neutral reward
    return [0.0] * len(responses)


# Reward function registry (for ms-swift integration)
REWARD_FUNCTIONS = {
    "label": label_reward,
    "format": format_reward,
    "consistency": consistency_reward,
}


def get_reward_function(name: str):
    """Get reward function by name.
    
    Args:
        name: Reward function name ("label", "format", "consistency")
        
    Returns:
        Reward function callable
        
    Raises:
        KeyError: If reward function not found
    """
    if name not in REWARD_FUNCTIONS:
        raise KeyError(
            f"Unknown reward function '{name}'. Available: {list(REWARD_FUNCTIONS.keys())}"
        )
    return REWARD_FUNCTIONS[name]

