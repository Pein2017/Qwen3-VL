#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Stage-B reward functions."""
from __future__ import annotations

from .rewards import label_reward, format_reward


def test_label_reward():
    """Test label reward function."""
    print("Testing label_reward...")
    
    # Test case 1: Correct verdict
    responses = ["通过\n理由: 符合要求"]
    row = {"group_label": "通过"}
    rewards = label_reward(responses, row)
    assert rewards == [1.0], f"Expected [1.0], got {rewards}"
    print("  ✓ Correct verdict → 1.0")
    
    # Test case 2: Wrong verdict
    responses = ["不通过\n理由: 缺少螺丝"]
    row = {"group_label": "通过"}
    rewards = label_reward(responses, row)
    assert rewards == [0.0], f"Expected [0.0], got {rewards}"
    print("  ✓ Wrong verdict → 0.0")
    
    # Test case 3: Invalid format (can't extract verdict)
    responses = ["可能通过\n理由: 不确定"]
    row = {"group_label": "通过"}
    rewards = label_reward(responses, row)
    assert rewards == [0.0], f"Expected [0.0], got {rewards}"
    print("  ✓ Invalid verdict → 0.0")
    
    # Test case 4: Multiple responses
    responses = ["通过\n理由: OK", "不通过\n理由: NO", "通过\n理由: YES"]
    row = {"group_label": "通过"}
    rewards = label_reward(responses, row)
    assert rewards == [1.0, 0.0, 1.0], f"Expected [1.0, 0.0, 1.0], got {rewards}"
    print("  ✓ Multiple responses → [1.0, 0.0, 1.0]")


def test_format_reward():
    """Test format reward function."""
    print("\nTesting format_reward...")
    
    # Test case 1: Valid format
    responses = ["通过\n理由: 符合要求"]
    row = {}
    rewards = format_reward(responses, row)
    assert rewards == [1.0], f"Expected [1.0], got {rewards}"
    print("  ✓ Valid two-line format → 1.0")
    
    # Test case 2: Three lines (invalid)
    responses = ["通过\n\n理由: 符合要求"]
    row = {}
    rewards = format_reward(responses, row)
    assert rewards == [0.0], f"Expected [0.0], got {rewards}"
    print("  ✓ Three lines → 0.0")
    
    # Test case 3: Only one line
    responses = ["通过"]
    row = {}
    rewards = format_reward(responses, row)
    assert rewards == [0.0], f"Expected [0.0], got {rewards}"
    print("  ✓ Only one line → 0.0")
    
    # Test case 4: First line has extra tokens
    responses = ["结论是通过\n理由: 符合要求"]
    row = {}
    rewards = format_reward(responses, row)
    assert rewards == [0.0], f"Expected [0.0], got {rewards}"
    print("  ✓ Extra tokens in line 1 → 0.0")
    
    # Test case 5: Second line missing "理由:"
    responses = ["通过\n符合要求"]
    row = {}
    rewards = format_reward(responses, row)
    assert rewards == [0.0], f"Expected [0.0], got {rewards}"
    print("  ✓ Missing '理由:' prefix → 0.0")
    
    # Test case 6: Full-width colon (理由：)
    responses = ["不通过\n理由：缺少螺丝"]
    row = {}
    rewards = format_reward(responses, row)
    assert rewards == [1.0], f"Expected [1.0], got {rewards}"
    print("  ✓ Full-width colon accepted → 1.0")
    
    # Test case 7: Trailing whitespace allowed
    responses = ["通过\n理由: 符合要求  \n"]
    row = {}
    rewards = format_reward(responses, row)
    assert rewards == [1.0], f"Expected [1.0], got {rewards}"
    print("  ✓ Trailing whitespace allowed → 1.0")


if __name__ == "__main__":
    test_label_reward()
    test_format_reward()
    print("\n✅ All tests passed!")

