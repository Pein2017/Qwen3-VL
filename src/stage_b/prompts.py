#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-B prompting: system and user message templates for group-level judgment."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Import centralized mission definitions
from src.config.missions import STAGE_B_MISSION_FOCUS as MISSION_FOCUS_MAP

STAGE_B_SYSTEM_PROMPT = """你是通信机房质检助手。

任务背景：基于多张图片的单行摘要，做一个简明的工单级判断说明。
输出格式严格为两行：
  第一行：通过 或 不通过
  第二行：理由: <用自然语言简述关键依据，可合并表达>
仅做格式约束；不要输出坐标/特殊标记；不强制引用固定词表。
判断原则：需结合当前任务的检查要点进行综合判断。
冲突处理：阅读每条摘要中的'备注'片段；若主句与备注矛盾，以备注为准，且备注中的负向细节（如'未套蛇形管'、'未拧紧'等）优先级更高。
表达风格：使用自然语言短句描述证据，不要使用计数（例如'×N'）、不做编号/列表。
请避免重复用语/模板化句式，允许多种合理表达；若关键项不可确认（如标签缺失/安装方向不明），请输出'不通过'，并简要说明原因。"""


_INDEX_RE = re.compile(r"(\d+)$")


def _sorted_summaries(stage_a_summaries: Dict[str, str]) -> List[Tuple[str, str]]:
    def _index(key: str) -> int:
        match = _INDEX_RE.search(key)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return 0
        return 0

    return sorted(stage_a_summaries.items(), key=lambda item: _index(item[0]))


def build_stage_b_messages(
    stage_a_summaries: Dict[str, str],
    task_type: str,
) -> List[Dict[str, str]]:
    """Build Stage-B messages for group-level judgment.
    
    Args:
        stage_a_summaries: Dict mapping image_i (or legacy 图片_i) to Chinese summary text
        task_type: One of the 4 supported missions
        
    Returns:
        List of message dicts with system and user roles (text-only, no images)
        
    Example:
        >>> summaries = {"image_1": "BBU设备/华为/显示完整", "image_2": "螺丝/符合要求"}
        >>> msgs = build_stage_b_messages(summaries, "挡风板安装检查")
        >>> len(msgs)
        2
        >>> msgs[0]["role"]
        'system'
    """
    sorted_items = _sorted_summaries(stage_a_summaries)

    # Build summary lines text with deterministic ordering
    summary_lines = [
        f"图像 {idx}: {text}"
        for idx, (_, text) in enumerate(sorted_items, start=1)
    ]
    summaries_text = "\n".join(summary_lines)
    
    # Get mission-specific focus
    mission_focus = MISSION_FOCUS_MAP.get(task_type, "请结合任务进行综合判断")
    
    # Build user message
    user_text = f"""任务: {task_type}
要点: {mission_focus}
格式: 仅输出两行；第一行严格为'通过'或'不通过'；第二行以'理由: '开头，不得重复摘要原文。
说明：以下为每张图片的一行摘要，请基于全部摘要综合判断工单级结论（不逐条打分）；若主句与'备注'冲突，以'备注'为准。

摘要列表：
{summaries_text}"""
    
    messages = [
        {"role": "system", "content": STAGE_B_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    
    return messages

