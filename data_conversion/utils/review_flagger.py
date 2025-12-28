#!/usr/bin/env python3
"""
Review flagger: rewrite contradictory labels to a unified “需复核” marker.

Heuristic:
- Base描述含正向词（符合/方向正确/按要求配备挡风板/半径合理/免装/完整）
- 备注含不确定或缺失信息（无法判断/不确定/疑似/未拍全/无品牌/缺少/角度问题等）
=> 将 desc 改写为 `<原类型>/需复核[,备注:... ]`
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


POSITIVE_TOKENS = [
    "符合",
    "方向正确",
    "按要求配备",
    "半径合理",
    "免装",
    "完整",
]

UNCERTAIN_OR_NEG = [
    # 不确定/信息不足
    "无法判断",
    "不确定",
    "疑似",
    "无法框选",
    "无法框全",
    "无法框",
    "无法辨别",
    "无法看",
    "范围过小",
    "角度问题",
    "拍摄角度",
    "未拍",
    "未显示",
    "无品牌",
    "品牌不完整",
    "未体现品牌",
    "无显示品牌",
    "显示小部分",
    "部分",
    "不完整",
    "空间有限",
    "位置错误",
    "应该",
    "推测",
    "判断",
    # 缺件/缺少
    "缺少",
    "缺失",
    "不全",
    "损坏",
    "部分未",
    "部分无",
    "部分没有",
    "未套",
    "不是常见",
    # 其他
    "建议",
    "背面",
    "不能确认",
    "不能判断",
    "不能确定",
    "无法确定",
]


def _split_desc(desc: str) -> Tuple[str, str]:
    """Split desc into base and remark parts."""
    if ",备注:" in desc:
        base, remark = desc.split(",备注:", 1)
        return base, remark
    for mark in ("备注：", "备注:", "备注"):
        if mark in desc:
            base, remark = desc.split(mark, 1)
            return base, remark
    return desc, ""


# Remarks containing这些短语视为标注说明噪声，直接移除
NOISE_REMARK_TOKENS = [
    "请参考学习",
    "建议看下操作手册中螺丝、插头的标注规范",
]


def _clean_remark(remark: str) -> str:
    if not remark:
        return remark
    for tok in NOISE_REMARK_TOKENS:
        if tok in remark:
            return ""
    return remark.strip()


def _head(desc: str) -> str:
    """Extract the leading object type/token."""
    prefix = desc.split("/", 1)[0]
    prefix = prefix.split(",", 1)[0]
    return prefix.strip()


def is_contradictory_desc(desc: str) -> bool:
    """Return True if desc has a remark containing uncertainty/缺失信号."""
    if not desc or "备注" not in desc:
        return False
    _, remark_raw = _split_desc(desc)
    remark = _clean_remark(remark_raw)
    if not remark:
        return False
    return any(tok in remark for tok in UNCERTAIN_OR_NEG)


def flag_objects_for_review(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rewrite contradictory desc to '<type>/需复核[,备注:...]'."""
    flagged: List[Dict[str, Any]] = []
    for obj in objects:
        desc = obj.get("desc", "")
        if not desc:
            flagged.append(obj)
            continue

        base, remark_raw = _split_desc(desc)
        remark = _clean_remark(remark_raw)
        clean_desc = base if not remark else f"{base},备注:{remark}"

        if is_contradictory_desc(clean_desc):
            head = _head(base) or "需复核"
            new_desc = f"{head}/需复核"
            if remark:
                new_desc = f"{new_desc},备注:{remark}"
            new_obj = obj.copy()
            new_obj["desc"] = new_desc
            flagged.append(new_obj)
        else:
            if clean_desc != desc:
                new_obj = obj.copy()
                new_obj["desc"] = clean_desc
                flagged.append(new_obj)
            else:
                flagged.append(obj)
    return flagged


__all__ = ["flag_objects_for_review", "is_contradictory_desc"]
