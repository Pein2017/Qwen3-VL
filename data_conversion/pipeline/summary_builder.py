#!/usr/bin/env python3
"""
Deterministic summary builder for BBU/rru QC dataset (raw-desc mode).

Current rule set (2025-11):
- Do NOT parse/canonicalize; use the raw `desc` string as the grouping key.
- Count identical desc values, sort by desc length (asc), tie-break by first appearance.
- Emit `desc×N` segments joined with '，'.
- Fail-fast: raises `ValueError` when objects list is empty or all desc are empty/blank.

Legacy parsing helpers remain below for reference/compatibility but are not used by the
active build_summary_from_objects implementation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


# Object type headers (Chinese)
BBU = "BBU设备"
SHIELD = "挡风板"
CONNECT = "螺丝、光纤插头"
LABEL = "标签"
FIB = "光纤"
WIRE = "电线"

# Canonical tokens
CP_COMPLY_OK = "符合要求"
CP_COMPLY_BAD = "不符合要求"
CP_ISSUES = {"未拧紧", "露铜", "复接", "生锈"}

# Connect point sub-types to retain in summary (first-level token)
CP_TYPES = {"BBU安装螺丝", "机柜处接地螺丝", "地排处接地螺丝", "ODF端光纤插头", "BBU端光纤插头"}

SHIELD_DIR_OK = "安装方向正确"
SHIELD_DIR_BAD = "安装方向错误"

FIB_PROTECT_NONE = "无保护措施"
FIB_PROTECT_HAVE = "有保护措施"
FIB_PROTECT_DETAILS = {"蛇形管", "铠装", "同时有蛇形管和铠装"}
FIB_BEND_OK = "弯曲半径合理"
FIB_BEND_BAD = "弯曲半径不合理（弯曲半径<4cm或者成环）"
# Legacy half-width variant (accepted for robustness)
FIB_BEND_BAD_HALF = "弯曲半径不合理(弯曲半径<4cm或者成环)"

WIRE_NEAT = "捆扎整齐"
WIRE_MESS = "分布散乱"

BBU_WS_REQ = "机柜空间充足需要安装"
BBU_WS_NO = "无需安装"
BBU_WS_OK = "这个BBU设备按要求配备了挡风板"
BBU_WS_BAD = "这个BBU设备未按要求配备挡风板"


_RE_REMARK = re.compile(r"备注[:：]\s*(.+)$")


def _split_commas(text: str) -> List[str]:
    """Split by Chinese/ASCII commas and strip."""
    if not isinstance(text, str) or not text:
        return []
    return [t.strip() for t in text.replace("，", ",").split(",") if t.strip()]


def _collect_remarks(objects: List[Dict[str, Any]]) -> List[str]:
    """Collect unique remarks from all objects (matches SummaryHandler)."""
    remarks: List[str] = []
    seen: set[str] = set()
    pattern = re.compile(r"备注[:：]\s*(.+)$")
    for obj in objects:
        desc = str(obj.get("desc", "")).strip()
        if not desc:
            continue
        for seg in desc.split("/"):
            m = pattern.search(seg)
            if not m:
                continue
            content = m.group(1).strip()
            if not content:
                continue
            # normalize ending punctuation
            content = content.strip("；，。;,")
            if content and content not in seen:
                seen.add(content)
                remarks.append(content)
    return remarks


def _summarize_object_string(desc: str) -> str:
    """Summarize a single object's desc to one canonical string with full context.

    Examples:
    - BBU/华为/无需挡风板
    - BBU/爱立信/挡风板未按要求配备
    - 连接点/不合规-未拧紧,生锈
    - 连接点/合规
    - 光纤/无保护/弯曲合理
    - 光纤/有保护/蛇形管/弯曲不合理
    - 电线/捆扎整齐
    - 标签/可以识别
    - 挡风板/安装方向错误
    """
    if not isinstance(desc, str) or not desc.strip():
        return ""
    parts = [p.strip() for p in desc.split("/") if p.strip()]
    if not parts:
        return ""

    kind = parts[0]

    if kind == BBU:
        lvl1 = _split_commas(parts[1]) if len(parts) >= 2 else []
        brand = ""
        for b in ("华为", "中兴", "爱立信"):
            if b in lvl1:
                brand = b
                break
        completeness = ""
        if "显示完整" in lvl1:
            completeness = "显示完整"
        elif "只显示部分" in lvl1:
            completeness = "只显示部分"
        status = ""
        if BBU_WS_REQ in lvl1:
            conf = parts[2].strip() if len(parts) >= 3 else ""
            if conf == BBU_WS_OK:
                status = "这个BBU设备按要求配备了挡风板"
            elif conf == BBU_WS_BAD or conf == "":
                status = "这个BBU设备未按要求配备挡风板"
        elif BBU_WS_NO in lvl1:
            status = "无需安装"
        tokens = ["BBU设备"]
        if brand:
            tokens.append(brand)
        if completeness:
            tokens.append(completeness)
        if status:
            # normalize to mapping's first value tokens
            tokens.append(status)
        return "/".join(tokens)

    if kind == CONNECT and len(parts) >= 2:
        lvl1 = _split_commas(parts[1])
        # Keep the connect_point subtype (e.g., BBU安装螺丝) if present
        cp_type = next((t for t in lvl1 if t in CP_TYPES), "")
        completeness = ""
        if "显示完整" in lvl1:
            completeness = "显示完整"
        elif "只显示部分" in lvl1:
            completeness = "只显示部分"
        if CP_COMPLY_BAD in lvl1:
            issues: List[str] = []
            if len(parts) >= 3:
                issues = [it for it in _split_commas(parts[2]) if it in CP_ISSUES]
            if issues:
                return "/".join([
                    p for p in [
                        "螺丝、光纤插头",
                        cp_type,
                        completeness,
                        f"不符合要求/{','.join(sorted(set(issues)))}",
                    ]
                    if p
                ])
            return "/".join([p for p in ["螺丝、光纤插头", cp_type, completeness, "不符合要求"] if p])
        if CP_COMPLY_OK in lvl1:
            return "/".join([p for p in ["螺丝、光纤插头", cp_type, completeness, "符合要求"] if p])
        return ""

    if kind == FIB and len(parts) >= 2:
        lvl1 = _split_commas(parts[1])
        protect = ""
        detail = ""
        if FIB_PROTECT_NONE in lvl1:
            protect = "无保护措施"
        elif FIB_PROTECT_HAVE in lvl1:
            protect = "有保护措施"
            det = parts[2].strip() if len(parts) >= 3 else ""
            if det in FIB_PROTECT_DETAILS:
                detail = det
        bend = ""
        if (FIB_BEND_BAD in lvl1) or (FIB_BEND_BAD_HALF in lvl1):
            bend = "弯曲半径不合理（弯曲半径<4cm或者成环）"
        elif FIB_BEND_OK in lvl1:
            bend = "弯曲半径合理"
        tokens = ["光纤"]
        if protect:
            tokens.append(protect)
        if detail:
            tokens.append(detail)
        if bend:
            tokens.append(bend)
        return "/".join(tokens)

    if kind == WIRE and len(parts) >= 2:
        lvl1 = _split_commas(parts[1])
        if WIRE_MESS in lvl1:
            return "电线/分布散乱"
        if WIRE_NEAT in lvl1:
            return "电线/捆扎整齐"
        return ""

    if kind == LABEL:
        text = parts[1].strip() if len(parts) >= 2 else ""
        if text in {"无法识别", "不能"}:
            return "标签/无法识别"
        return "标签/可以识别"

    if kind == SHIELD and len(parts) >= 2:
        lvl1 = _split_commas(parts[1])
        completeness = ""
        if "显示完整" in lvl1:
            completeness = "显示完整"
        elif "只显示部分" in lvl1:
            completeness = "只显示部分"
        if SHIELD_DIR_BAD in lvl1:
            return "/".join([p for p in ["挡风板", completeness, "安装方向错误"] if p])
        if SHIELD_DIR_OK in lvl1:
            return "/".join([p for p in ["挡风板", completeness, "安装方向正确"] if p])
        return ""

    return ""


def _type_order_key(s: str) -> Tuple[int, int, int, str]:
    """Stable ordering: type → negative-first → brand → lexicographic."""
    # Custom ordering: BBU设备 → 挡风板 → 光纤 → 电线 → 螺丝、光纤插头 → 标签
    type_map = {
        "BBU设备": 0,
        "挡风板": 1,
        "光纤": 2,
        "电线": 3,
        "螺丝、光纤插头": 4,
        "标签": 5,
    }
    head = s.split("/", 1)[0]
    primary = type_map.get(head, 99)
    # negative-first within types
    negative_score = 1
    if head == "螺丝、光纤插头":
        negative_score = 0 if "/不符合要求" in s else 1
    elif head == "光纤":
        negative_score = 0 if "弯曲半径不合理" in s else 1
    elif head == "挡风板":
        negative_score = 0 if "安装方向错误" in s else 1
    elif head == "BBU设备":
        if "未按要求配备挡风板" in s:
            negative_score = 0
        elif "按要求配备了挡风板" in s:
            negative_score = 1
        elif "无需安装" in s:
            negative_score = 2
        else:
            negative_score = 3
    # brand order for BBU
    brand_score = 9
    if head == "BBU设备":
        parts = s.split("/")
        brand = parts[1] if len(parts) >= 2 else ""
        brand_order = {"华为": 0, "中兴": 1, "爱立信": 2, "": 3}
        brand_score = brand_order.get(brand, 8)
    return (primary, negative_score, brand_score, s)


def build_summary_from_objects(objects: List[Dict[str, Any]]) -> str:
    """Build one-line summary by grouping on the raw `desc` strings.

    - No parsing/canonicalization: use the full desc as the grouping key.
    - Count identical desc values.
    - Sort by string length (ascending); ties keep first appearance order.
    - Emit segments like: 'BBU设备/需复核,备注:无法判断品牌×1，标签/无法识别×2'.
    """
    if not objects:
        raise ValueError("build_summary_from_objects: no objects provided (fail-fast)")

    descs: List[str] = []
    for obj in objects:
        desc = obj.get("desc", "")
        if not isinstance(desc, str):
            desc = str(desc)
        desc = desc.strip()
        if desc:
            descs.append(desc)

    if not descs:
        raise ValueError("build_summary_from_objects: objects missing non-empty desc (fail-fast)")

    first_seen: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for idx, desc in enumerate(descs):
        if desc not in first_seen:
            first_seen[desc] = idx
            counts[desc] = 0
        counts[desc] += 1

    # Stable sort: length first, then first appearance index to preserve input order for ties
    sorted_descs = sorted(counts.keys(), key=lambda d: (len(d), first_seen[d]))
    segments = [f"{d}×{counts[d]}" for d in sorted_descs]

    summary = "，".join(segments)
    summary = summary.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
    return summary
