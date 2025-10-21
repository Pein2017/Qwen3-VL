#!/usr/bin/env python3
"""
Centralized constants for the data conversion pipeline.

Contains the canonical set of object types and the default label hierarchy
fallback used by the unified processor in Chinese-only mode.

Derived from `data_conversion/hierarchical_attribute_mapping.json` and
`data_conversion/attribute_taxonomy.json`.
"""

from typing import Dict, List, Set


# Canonical set of supported object types
OBJECT_TYPES: Set[str] = {
    "bbu",
    "bbu_shield",
    "label",
    "fiber",
    "wire",
    "connect_point",
}

# Chinese display labels for object types (from both mapping files)
CHINESE_LABELS: Dict[str, str] = {
    "bbu": "BBU设备",
    "bbu_shield": "挡风板",
    "connect_point": "螺丝、光纤插头",
    "label": "标签",
    "fiber": "光纤",
    "wire": "电线",
}

# Reverse mapping for convenience: Chinese -> English key
REVERSE_CHINESE_LABELS: Dict[str, str] = {v: k for k, v in CHINESE_LABELS.items()}

# Geometry constraints by object type (from hierarchical_attribute_mapping.metadata.geometry_constraints)
LINE_OBJECT_TYPES: Set[str] = {"fiber", "wire"}
QUAD_BBOX_OBJECT_TYPES: Set[str] = {"bbu", "bbu_shield", "connect_point", "label"}

# Default hierarchy matching the v2 data structure (Chinese-only mode)
# Each key is the Chinese object label; values are a permissive set of accepted
# property tokens derived from mapping templates and enumerations. This is used
# only when no explicit hierarchy file is provided.
DEFAULT_LABEL_HIERARCHY: Dict[str, List[str]] = {
    # connect_point
    # - type: BBU安装螺丝, 机柜处接地螺丝, 地排处接地螺丝, ODF端光纤插头, BBU端光纤插头
    # - completeness: 显示完整, 只显示部分
    # - compliance: 符合要求, 不符合要求
    "螺丝、光纤插头": [
        "BBU安装螺丝",
        "机柜处接地螺丝",
        "地排处接地螺丝",
        "ODF端光纤插头",
        "BBU端光纤插头",
        "显示完整",
        "只显示部分",
        "符合要求",
        "不符合要求",
    ],
    # label (free text in mapping) — keep empty to avoid constraining values
    # Note: when using a hierarchy file, prefer a rule that does not restrict label free text
    "标签": [],
    # bbu
    # - brand: 华为, 中兴, 爱立信
    # - completeness: 显示完整, 只显示部分
    # - windshield requirement/conformity (combined forms appear in labels):
    #   无需安装, 机柜空间充足，需要安装/这个BBU设备按要求配备了挡风板,
    #   机柜空间充足，需要安装/这个BBU设备未按要求配备挡风板
    "BBU设备": [
        "华为",
        "中兴",
        "爱立信",
        "显示完整",
        "只显示部分",
        "无需安装",
        "机柜空间充足，需要安装/这个BBU设备按要求配备了挡风板",
        "机柜空间充足，需要安装/这个BBU设备未按要求配备挡风板",
    ],
    # fiber
    # - obstruction: 无遮挡, 有遮挡
    # - protection: 无保护措施, 有保护措施（细化: 蛇形管/铠装/同时有蛇形管和铠装）
    # - bend_radius: 弯曲半径合理, 弯曲半径不合理（弯曲半径<4cm或者成环）
    "光纤": [
        # occlusion tokens removed (低价值): "无遮挡", "有遮挡"
        "无保护措施",
        "有保护措施",
        "蛇形管",
        "铠装",
        "同时有蛇形管和铠装",
        "弯曲半径合理",
        "弯曲半径不合理（弯曲半径<4cm或者成环）",
    ],
    # wire
    # - obstruction: 无遮挡, 有遮挡
    # - organization: 捆扎整齐, 分布散乱
    "电线": [
        # occlusion tokens removed (低价值): "无遮挡", "有遮挡"
        "捆扎整齐",
        "分布散乱",
    ],
    # bbu_shield
    # - brand: 华为, 中兴
    # - completeness: 显示完整, 只显示部分
    # - obstruction: 挡风板无遮挡, 挡风板有遮挡
    # - direction: 安装方向正确, 安装方向错误
    "挡风板": [
        "华为",
        "中兴",
        "显示完整",
        "只显示部分",
        # occlusion tokens removed (低价值): "挡风板无遮挡", "挡风板有遮挡"
        "安装方向正确",
        "安装方向错误",
    ],
}
