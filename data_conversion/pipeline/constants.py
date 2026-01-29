#!/usr/bin/env python3
"""
Centralized constants for the data conversion pipeline.

These constants are kept for backward compatibility; they should not be used
to filter objects anymore. The pipeline now accepts all objects and relies on
validation to prune invalid entries. Taxonomy JSONs are documentation-only.
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
    # RRU dataset
    "station",
    "rru",
    "rru_screw",
    "ground_screw",
    "fastener",
    "lable",
    "fiber_rru",
    "wire_rru",
}

# Chinese display labels for object types (from both mapping files)
CHINESE_LABELS: Dict[str, str] = {
    "bbu": "BBU设备",
    "bbu_shield": "挡风板",
    "connect_point": "螺丝、光纤插头",
    "label": "标签",
    "fiber": "光纤",
    "wire": "电线",
    # RRU dataset
    "station": "站点",
    "rru": "RRU设备",
    "rru_screw": "RRU接地端",
    "ground_screw": "地排接地端螺丝",
    "fastener": "紧固件",
    "lable": "标签(文本)",
    "fiber_rru": "尾纤",
    "wire_rru": "接地线",
}

# Reverse mapping for convenience: Chinese -> English key
REVERSE_CHINESE_LABELS: Dict[str, str] = {v: k for k, v in CHINESE_LABELS.items()}

# Geometry constraints by object type (from hierarchical_attribute_mapping.metadata.geometry_constraints)
LINE_OBJECT_TYPES: Set[str] = {"fiber", "wire"}
QUAD_BBOX_OBJECT_TYPES: Set[str] = {"bbu", "bbu_shield", "connect_point", "label"}

# Default hierarchy matching the grpo_summary_1024_attr_key_recall data structure (Chinese-only mode)
# Each key is the Chinese object label; values are a permissive set of accepted
# property tokens derived from mapping templates and enumerations. This is used
# only when no explicit hierarchy file is provided.
DEFAULT_LABEL_HIERARCHY: Dict[str, List[str]] = {
    # connect_point
    # - type: BBU安装螺丝, 机柜处接地螺丝, 地排处接地螺丝, ODF端光纤插头, BBU端光纤插头
    # - completeness: 完整, 部分
    # - compliance: 符合, 不符合
    "螺丝、光纤插头": [
        "BBU安装螺丝",
        "机柜处接地螺丝",
        "地排处接地螺丝",
        "ODF端光纤插头",
        "BBU端光纤插头",
        "完整",
        "部分",
        "符合",
        "不符合",
    ],
    # label (free text in mapping) — keep empty to avoid constraining values
    # Note: when using a hierarchy file, prefer a rule that does not restrict label free text
    "标签": [],
    # bbu
    # - brand: 华为, 中兴, 爱立信
    # - completeness: 完整, 部分
    # - windshield requirement/conformity (combined forms appear in labels):
    #   免装, 空间充足需安装
    "BBU设备": [
        "华为",
        "中兴",
        "爱立信",
        "完整",
        "部分",
        "免装",
        "空间充足需安装",
    ],
    # fiber
    # - obstruction: 无遮挡, 有遮挡
    # - protection: 无保护, 有保护（细化: 蛇形管/铠装/蛇形管+铠装）
    # - bend_radius: 半径合理, 半径不合理<4cm或成环
    "光纤": [
        # occlusion tokens removed (低价值): "无遮挡", "有遮挡"
        "无保护",
        "有保护",
        "蛇形管",
        "铠装",
        "蛇形管+铠装",
        "半径合理",
        "半径不合理<4cm或成环",
    ],
    # wire
    # - obstruction: 无遮挡, 有遮挡
    # - organization: 整齐, 散乱
    "电线": [
        # occlusion tokens removed (低价值): "无遮挡", "有遮挡"
        "整齐",
        "散乱",
    ],
    # bbu_shield
    # - brand: 华为, 中兴
    # - completeness: 完整, 部分
    # - obstruction: 挡风板无遮挡, 挡风板有遮挡
    # - direction: 方向正确, 方向错误
    "挡风板": [
        "华为",
        "中兴",
        "完整",
        "部分",
        # occlusion tokens removed (低价值): "挡风板无遮挡", "挡风板有遮挡"
        "方向正确",
        "方向错误",
    ],
}
