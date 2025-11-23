#!/usr/bin/env python3
"""
List labels containing '备注' and flag likely contradictions.

Usage:
  python scripts/list_labels_with_remarks.py [--vocab data/bbu_full_768_poly/label_vocabulary.json]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


REMARK_MARK = "备注"

# Heuristics: main label looks positive/definitive, remark is uncertain or negative.
POSITIVE_TOKENS = [
    "符合要求",
    "安装方向正确",
    "这个BBU设备按要求配备了挡风板",
    "弯曲半径合理",
    "无需安装",
    "显示完整",
]
UNCERTAIN_OR_NEG = [
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
    "缺少",
    "缺失",
    "不全",
    "损坏",
    "建议",
    "背面",
    "不能确认",
    "不能判断",
    "不能确定",
    "无法确定",
]


def load_labels(vocab_path: Path) -> List[str]:
    data = json.loads(vocab_path.read_text())
    return list(data["vocabulary"]["all_unique_labels"])


def split_label(label: str) -> Tuple[str, str]:
    if REMARK_MARK not in label:
        return label, ""
    # Prefer explicit separator if present
    if ",备注:" in label:
        base, remark = label.split(",备注:", 1)
        return base, remark
    base, remark = label.split(REMARK_MARK, 1)
    return base, remark


def find_with_remarks(labels: List[str]) -> List[str]:
    return [s for s in labels if REMARK_MARK in s]


def is_contradictory(label: str) -> bool:
    base, remark = split_label(label)
    return any(tok in base for tok in POSITIVE_TOKENS) and any(
        tok in remark for tok in UNCERTAIN_OR_NEG
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="List labels with remarks")
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("data/bbu_full_768_poly/label_vocabulary.json"),
        help="Path to label_vocabulary.json",
    )
    args = parser.parse_args()

    labels = load_labels(args.vocab)
    remark_labels = find_with_remarks(labels)
    contradictory = [s for s in remark_labels if is_contradictory(s)]

    print(f"含“备注”的 unique labels: {len(remark_labels)}")
    for s in remark_labels:
        print(s)

    print("\n疑似矛盾/不确定的词条: {}".format(len(contradictory)))
    for s in contradictory:
        print(s)


if __name__ == "__main__":
    main()
