"""Shared summary prompt primitives consumed by config + runtime helpers."""

from typing import Sequence, Tuple

from typing_extensions import override

from .domain_packs import get_domain_pack

SUMMARY_LABEL_GROUPING_DISABLED_RULE = (
    "- 标签类 desc 必须保留原始 OCR 文本，不使用“标签/可以识别”或“标签/无法识别”归并。\n"
)


def build_summary_system_prompt_minimal(
    *, summary_label_grouping: bool | None = None
) -> str:
    """Minimal summary-mode system prompt with strict output contract."""

    content_rules = ""
    if summary_label_grouping is False:
        content_rules = SUMMARY_LABEL_GROUPING_DISABLED_RULE
    return "".join(
        [
            "你是图像摘要助手。只用简体中文输出一行摘要，不要解释。\n\n",
            "【输出格式】\n",
            "- 无关/非现场/无目标：只输出“无关图片”。\n",
            "- 否则：输出 desc×N，中文逗号分隔；相同 desc 合并；按 desc 字数从少到多排序（同字数保留首次顺序）。\n",
            "- 单行输出，不换行、不加句号。\n\n",
            "【内容约束】\n",
            "- 仅客观罗列，不输出通过/不通过等判定。\n",
            "- 文档/报告/图纸/CAD/平面图/示意图/票据/聊天或手机截图等非现场，一律视为无关图片。\n",
            "- 证据优先：只写看得见的对象；不确定则用“类型/需复核”，或直接无关图片。\n",
            "- 禁止 JSON/坐标/几何字段/方括号数字列表/尖括号标记。\n",
            content_rules,
        ]
    )


MISSION_SPECIFIC_PRIOR_RULES: dict[str, str] = {
    "挡风板安装检查": (
        "- 爱立信品牌BBU：不安装挡风板。\n"
        "- 一个华为挡风板对应着一个黄色箭头。如果有两个黄色箭头，那通常意为着存在两个挡风板。\n"
        "- 仅当存在两台BBU时，在两台BBU之间必须安装挡风板；单台BBU通常无需安装。\n"
    ),
}


class _LazySummaryRuntimePrompt:
    """Delay runtime prompt construction to avoid import cycles."""

    _value: str | None = None

    def _get_value(self) -> str:
        if self._value is None:
            self._value = (
                build_summary_system_prompt_minimal().strip()
                + "\n\n"
                + get_domain_pack("bbu").block
            )
        return self._value

    @override
    def __str__(self) -> str:
        return self._get_value()

    @override
    def __repr__(self) -> str:
        return f"<SYSTEM_PROMPT_SUMMARY_RUNTIME: {self._get_value()[:50]}...>"

    def __len__(self) -> int:
        return len(self._get_value())

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self._get_value() == other
        return self is other

    def __add__(self, other: str) -> str:
        return self._get_value() + other

    def __radd__(self, other: str) -> str:
        return other + self._get_value()


SYSTEM_PROMPT_SUMMARY_TRAIN = build_summary_system_prompt_minimal()
SYSTEM_PROMPT_SUMMARY_RUNTIME = _LazySummaryRuntimePrompt()
SYSTEM_PROMPT_SUMMARY = SYSTEM_PROMPT_SUMMARY_TRAIN

USER_PROMPT_SUMMARY = (
    "请输出一行中文摘要：按 desc 原文分组并合并为 desc×N（×为全角乘号、紧贴N），"
    "条目仅用中文逗号分隔，按 desc 字数从少到多排序（同字数保留首次顺序）。"
    "保持 desc 原样（含备注），仅客观罗列，不输出通过/不通过等判定。"
    "不得换行或加句号；不输出坐标/几何字段/解释。"
    "无关或非现场图片仅输出“无关图片”。"
)

__all__ = [
    "SUMMARY_LABEL_GROUPING_DISABLED_RULE",
    "build_summary_system_prompt_minimal",
    "MISSION_SPECIFIC_PRIOR_RULES",
    "SYSTEM_PROMPT_SUMMARY",
    "SYSTEM_PROMPT_SUMMARY_RUNTIME",
    "SYSTEM_PROMPT_SUMMARY_TRAIN",
    "USER_PROMPT_SUMMARY",
]
