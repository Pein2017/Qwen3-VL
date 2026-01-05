"""Shared summary prompt primitives consumed by config + runtime helpers."""


from typing_extensions import override

from .domain_packs import get_domain_pack

SUMMARY_LABEL_GROUPING_DISABLED_RULE = (
    "- 标签类统计必须保留 OCR 原文：用“文本”字段逐条计数，不做“可以识别”归并。\n"
)


def build_summary_system_prompt_minimal() -> str:
    """Minimal summary-mode system prompt with strict output contract."""

    content_rules = SUMMARY_LABEL_GROUPING_DISABLED_RULE
    return "".join(
        [
            "你是图像摘要助手。非无关图片输出两行：第1行 `<DOMAIN={domain}>, <TASK={task}>`；第2行输出 JSON 字符串。无关/非现场/无目标仅输出单行“无关图片”。不要解释。\n\n",
            "【输出格式】\n",
            "- 非无关：第1行固定前缀：`<DOMAIN={domain}>, <TASK={task}>`。\n",
            "- 无关/非现场/无目标：仅输出单行“无关图片”，不输出第1行。\n",
            "- 否则：第2行输出单行 JSON 字符串（不可换行、不加句号）。\n",
            "- 逗号和冒号后各保留一个空格。\n\n",
            "【JSON 约束】\n",
            "- JSON 必须严格有效（双引号键值、无尾逗号），以 `}` 结束。\n",
            "- JSON 之后不得追加任何字符（例如：`×1`、`份`、`次`）。\n\n",
            "【JSON 结构】\n",
            "- 顶层键：统计。\n",
            "- BBU 可包含 备注（字符串列表，去重）；RRU 不包含 备注，可包含 分组统计。\n",
            "- 统计: 列表；每项包含“类别”，以及该类别下出现过的属性计数（值→次数）。\n",
            "- 多值属性使用 `|` 分隔时，需分别计数。\n",
            "- 仅统计实际观察到的值；不要生成“缺失/第三态占位/遮挡”等未观测条目。\n",
            "- OCR：可读时只输出 文本=原文；不可读写 可读性=不可读；去空格，保留 `-` `/` `,` `|` `=`。\n\n",
            "【内容约束】\n",
            "- 仅客观罗列，不输出通过/不通过等判定。\n",
            "- 文档/报告/图纸/CAD/平面图/示意图/票据/聊天或手机截图等非现场，一律视为无关图片。\n",
            "- 证据优先：只写看得见的对象；不要填充不可见属性。\n",
            "- 禁止输出坐标/几何字段/方括号数字列表/尖括号标记。\n",
            content_rules,
        ]
    )


MISSION_SPECIFIC_PRIOR_RULES: dict[str, str] = {
    "挡风板安装检查": (
        "- 爱立信品牌BBU：不安装挡风板。\n"
        "- 一个华为挡风板对应着一个黄色箭头。如果有两个黄色箭头，那通常意为着存在两个挡风板。\n"
        "- 仅当存在两台BBU时，在两台BBU之间必须安装挡风板；单台BBU通常免装。\n"
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
    "请按如下格式输出：非无关图片输出两行，第1行 `<DOMAIN={domain}>, <TASK={task}>`；"
    "第2行输出单行 JSON 摘要。"
    "无关或非现场图片仅输出单行“无关图片”（不输出第1行）。"
    "JSON 包含 统计 字段。"
    "统计为类别+属性值计数，仅统计可见信息。"
    "BBU 需要 备注 列表；RRU 不包含 备注，可包含 分组统计。"
    "OCR 文本保留原文（去空格，保留- / , | =）；不可读写 可读性=不可读。"
    "JSON 必须严格有效（双引号、无尾逗号），并以 `}` 结束；禁止追加×N等后缀。"
)

__all__ = [
    "build_summary_system_prompt_minimal",
    "MISSION_SPECIFIC_PRIOR_RULES",
    "SYSTEM_PROMPT_SUMMARY",
    "SYSTEM_PROMPT_SUMMARY_RUNTIME",
    "SYSTEM_PROMPT_SUMMARY_TRAIN",
    "USER_PROMPT_SUMMARY",
]
