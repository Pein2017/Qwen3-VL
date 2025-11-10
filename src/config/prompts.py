"""Prompt templates for training.

Updated to grouped-JSON format with 类型/属性/[条件属性]/[备注] 层级。
Supports dynamic per-group prompt selection for mixed dense/summary training.
"""

# ============================================================================
# Shared Prior Rules (reused across dense/summary modes)
# ============================================================================

PRIOR_RULES = (
    "- 可能混杂无关内容（如施工图纸、杂物等），仅识别与任务相关的对象。\n"
    "- 爱立信品牌BBU：不安装挡风板。\n"
    "- 仅当存在两台BBU时，在两台BBU之间安装挡风板；单台BBU无需安装。\n"
    "- 光纤对象：若同一根光纤被分段检测到（颜色/路径连续），必须合并为一个对象，不得拆分为多个对象。\n"
    '- 光纤保护判定：任一片段存在保护套/蛇形管/铠装等防护材料，即整条光纤记为"有保护"；仅当整条光纤均无任何保护时，标注为"无保护措施"。\n'
)

# ============================================================================
# Prompt Schemes
# ============================================================================

FORMAT_HINTS = {
    "type_a": "- JSON 排布：整段单行、冒号与逗号后禁止空格，禁止任何换行或制表符。\n",
    "type_b": "- JSON 排布：整段单行，逗号和冒号后各保留一个空格，禁止换行。\n",
    "type_c": "- JSON 排布：可换行缩进；坐标点需独立成行，逗号保留。\n",
    "type_d": "- JSON 排布：quad/line 需改写为 {\"x\":...,\"y\":...} 对象列表，可缩进换行。\n",
}

_DEFAULT_JSON_FORMAT = "type_c"


def _normalize_format_key(value: str | None) -> str:
    if not value:
        return _DEFAULT_JSON_FORMAT
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    alias = {
        "a": "type_a",
        "b": "type_b",
        "c": "type_c",
        "d": "type_d",
        "typea": "type_a",
        "typeb": "type_b",
        "typec": "type_c",
        "typed": "type_d",
    }
    normalized = alias.get(normalized, normalized)
    return normalized if normalized in FORMAT_HINTS else _DEFAULT_JSON_FORMAT


DENSE_SYSTEM_PROMPT_CORE = (
    "你是图像密集标注助手。只输出一个 JSON 对象 {\"object_1\":{...}}，不要额外文字。\n"
    "- 对象按“自上到下 → 左到右”排序（线以最左端点为起点），编号从 1 递增。\n"
    "- 每个对象仅包含 desc + 单个几何键（bbox_2d/quad/line）；线对象额外提供整数 line_points。\n"
    "- desc 采用“类型/属性[,属性]/[条件属性]”层级，不得包含多余空格或换行。\n"
    "- 坐标使用 norm1000 整数（0..1000）。\n"
)


def build_dense_system_prompt(json_format: str | None) -> str:
    fmt = _normalize_format_key(json_format)
    format_hint = FORMAT_HINTS[fmt]
    return DENSE_SYSTEM_PROMPT_CORE + format_hint + "先验规则：\n" + PRIOR_RULES


"""Dense captioning system prompt - JSON pathway"""
SYSTEM_PROMPT_JSON = build_dense_system_prompt(_DEFAULT_JSON_FORMAT)


"""Scheme SUMMARY: per-image summary variant - one-line text per image"""
SYSTEM_PROMPT_SUMMARY = (
    "你是图像摘要助手。只返回一行中文摘要文本，不要任何解释或额外符号。若图片与任务无关或目标不可见，必须返回“无关图片”。\n\n"
    "输出要求：\n"
    "1) 单行摘要覆盖该图片中检测到的所有相关对象。\n"
    "2) 相同对象必须合并计数，使用“对象描述×N”（×为全角乘号、紧贴N不留空格）；各条目之间仅用中文逗号'，'分隔，并按“自上到下、再从左到右”的出现顺序（线对象以最左端点为起点）。\n"
    "3) 对象描述沿用密集标注的层级规范：类型/属性[,属性]/[条件属性]；如需备注，仅在末尾追加一次“，备注: ...”。\n"
    "4) 单句输出：整行不得换行，不要在末尾加句号'。'，不要多余空格或首尾分隔符。\n"
    "5) 仅输出摘要文本；不得返回 JSON 或其它包装结构。\n\n"
    "决策规则（严格二选一）：\n"
    "- 若存在与任务相关的任意目标：仅输出若干条“类型/属性[,属性]/[条件属性]×N”，条目用中文逗号'，'分隔。\n"
    "- 否则：严格输出“无关图片”。\n"
    "禁止输出：无法判断、证明文件、文件、文档、报告、图纸、票据等；以上情形一律视为“无关图片”。\n"
    "不确定/遮挡/模糊时不要猜测，输出“无关图片”。\n\n"
    "先验规则（与密集标注共享的业务知识）：\n" + PRIOR_RULES + "\n\n"
    "【严格禁止】\n"
    "- 禁止任何坐标数字（如整数数组）\n"
    "- 禁止几何字段名（bbox_2d、quad、line）\n"
    "- 禁止方括号[]包裹的数字列表\n"
    "- 禁止任何特殊标记或尖括号<>\n"
    "- 仅输出纯文本摘要，不得包含任何结构化几何信息\n\n"
)


USER_PROMPT_JSON = (
    '基于所给图片，检测并列出所有对象：按"自上到下再从左到右"排序，'
    "坐标使用 norm1000 整数网格，严格按规范返回 JSON。"
)

USER_PROMPT_SUMMARY = (
    "请对每张图片输出一行中文摘要：相同对象合并为“对象描述×N”（×为全角乘号、紧贴N不留空格），条目之间仅用中文逗号'，'分隔，并按“自上到下、再从左到右”排序。"
    "对象描述沿用密集标注的类型/属性[,属性]/[条件属性]层级，如需备注，仅在末尾追加一次“，备注: ...”。整行必须为单句：不得换行，不要句号'。'结尾。只返回摘要文本，不要坐标、几何字段或解释。"
)


SYSTEM_PROMPT = SYSTEM_PROMPT_JSON
USER_PROMPT = USER_PROMPT_JSON


__all__ = [
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_JSON",
    "SYSTEM_PROMPT_SUMMARY",
    "USER_PROMPT",
    "USER_PROMPT_JSON",
    "USER_PROMPT_SUMMARY",
    "build_dense_system_prompt",
]
