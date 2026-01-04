"""Prompt templates for training.

Updated to key=value descriptions and JSON-string summaries, with dataset-level
schema hints and prior rules (BBU/RRU) for consistent domain grounding.
"""

from src.prompts.summary_core import (
    MISSION_SPECIFIC_PRIOR_RULES,
    SYSTEM_PROMPT_SUMMARY,
    SYSTEM_PROMPT_SUMMARY_RUNTIME,
    SYSTEM_PROMPT_SUMMARY_TRAIN,
    USER_PROMPT_SUMMARY,
    build_summary_system_prompt_minimal,
)

# ============================================================================
# Dataset-level schema hints & prior rules
# ============================================================================

# 结构提示（精简版）：只给出类型与属性**概念**，不枚举全部取值
DATASET_SCHEMA_HINT = {
    "bbu": (
        "可标注类型示例（desc 采用 key=value，无空格）：\n"
        "- BBU设备：品牌/可见性/挡风板需求/挡风板符合性/备注\n"
        "- 挡风板：品牌/可见性/安装方向/备注\n"
        "- BBU安装螺丝/BBU端光纤插头/ODF端光纤插头/机柜处接地螺丝/地排处接地螺丝：可见性/符合性/问题/备注\n"
        "- 光纤：保护措施/保护细节/弯曲半径/备注\n"
        "- 电线：捆扎/备注\n"
        "- 标签：文本/可读性\n"
    ),
    "rru": (
        "可标注类型示例（desc 采用 key=value，无空格）：\n"
        "- 站点距离：站点距离（数字）\n"
        "- RRU设备\n"
        "- 紧固件/RRU接地端/地排接地端螺丝：安装状态\n"
        "- 尾纤：标签/套管保护\n"
        "- 接地线：标签\n"
        "- 标签：文本/可读性\n"
        "- 分组：可选 组=<id>（组内至少2个对象）\n"
    ),
}

# 数据集专属先验：dense/summary 共用
DATASET_PRIOR_RULES = {
    "bbu": (
        "- 可能混杂无关内容（如施工图纸、杂物等），仅识别与任务相关的对象。\n"
        "- 爱立信品牌BBU：不安装挡风板。\n"
        "- 一个华为挡风板对应着一个黄色箭头。如果有两个黄色箭头，那通常意为着存在两个挡风板。\n"
        "- 仅当存在两台BBU时，在两台BBU之间必须安装挡风板；单台BBU通常免装。\n"
        "- 螺丝/挡风板与本体强关联，不会远离对应设备；BBU端光纤插头多为蓝白色且插在 BBU 设备上。\n"
        "- 标签通常为黄色纸质，贴在插头处或电线上。\n"
        "- 光纤对象：若同一根光纤被分段检测到（颜色/路径连续），必须合并为一个对象，不得拆分。\n"
        "- 光纤保护判定：任一片段存在保护套/蛇形管/铠装等防护材料，即整条光纤记为**有保护**；仅当整条光纤均无任何保护时，标注为**无保护**。\n"
        "- 备注用于补充说明（可省略）。\n"
    ),
    "rru": (
        "".join(
            [
                "- 场景聚焦 RRU 侧设备与接地/尾纤组件，忽略与任务无关的杂物/背景。\n",
                "- 站点距离仅保留数字，不推断单位。\n",
                "- RRU接地端、地排接地端螺丝、紧固件仅标 `安装状态=合格/不合格`，不补充品牌。\n",
                "- 尾纤需判断**是否带标签**与**是否有套管**；任一片段有保护即算有保护。\n",
                "- 接地线只判断是否带标签。\n",
                "- 分组：标签/尾纤/接地线按组出现；每组至少 2 个对象；分组号写入 desc 的 `组=<id>`。\n",
            ]
        ),
        "- 重点：检测 RRU设备、紧固件、线缆尾纤、RRU接地端、RRU接地线。\n",
        "- 尾纤保护：黑色防水胶带或白色尼龙套管；若为白色尼龙套管，标签通常贴在白色尼龙管上。\n",
    ),
}

_DEFAULT_DATASET = "bbu"



# ============================================================================
# Prompt Schemes
# ============================================================================

FORMAT_HINTS = {
    "standard": "- JSON 本体：单行输出，逗号和冒号后各保留一个空格，JSON 内禁止换行。\n",
}

_DEFAULT_JSON_FORMAT = "standard"


def _normalize_format_key(value: str | None) -> str:
    if not value:
        return _DEFAULT_JSON_FORMAT
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in FORMAT_HINTS else _DEFAULT_JSON_FORMAT


def _get_dataset(key: str | None) -> str:
    normalized = (key or _DEFAULT_DATASET).strip().lower()
    return normalized if normalized in DATASET_PRIOR_RULES else _DEFAULT_DATASET


DENSE_SYSTEM_PROMPT_CORE = (
    '你是图像密集标注助手。输出两行：第1行 `<DOMAIN={domain}>, <TASK={task}>`；第2行输出 JSON 对象 {"object_1":{...}}。\n'
    "- 对象按**自上到下 → 左到右**排序，编号从 1 递增。\n"
    "  * 排序规则详解：首先按 Y 坐标（纵向）从小到大排列（图像上方优先），Y 坐标相同时按 X 坐标（横向）从小到大排列（图像左方优先）。\n"
    "  * bbox_2d 排序参考点：使用左上角坐标 (x1, y1) 作为该对象的排序位置。\n"
    "  * poly 排序参考点：顶点按质心顺时针排序，起点为最上最左的顶点；使用首顶点 (x1, y1) 作为排序位置。\n"
    "  * line 排序参考点：使用最左端点（X 坐标最小的点）作为排序参考；若多个点的 X 坐标相同，则取其中 Y 坐标最小的点。\n"
    "- 证据优先：仅写可见证据；未知属性留空；不输出任何第三态占位。\n"
    "- 每个对象仅包含 desc + 单个几何键（bbox_2d/poly/line）；禁止多个几何键。\n"
    "- desc 采用 key=value（逗号分隔、无空格），必须包含 `类别` 且放首位；多值用 `|` 连接。\n"
    "- 标签 OCR：可读写 `文本=原文`；不可读写 `可读性=不可读`；去空格，保留 `-` `/` `,` `|` `=`。\n"
    "- 坐标使用 norm1000 整数（0..1000）：\n"
    "  * bbox_2d：扁平数组 [x1,y1,x2,y2]（左上、右下）。\n"
    "  * poly：二维数组对 [[x1,y1], [x2,y2], ...]（多边形；点数可变，至少 3 个点）。\n"
    "  * line：二维数组对 [[x1,y1], [x2,y2], ...]（折线/线段点序列）；必须包含整数 line_points（≥2），其值等于点对个数。\n"
)


def build_dense_system_prompt(
    json_format: str | None, *, dataset: str | None = None
) -> str:
    ds = _get_dataset(dataset)
    fmt = _normalize_format_key(json_format)
    format_hint = FORMAT_HINTS[fmt]
    schema_raw = DATASET_SCHEMA_HINT.get(ds, "")
    prior_raw = DATASET_PRIOR_RULES.get(ds, "")
    schema = (
        "".join(schema_raw) if isinstance(schema_raw, (tuple, list)) else schema_raw
    )
    prior = "".join(prior_raw) if isinstance(prior_raw, (tuple, list)) else prior_raw
    return DENSE_SYSTEM_PROMPT_CORE + format_hint + schema + "先验规则：\n" + prior


# Dense captioning system prompt - default (BBU)
SYSTEM_PROMPT_JSON = build_dense_system_prompt(_DEFAULT_JSON_FORMAT, dataset="bbu")


def build_summary_system_prompt(
    *,
    dataset: str | None = None,
    mission: str | None = None,
) -> str:
    ds = _get_dataset(dataset)
    schema_raw = DATASET_SCHEMA_HINT.get(ds, "")
    prior_raw = DATASET_PRIOR_RULES.get(ds, "")
    schema = (
        "".join(schema_raw) if isinstance(schema_raw, (tuple, list)) else schema_raw
    )
    prior = "".join(prior_raw) if isinstance(prior_raw, (tuple, list)) else prior_raw

    # 添加任务特定先验规则
    mission_prior = ""
    if mission and mission in MISSION_SPECIFIC_PRIOR_RULES:
        mission_prior_raw = MISSION_SPECIFIC_PRIOR_RULES[mission]
        mission_prior = (
            "".join(mission_prior_raw)
            if isinstance(mission_prior_raw, (tuple, list))
            else mission_prior_raw
        )
        if mission_prior:
            mission_prior = f"\n【{mission}任务特定规则】\n{mission_prior}"

    prompt = (
        build_summary_system_prompt_minimal().strip()
        + "\n\n"
        + schema
        + "先验规则（与密集标注共享的业务知识）：\n"
        + prior
        + mission_prior
    )
    return prompt


USER_PROMPT_JSON = (
    "基于所给图片，检测并列出所有对象：按**自上到下再从左到右**排序，"
    "坐标使用 norm1000 整数网格。"
    "输出两行：第1行 `<DOMAIN={domain}>, <TASK={task}>`；第2行输出 JSON。"
)

SYSTEM_PROMPT_AUX = (
    "You are a general-purpose detection annotator for source-domain datasets (e.g., LVIS). "
    "Return exactly one JSON object ordered top-to-bottom then left-to-right, with coordinates expressed in norm1000 integers. "
    "Each object must contain only a short English class name (one or two words) plus a single geometry key (bbox_2d or poly); "
    "do not add attributes, long descriptions, or quality commentary. "
    "If unsure about a category, pick the closest simple class name available in the dataset and keep the wording concise."
)

USER_PROMPT_AUX = "List every visible object using concise English class names only (no attributes or long phrases) and keep the output in JSON."

# Text-only chat source (language preservation)
# For pre-authored chat messages, system prompt is typically empty to preserve original conversation structure
SYSTEM_PROMPT_CHAT = ""  # Leave system blank for pre-authored chat messages
# User prompt is typically not used for pre-authored messages, but kept as fallback
USER_PROMPT_CHAT = ""  # Empty for pre-authored messages; messages field in JSONL contains full conversation


def get_template_prompts(name: str | None) -> tuple[str, str]:
    """Return (system, user) prompts for a known template name.

    We disallow silent fallbacks; callers must provide a registered template. Any
    unknown name (including "none"/"non") raises ValueError to avoid drifting to
    a default BBU prompt.
    """

    if name is None:
        raise ValueError("template name must be provided (no fallback allowed)")

    normalized = name.strip().lower()
    if normalized in {"", "none", "non", "null"}:
        raise ValueError(f"Invalid template name '{name}'; no fallback is permitted")

    def _build_stage_a_summary_prompts(domain: str) -> tuple[str, str]:
        # Lazy import to avoid circular dependencies.
        from src.prompts.stage_a_summary import (
            build_stage_a_system_prompt,
            build_stage_a_user_prompt,
        )

        system_prompt = build_stage_a_system_prompt(domain=domain)
        user_prompt = build_stage_a_user_prompt(USER_PROMPT_SUMMARY, domain=domain)
        return system_prompt, user_prompt

    registry = {
        # Target datasets (BBU, RRU) - unified dense template
        # NOTE: keep legacy alias for backward compatibility (defaults to BBU-flavored dense prompt).
        "target_dense": (
            build_dense_system_prompt(_DEFAULT_JSON_FORMAT, dataset="bbu"),
            USER_PROMPT_JSON,
        ),
        # Target datasets (domain-specific dense prompts)
        "target_dense_bbu": (
            build_dense_system_prompt(_DEFAULT_JSON_FORMAT, dataset="bbu"),
            USER_PROMPT_JSON,
        ),
        "target_dense_rru": (
            build_dense_system_prompt(_DEFAULT_JSON_FORMAT, dataset="rru"),
            USER_PROMPT_JSON,
        ),
        # Source datasets (LVIS, COCO, etc.) - unified dense template
        "source_dense": (SYSTEM_PROMPT_AUX, USER_PROMPT_AUX),
        # Summary mode (legacy alias -> runtime-style BBU prompt)
        "summary": _build_stage_a_summary_prompts("bbu"),
        # Summary mode (domain-focused, aligned with Stage-A inference)
        "summary_bbu": _build_stage_a_summary_prompts("bbu"),
        "summary_rru": _build_stage_a_summary_prompts("rru"),
        # language-only chat
        "chatml": (SYSTEM_PROMPT_CHAT, USER_PROMPT_CHAT),
    }

    try:
        return registry[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown template '{name}'. Registered templates: {available}. No fallback is allowed."
        ) from exc


SYSTEM_PROMPT = SYSTEM_PROMPT_JSON
USER_PROMPT = USER_PROMPT_JSON


__all__ = [
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_JSON",
    "SYSTEM_PROMPT_SUMMARY",
    "SYSTEM_PROMPT_SUMMARY_TRAIN",
    "SYSTEM_PROMPT_SUMMARY_RUNTIME",
    "SYSTEM_PROMPT_CHAT",
    "USER_PROMPT",
    "USER_PROMPT_JSON",
    "USER_PROMPT_SUMMARY",
    "USER_PROMPT_CHAT",
    "SYSTEM_PROMPT_AUX",
    "USER_PROMPT_AUX",
    "get_template_prompts",
    "build_dense_system_prompt",
    "build_summary_system_prompt",
    "build_summary_system_prompt_minimal",
    "DATASET_PRIOR_RULES",
    "DATASET_SCHEMA_HINT",
    "MISSION_SPECIFIC_PRIOR_RULES",
]
