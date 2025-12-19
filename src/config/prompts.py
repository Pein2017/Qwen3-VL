"""Prompt templates for training.

Updated to grouped-JSON format（类型/属性/[条件属性]/[备注]）并支持数据集级别的
schema 提示与先验规则（BBU、RRU），在保持统一结构的同时插拔业务知识。
"""

# ============================================================================
# Dataset-level schema hints & prior rules
# ============================================================================

# 结构提示（精简版）：只给出类型与属性**概念**，不枚举全部取值
DATASET_SCHEMA_HINT = {
    "bbu": (
        "可标注类型示例：\n"
        "- BBU设备：品牌/完整性/挡风板需求\n"
        "- 挡风板：品牌/安装方向\n"
        "- 螺丝、光纤插头：部件类型 + 符合性\n"
        "- 光纤：保护状态 + 弯曲半径\n"
        "- 电线：捆扎状态\n"
        "- 标签：文本内容或无法识别\n"
    ),
    "rru": (
        "可标注类型示例：\n"
        "- 站点距离：单一文本标签\n"
        "- RRU设备\n"
        "- RRU接地端 / 地排接地端螺丝 / 紧固件：安装状态（合格/不合格）\n"
        "- 标签：文本内容\n"
        "- 尾纤：是否带标签 + 是否有套管保护\n"
        "- 接地线：是否带标签\n"
        "- 分组：组N: 标签/… | 尾纤/… | 接地线/…（组内至少2个对象）\n"
    ),
}

# 数据集专属先验：dense/summary 共用
DATASET_PRIOR_RULES = {
    "bbu": (
        "- 可能混杂无关内容（如施工图纸、杂物等），仅识别与任务相关的对象。\n"
        "- 爱立信品牌BBU：不安装挡风板。\n"
        "- 一个华为挡风板对应着一个黄色箭头。如果有两个黄色箭头，那通常意为着存在两个挡风板。"
        "- 仅当存在两台BBU时，在两台BBU之间必须安装挡风板；单台BBU通常无需安装。\n"
        "- 螺丝/挡风板与本体强关联，不会远离对应设备；BBU端光纤插头多为蓝白色且插在 BBU 设备上。\n"
        "- 标签通常为黄色纸质，贴在插头处或电线上。\n"
        "- 光纤对象：若同一根光纤被分段检测到（颜色/路径连续），必须合并为一个对象，不得拆分。\n"
        "- 光纤保护判定：任一片段存在保护套/蛇形管/铠装等防护材料，即整条光纤记为**有保护**；仅当整条光纤均无任何保护时，标注为**无保护措施**。\n"
    ),
    "rru": (
        "- 场景聚焦 RRU 侧设备与接地/尾纤组件，忽略与任务无关的杂物/背景。\n"
        "- 站点距离通常只有一处文本标签，记录原文即可，不推断单位。\n"
        "- RRU接地端、地排接地端螺丝、紧固件仅标**紧固合格/不合格**，不补充品牌。\n"
        "- 尾纤需判断**是否带标签**与**是否有套管保护**；任一片段有保护即算有保护。\n"
        "- 接地线只判断是否带标签。\n"
        "- 分组：标签/尾纤/接地线按组出现；若任一组内对象数 < 2 视为异常样本。分组号需写入 desc 前缀（如**组1: 尾纤/...**）。\n"
        "- 不生成**审核通过/不通过**等工单级字段。\n"
        "检测RRU设备、紧固件、线缆尾纤，RRU接地端、RRU接地线",
        "尾纤的保护方式有两种，黑色的防水胶带或者白色的尼龙套管；若是白色的尼龙套管，标签会贴在白色尼龙管上",
    ),
}

_DEFAULT_DATASET = "bbu"

# ============================================================================
# Mission-specific prior rules (appended to dataset priors)
# ============================================================================
# 任务特定先验规则：在数据集级别先验规则基础上，为特定任务添加额外规则
MISSION_SPECIFIC_PRIOR_RULES: dict[str, str] = {
    "挡风板安装检查": (
        "- 爱立信品牌BBU：不安装挡风板。\n"
        "- 一个华为挡风板对应着一个黄色箭头。如果有两个黄色箭头，那通常意为着存在两个挡风板。\n"
        "- 仅当存在两台BBU时，在两台BBU之间必须安装挡风板；单台BBU通常无需安装。\n"
    ),
}

# ============================================================================
# Prompt Schemes
# ============================================================================

FORMAT_HINTS = {
    "standard": "- JSON 排布：整段单行，逗号和冒号后各保留一个空格，禁止换行。\n",
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
    '你是图像密集标注助手。只输出一个 JSON 对象 {"object_1":{...}}，不要额外文字。\n'
    "- 对象按**自上到下 → 左到右**排序（线以最左端点为起点），编号从 1 递增。\n"
    "  * 排序规则详解：首先按 Y 坐标（纵向）从小到大排列（图像上方优先），Y 坐标相同时按 X 坐标（横向）从小到大排列（图像左方优先）。\n"
    "  * bbox_2d 排序参考点：使用左上角坐标 (x1, y1) 作为该对象的排序位置。\n"
    "  * poly 排序参考点：顶点按质心顺时针排序，起点为最上最左的顶点；使用首顶点 (x1, y1) 作为排序位置。\n"
    "  * line 排序参考点：使用最左端点（X 坐标最小的点）作为排序参考；若多个点的 X 坐标相同，则取其中 Y 坐标最小的点。\n"
    "- 信息不足/不确定时的标注：若仅拍到部分、遮挡、角度/范围过小、品牌或方向无法判断、空间/安装情况不明、缺件/缺失，请输出**<类型>/需复核**，可在备注中补充原因。\n"
    "- 每个对象仅包含 desc + 单个几何键（bbox_2d/poly/line）；禁止多个几何键。\n"
    "- desc 采用**类型/属性[,属性]/[条件属性]**层级，不得包含多余空格或换行。\n"
    "- 坐标使用 norm1000 整数（0..1000）：\n"
    "  * bbox_2d：扁平数组 [x1,y1,x2,y2]（左上、右下）。\n"
    "  * poly：二维数组对 [[x1,y1], [x2,y2], ...]（至少四个顶点，当前使用 4 个点表示四边形，未来可处理更多点）。\n"
    "  * line：二维数组对 [[x1,y1], [x2,y2], ...]（线段端点）；必须包含整数 line_points（≥2），其值等于点对个数。\n"
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
    *, dataset: str | None = None, mission: str | None = None
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

    return (
        """你是图像摘要助手。请始终使用简体中文作答。只返回一行中文摘要文本，不要任何解释或额外符号。

        无关图片输出协议（强规则）：
        - 若图片与任务无关或目标完全不可见：你必须只输出 无关图片（四个字），不得包含任何其它文字/标点/空格/解释。
        - 证明文件、文档、报告、图纸/CAD/平面图/示意图、票据、聊天/手机截图等非现场照片，一律视为无关图片；禁止将此类图片描述为"室内场景/办公室/房间"等。
        - 对于纯图纸/工程施工方案/设备安装示意图/机柜布局平面图等，只要可以判断为纸质或电子图纸（以线条、符号、表格为主），即使其中出现"BBU、机柜、挡风板"等文字或设备草图，也必须按 无关图片 处理，严禁推断为真实拍摄的BBU机房或生成任何"螺丝/挡风板/BBU设备"类 desc。

        证据优先（反幻觉，最高优先级）：
        - 先验规则仅用于**已被视觉证据确认存在**的对象；不得仅凭场景像"机房/室内/设备间"等就生成对象条目。
        - 属性不确定时：保守输出"需复核"类 desc；严禁补全品牌/方向/合格性等不可见属性。
        - 若无法从图像中确认任何与任务相关的目标存在：严格输出 无关图片（仅四个字）。

        输出要求（保持中立、专注"证据罗列"，不直接判定工单通过/不通过）：
        1) 单行摘要覆盖该图片中检测到的所有相关对象，既要包含正向信息，也要完整保留异常/缺失/需复核等负向信息，不得因为大部分对象合格而省略少数问题点。
        2) 使用对象的 desc 原文作为分组键，不拆分或改写；相同 desc 合并为**desc×N**（×为全角乘号、紧贴N不留空格）。
        3) 排序：先按 desc 字数从少到多；若字数相同，保留首次出现的先后顺序。条目之间仅用中文逗号'，'分隔。
        4) desc 中若含**备注: …**保持原样放在该条目内部，不要额外拆出独立备注。
        5) 单句输出：整行不得换行，不要在末尾加句号'。'，不要多余空格或首尾分隔符。
        6) 仅输出摘要文本；不得返回 JSON 或其它包装结构；不要额外生成"通过/不通过/合格/不合格/需整改"等判定性结论，这些由后续判决流程处理。

        决策规则（严格二选一）：
        - 若存在与任务相关的任意目标：输出若干条**desc×N**，条目用中文逗号'，'分隔。
        - 否则：严格输出 无关图片（仅四个字），并且不允许在同一行再出现任何 desc、数字或说明性文字。
        禁止单独输出"无法判断"等含糊短语；若无法识别具体类别，应在 desc 中如实表述（例如"标签/无法识别×1"），而不是整行写"无法判断"。
        在目标被严重遮挡/模糊且无法判断其类型时，不要猜测类别或合格性；若整图均为此类情况，可按 无关图片 处理。

        """
        + schema
        + "先验规则（与密集标注共享的业务知识）：\n"
        + prior
        + mission_prior
        + """

【严格禁止】
- 禁止任何坐标数字（如整数数组）
- 禁止几何字段名（bbox_2d、poly、line）
- 禁止方括号[]包裹的数字列表
- 禁止任何特殊标记或尖括号<>
- 仅输出纯文本摘要，不得包含任何结构化几何信息

"""
    )


def build_summary_system_prompt_minimal() -> str:
    """Minimal summary-mode system prompt for training.

    This prompt intentionally avoids business/domain priors. It only specifies:
    - strict one-line output;
    - the desc×N aggregation format;
    - the exact '无关图片' output contract for irrelevant images.
    """

    return (
        "你是图像摘要助手。请始终使用简体中文作答。只返回一行中文摘要文本，不要任何解释或额外符号。\n\n"
        "【输出格式】\n"
        "- 若图片与任务无关或没有任何可用于任务的目标信息：你必须只输出 无关图片（四个字），不得包含任何其它文字/标点/空格/解释。\n"
        "- 否则：输出若干条 desc×N（×为全角乘号、紧贴N不留空格），条目之间仅用中文逗号'，'分隔；相同 desc 合并并累加 N。\n"
        "- 排序：先按 desc 字数从少到多；字数相同保持首次出现顺序。\n"
        "- 单句输出：整行不得换行，不要在末尾加句号'。'。\n\n"
        "【内容约束】\n"
        "- 只做客观罗列，不输出“通过/不通过/合格/不合格/需整改”等判定性结论。\n"
        "- 不要猜测场景或类别；对文档/报告/图纸/CAD/平面图/示意图/票据/聊天或手机截图等非现场照片，一律输出 无关图片。\n"
        "- 证据优先：只输出你能在图像中直接观察到的对象信息；禁止依据背景环境、常识、文字线索或“像某类场景”来推断并输出对象。\n"
        "- 不确定时要保守：若能确认类型但属性/状态不清晰，优先输出“类型/需复核”类 desc；若连类型都无法确认或整图缺乏目标证据，输出 无关图片。\n"
        "- 禁止输出 JSON、坐标数字、几何字段名（bbox_2d/poly/line）、方括号数字列表或尖括号标记。\n"
    )


# Summary prompts
# - TRAIN: minimal format-only prompt (avoid injecting business priors into the model).
# - RUNTIME: richer prompt with schema + priors for Stage-A inference (default, no mission).
SYSTEM_PROMPT_SUMMARY_TRAIN = build_summary_system_prompt_minimal()
SYSTEM_PROMPT_SUMMARY_RUNTIME = build_summary_system_prompt(dataset="bbu")
# Backward-compat default for training pipelines.
SYSTEM_PROMPT_SUMMARY = SYSTEM_PROMPT_SUMMARY_TRAIN


USER_PROMPT_JSON = (
    "基于所给图片，检测并列出所有对象：按**自上到下再从左到右**排序，"
    "坐标使用 norm1000 整数网格，严格按规范返回 JSON。"
)

USER_PROMPT_SUMMARY = (
    """请对每张图片输出一行简体中文摘要：使用检测到的 desc 原文分组，相同 desc 合并为**desc×N**（×为全角乘号、紧贴N不留空格），按 desc 字数从少到多排序，字数相同保持首次出现顺序，条目之间仅用中文逗号'，'分隔。"""
    """保持 desc 原样（包含备注时也写在该条目内，不额外拆分），既不要遗漏异常/需复核类 desc，也不要在摘要中给出“通过/不通过/合格/不合格”等判定性结论；摘要只做客观罗列。不得换行或添加句号。只返回摘要文本，不要坐标、几何字段或解释。若图片无关/非现场照片，只输出 无关图片。请始终使用简体中文作答。"""
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

    registry = {
        # Target datasets (BBU, RRU) - unified dense template
        "target_dense": (
            build_dense_system_prompt(_DEFAULT_JSON_FORMAT, dataset=None),
            USER_PROMPT_JSON,
        ),
        # Source datasets (LVIS, COCO, etc.) - unified dense template
        "source_dense": (SYSTEM_PROMPT_AUX, USER_PROMPT_AUX),
        # Summary mode (unified for all datasets)
        "summary": (
            SYSTEM_PROMPT_SUMMARY_TRAIN,
            USER_PROMPT_SUMMARY,
        ),
        # language-only chat
        "chatml": (SYSTEM_PROMPT_CHAT, USER_PROMPT_CHAT),
    }

    try:
        return registry[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown template '{name}'. Registered templates: {available}. "
            "No fallback is allowed."
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
