## 背景与目标

通信工程竣工验收场景中，我们以 AI 辅助人工坐席进行工单审核。输入为一组与某个“任务/条目（mission）”相关的图片，输出为组级别的“审核通过/审核不通过”以及规范化的理由说明。整体采用“两阶段”流程：
- Stage-A：逐图片的客观事实归纳（摘要）
- Stage-B：基于多图摘要与业务规则的组级判定与反思更新

Stage-B 现为“prompt-only”流程：模型输出固定两行（Verdict/Reason），不再运行 CriticEngine，也不使用证据数组或置信度/自洽度信号；无法解析的样本会进入 `manual_review_queue.jsonl`，解析失败会写入 `failure_malformed.jsonl` 以便改进提示。

业务目标：
- 最大化与历史人工审核一致性（主指标：Label Match）
- 控制“人工判不通过而 AI 判通过”的比例 < 5%（硬约束）
- 产出可审计的决策与规则演进链路（治理友好）

---

## 数据与属性摘要（原始 detection 到生产摘要的一致性）

原始标注采用“分层属性摘要”，关键类型与层级见 `data_conversion/hierarchical_attribute_mapping.json`（下简称“属性映射”）。要点：
- 对象类型：`bbu`（BBU设备）、`bbu_shield`（挡风板）、`connect_point`（螺丝/光纤插头）、`label`（标签）、`fiber`（光纤，线对象）、`wire`（电线，线对象）
- 几何约束：硬件采用 `poly/bbox_2d`，线缆采用 `line`；训练与推理链路保持几何与文本同步
- 层级语义：使用“/”分隔属性层级；条件属性仅在父属性满足时出现；`备注` 为自由文本备注

生产摘要（Stage-A与Stage-B共用）的格式规范见 `docs/data/DATA_AND_DATASETS.md`：
- 单行中文汇总；对象条目用中文逗号分隔，计数用“×N”
- 条目内按“类型/属性[,属性]/[条件属性]”组织，备注统一以“，备注: ...”结尾
- 自上而下、左到右的视觉顺序；无坐标数组

保持“属性映射”与“摘要规范”的一致，是保证训练数据、Stage-A 摘要与 Stage-B 判定互相兼容的基础。

---

## 训练过程简述（Dense → Summary）

我们已完成基于 `dense captioning` 的训练，模型能够：
- 准确定位关键物体并输出类型/属性/备注的层级结构
- 在 `summary` 模式下弱化定位能力，面向生产输出要求的“单行摘要”

训练与数据要点（详见 `docs/data/DATA_AND_DATASETS.md` 与 `docs/training/REFERENCE.md`）：
- JSONL 记录包含图片、对象几何与可选的 `summary` 字段；几何在磁盘保持像素坐标，模板在编码时归一化
- 必备占位配置：`data.dataset: ["dummy"]` 用于 ms‑swift 初始化校验；实际训练数据通过 `custom.train_jsonl` 提供
- 模板自动插入视觉占位符，打包与增广均为几何感知，确保图文对齐

---

## 已有数据与场景范围

当前聚焦 4 个任务（mission）：
- 挡风板安装检查
- BBU 接地线检查
- BBU 线缆布放要求
- BBU 安装方式检查（正装）

对应的 Stage-A 输出示例（位于 `./output_post/stage_a`）：
```json
{"group_id": "QC-TEMP-20241206-0015502", "mission": "挡风板安装检查", "label": "fail", "images": ["QC-TEMP-20241206-0015502_4348975.jpeg"], "per_image": {"image_1": "BBU设备/华为/显示完整/这个BBU设备未按要求配备挡风板×2，螺丝、光纤插头/BBU安装螺丝/显示完整/符合要求×3，螺丝、光纤插头/BBU端光纤插头/显示完整/符合要求×6，光纤/有保护措施/蛇形管/弯曲半径合理×2，标签/可以识别×7，标签/无法识别×2，挡风板/显示完整/安装方向正确×1，备注: 无法判断品牌和是否足够空间安装挡风板"}}
```

结合当前 `bbu_full_768_poly` 词表，可以大致把摘要里的对象与短语理解为几类模式（示意，非穷举）：
- BBU 设备本体：`BBU设备/{品牌},显示完整|只显示部分,(无需安装|机柜空间充足需要安装)/这个BBU设备(按要求配备了挡风板|未按要求配备挡风板)`，`备注:` 通常说明拍摄是否完整、机柜空间是否可判定、品牌是否可辨认等。
- 挡风板：`挡风板/{品牌},显示完整|只显示部分,安装方向正确|安装方向错误`，`备注:` 中常出现“未拍全/范围过小/无法判断品牌/无法判断安装方向是否正确”等表述。
- 接地相关：`螺丝、光纤插头/机柜处接地螺丝,符合要求`、`螺丝、光纤插头/地排处接地螺丝,符合要求`，`备注:` 里可能提示“背面疑似连接多条电线”等潜在风险。
- 线缆布放：`光纤/有保护措施,弯曲半径合理/蛇形管`、`光纤/无保护措施,弯曲半径不合理(...)`，以及“部分未套蛇形管/未显示弯曲情况”等局部问题。
- 电线捆扎：`电线/捆扎整齐` vs `电线/分布散乱`，直接对应线缆布放任务中的“整齐/散乱”判断。
- 标签与 OCR：原始标签文本在摘要阶段被归一为 `标签/可以识别` 与 `标签/无法识别` 两种状态，具体站点名、设备标识等不直接出现在 Stage-B 的输入中。

先验业务要求（简化版）：
- BBU安装方式检查（正装）：至少检测到 BBU 设备与 BBU 安装螺丝，且“符合要求”
- BBU接地线检查：至少检测到机柜处接地螺丝与地排处接地螺丝且“符合要求”，并检测到电线且“捆扎整齐”
- BBU线缆布放要求：至少检测到 BBU 端与 ODF 端光纤插头且“符合要求”，并检测到光纤“有保护措施/弯曲半径合理”
- 挡风板安装检查：识别 BBU 设备并判断是否需要挡风板；若需要，则判定是否“按要求配备/安装方向正确”

备注：真实生产中需结合 `备注` 等自由文本进行灵活解释。

---

## Stage-A 流程（已完成）

职责与产物：
- 使用“summary”能力对每张图片进行事实性归纳，输出每图单行摘要
- 产出一份组级 JSONL：包含 `group_id / mission / label（历史人工）/ images / per_image` 的摘要字典
- 作为 Stage-B 的唯一上游输入（事实来源），也是审核可追溯的依据

质量注意点：
- 摘要存在识别偏差（漏检/误检/幻觉），但统一格式与词表可显著降低 Stage-B 的解释难度

---

## Stage-B：业务需求与架构拆解（训练后优化，训练自由）

问题本质：
- 双重噪声：Stage-A 摘要有偏，历史人工标签亦有小概率偏差
- 数据量有限：直接 RL 往往不稳定且成本较高

设计选择：采用“training-free GRPO 风格”的 Prompt 优化与反思循环，维护一份“经验/知识库（guidance）”，由同一个底座模型（Qwen3-VL-4B-Instruct）在推理时读取，持续提升判定一致性。

核心需求（业务视角）：
- 输入：多图摘要 + 任务/条目 + 当前 guidance + 历史人工 label
- 输出：组级“通过/不通过”与可读理由；并在反思环节提议对 guidance 的增量修改
- 约束：误放行率（人工不通过且 AI 通过）< 5%；理由需与摘要要点对齐，便于审核
- 审计：每次 guidance 变更需记录证据（group_id）、反思 ID、适用范围与变更理由

高层架构（运行时）：
1) Ingest：读取 Stage-A JSONL，规范化成组级工单（GroupTicket）
2) Rollout：按解码网格（温度、top_p、max_new_tokens 等）对每组生成一个或多个候选判定，每个候选都遵守两行输出协议（Verdict/Reason）。
3) Signals：为候选附加确定性信号（如与历史标签的一致性 `label_match`、在候选集合中的自洽度 self_consistency 等），不再调用 CriticEngine。
4) Selection：基于多数表决 + fail-first 策略选择最终判定（多数票若打平，优先选择 verdict=`fail` 的候选；再按温度和候选索引打破平局）。
5) Reflection：对近期包含标签冲突或部分正确的轨迹进行批处理，比较模型判定与 GT 差异，构造 JSON-only 的 guidance 变更提案（增加/更新/删除经验规则）。
6) Guidance Repository：对提案进行应用/留存，并以快照形式记录每次变更，支持回滚与再训练使用。
- LLM 输出协议：Rollout 使用两行判定格式（`Verdict:` / `Reason:`）；Reflection 输出合法 JSON 数组，每个元素代表一条规则操作（add/update/delete/none）。

产物与目录结构（概览）：
- selections.jsonl：最终选择的判定结果及其元数据（vote_strength、label_match、conflict_flag 等）
- trajectories.jsonl：候选级别的完整生成与解析轨迹（含解码温度、候选索引、解析是否成功等）
- guidance.json + snapshots/：当前生效 guidance 及每次变更的快照
- reflection.jsonl：反思提案、是否应用、关联样本 ID 与指导变更记录

策略与信号（当前实现）：
- label_match：候选 verdict 与历史人工标签是否一致
- self_consistency：在同一组候选中，某一 verdict 出现的比例（启发式一致性）
- vote_strength：最终被选 verdict 在候选集合中的票数占比
- conflict_flag：最终选择与标签明显矛盾时为 true，用于后续反思和人工排查

---

## 生产推理与运行建议

运行入口与配置：
- 通过 `scripts/stage_b_run.sh` 或 `python -m src.stage_b.runner --config /abs/path/to/config.yaml --log-level debug` 运行
- 关键配置项（见 `docs/training/REFERENCE.md` “Stage-B” 与仓库 `configs/stage_b/`）：
  - stage_a_paths：Stage-A JSONL 路径列表
  - model：`model_name_or_path / torch_dtype / device_map`
  - sampler：解码网格（temperature / top_p / max_new_tokens / samples_per_decode）
  - signals / selection：开关与权重；tie-break 策略
  - reflection：prompt 模板、批大小、变更上限、长度上限、快照保留
  - output：根目录与 run_name；按 mission 自动划分子目录

运营守则：
- 高优先 mission：持续跑 Stage-A，Stage-B 小批量高频反思（如每小时 32 组），严格控制误放行 < 5%
- 低频 mission：按需运行，反思提案需人工审批后合并
- 指标看板：Label Match、误放行率、置信度分布、Holdout 前后 uplift、Guidance 变更频率

---

## 判定规则示例（对齐属性映射）

> 说明：以下示例用于提示 LLM 聚焦的关键对象，不再作为硬编码规则执行。Stage-B 以 prompt + guidance 为主，历史标签只作为冲突标记和反思信号，不做硬兜底。当前对挡风板等高风险任务采用“只要存在明显问题或无法确认关键要素，就判为不通过”的保守策略；极端矛盾或难以解释的样本会写入 `manual_review_queue.jsonl` 供人工复核。

将“先验要求”具体化为可执行检查（基于 Stage-A 摘要）：
- BBU安装（正装）：必须同时出现“BBU设备/显示完整”与“螺丝、光纤插头/BBU安装螺丝/显示完整/符合要求”；若摘要出现“不符合要求/未拧紧/露铜/复接/生锈”，则直接“不通过”
- 接地线检查：需出现“机柜处接地螺丝/显示完整/符合要求”与“地排处接地螺丝/显示完整/符合要求”；且“电线/捆扎整齐”；若任一缺失或不合规，则“不通过”
- 线缆布放：需出现“BBU端光纤插头/显示完整/符合要求”与“ODF端光纤插头/显示完整/符合要求”；出现“光纤/有保护措施/（蛇形管|铠装|同时有蛇形管和铠装）/弯曲半径合理”；否则“不通过”
- 挡风板安装：若“BBU设备/机柜空间充足需要安装”，则必须“挡风板/显示完整/安装方向正确”或“BBU设备/按要求配备挡风板”；否则“不通过”；若“无需安装”，以其他要点决定

特殊情况处理：
- `备注` 作为 override 线索（如：施工临时方案/勘误说明），可触发“需要人工复核”而非直接通过
- 遇到“无法判断/模糊/遮挡/只显示部分”等描述且影响到任务关注的关键要素（如挡风板是否配备、接地线是否存在、线缆是否有保护等）时，当前策略是直接判为“不通过”，在 Reason 中明确写出“关键要素无法确认”的原因；仅在反思环节发现“完全找不到支持标签的证据”或其它异常情况时，才将样本写入 `manual_review_queue.jsonl` 交由人工复核

---

## 风险、边界与质量控制

- 双重噪声：Stage-A 与人工标签均可能有偏；通过多候选 + 确定性信号 + 反思提案缓解
- 误放行控制：通过保守的判定 prompt（关键要素只要存在问题或无法确认即判为“不通过”）+ fail-first 选择策略，把“人工 fail 而 AI pass”的风险压到最低；冲突样本通过 `conflict_flag` 与 `manual_review_queue.jsonl` 路由给人工排查，而不是简单用标签覆盖模型判定
- 冲突/不确定反馈：`conflict_flag` 表示与历史标签矛盾；需要人工介入的样本会追加到 `manual_review_queue.jsonl` 并标记原因（如 `no_evidence_for_label`、`no_support_after_reflection` 等），供质检同学重点复核
- 指标退化监控：持续追踪 Label Match、误放行率、verdict/Reason 分布，异常时冻结 guidance 变更并回滚到最近的稳定快照
- 词表与格式漂移：严守 `DATA_AND_DATASETS.md` 摘要规范；新增检查点须同步更新属性映射与摘要生成

---

## 术语对照

- Mission：任务/条目（如挡风板安装检查）
- Guidance（经验库）：供模型读取的编号规则片段（[G0]、[G1]…）
- Reflection（反思）：基于近期表现自动提出的规则增量修改，运行时采用“三段式 JSON-only”（summary→critique→batch update），仅对冲突/部分正确批次生效，产物落盘可复用
- Verdict（判定）：组级“通过/不通过”与理由
- Stage-A 摘要：每图单行事实归纳，Stage-B 的事实基础

---

## 附：与仓库文档的映射关系

- 属性与层级：`data_conversion/hierarchical_attribute_mapping.json`
- 数据与摘要规范：`docs/data/DATA_AND_DATASETS.md`
- 训练与推理参考：`docs/training/REFERENCE.md`
- 业务背景与双阶段关系：`docs/runtime/STAGE_A_STAGE_B.md`
