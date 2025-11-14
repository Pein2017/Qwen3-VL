## 背景与目标

通信工程竣工验收场景中，我们以 AI 辅助人工坐席进行工单审核。输入为一组与某个“任务/条目（mission）”相关的图片，输出为组级别的“审核通过/审核不通过”以及规范化的理由说明。整体采用“两阶段”流程：
- Stage-A：逐图片的客观事实归纳（摘要）
- Stage-B：基于多图摘要与业务规则的组级判定与反思更新

业务目标：
- 最大化与历史人工审核一致性（主指标：Label Match）
- 控制“人工判不通过而 AI 判通过”的比例 < 5%（硬约束）
- 产出可审计的决策与规则演进链路（治理友好）

---

## 数据与属性摘要（原始 detection 到生产摘要的一致性）

原始标注采用“分层属性摘要”，关键类型与层级见 `data_conversion/hierarchical_attribute_mapping.json`（下简称“属性映射”）。要点：
- 对象类型：`bbu`（BBU设备）、`bbu_shield`（挡风板）、`connect_point`（螺丝/光纤插头）、`label`（标签）、`fiber`（光纤，线对象）、`wire`（电线，线对象）
- 几何约束：硬件采用 `quad/bbox_2d`，线缆采用 `line`；训练与推理链路保持几何与文本同步
- 层级语义：使用“/”分隔属性层级；条件属性仅在父属性满足时出现；`备注` 为自由文本备注

生产摘要（Stage-A与Stage-B共用）的格式规范见 `docs/DATA_AND_DATASETS.md`：
- 单行中文汇总；对象条目用中文逗号分隔，计数用“×N”
- 条目内按“类型/属性[,属性]/[条件属性]”组织，备注统一以“，备注: ...”结尾
- 自上而下、左到右的视觉顺序；无坐标数组

保持“属性映射”与“摘要规范”的一致，是保证训练数据、Stage-A 摘要与 Stage-B 判定互相兼容的基础。

---

## 训练过程简述（Dense → Summary）

我们已完成基于 `dense captioning` 的训练，模型能够：
- 准确定位关键物体并输出类型/属性/备注的层级结构
- 在 `summary` 模式下弱化定位能力，面向生产输出要求的“单行摘要”

训练与数据要点（详见 `docs/DATA_AND_DATASETS.md` 与 `docs/REFERENCE.md`）：
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
2) Rollout：按解码网格（温度、top_p、max_new_tokens 等）对每组生成多个候选判定
3) Signals：对候选进行确定性打分（标签一致性、任务聚焦一致性、启发式置信度、自洽性等）；CriticEngine 对每个候选生成结构化评述（摘要与评述）
4) Selection：依据策略选择最终判定（如 top_semantic，平分时按置信度/温度打破平局）
5) Reflection：批处理近期轨迹，分析 wins/losses 与 GT 差异，融合 CriticEngine 评述，提出结构化 guidance 变更提案
6) Guidance Repository：落库与快照；通过留存与回滚策略保障可治理性

产物与目录结构（概览）：
- selections.jsonl：最终选择的判定结果与分数
- trajectories.jsonl：候选级别的完整生成与评分轨迹
- guidance.json + snapshots/：当前生效 guidance 与每次变更的快照
- reflection.jsonl：反思提案、合并/拒绝、影响评估与 KPI 记录

策略与信号（示例）：
- label_match（与历史人工标签一致性）
- focus_consistency（是否紧扣 mission 关注点）
- summary_confidence（摘要质量启发式：是否出现关键要素、是否存在互相矛盾的条目）
- semantic_advantage（相对优势分，综合理由质量与一致性）
- needs_recheck / label_contradiction（用于人工加审阈值）
- critic_summary / critic_critique（CriticEngine 生成的结构化评述，用于反思阶段的决策增强）

---

## 生产推理与运行建议

运行入口与配置：
- 通过 `scripts/stage_b_run.sh` 或 `python -m src.stage_b.runner --config /abs/path/to/config.yaml --log-level debug` 运行
- 关键配置项（见 `docs/REFERENCE.md`“Stage-B”与仓库 `configs/stage_b/`）：
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

将“先验要求”具体化为可执行检查（基于 Stage-A 摘要）：
- BBU安装（正装）：必须同时出现“BBU设备/显示完整”与“螺丝、光纤插头/BBU安装螺丝/显示完整/符合要求”；若摘要出现“不符合要求/未拧紧/露铜/复接/生锈”，则直接“不通过”
- 接地线检查：需出现“机柜处接地螺丝/显示完整/符合要求”与“地排处接地螺丝/显示完整/符合要求”；且“电线/捆扎整齐”；若任一缺失或不合规，则“不通过”
- 线缆布放：需出现“BBU端光纤插头/显示完整/符合要求”与“ODF端光纤插头/显示完整/符合要求”；出现“光纤/有保护措施/（蛇形管|铠装|同时有蛇形管和铠装）/弯曲半径合理”；否则“不通过”
- 挡风板安装：若“BBU设备/机柜空间充足需要安装”，则必须“挡风板/显示完整/安装方向正确”或“BBU设备/按要求配备挡风板”；否则“不通过”；若“无需安装”，以其他要点决定

特殊情况处理：
- `备注` 作为 override 线索（如：施工临时方案/勘误说明），可触发“需要人工复核”而非直接通过

---

## 风险、边界与质量控制

- 双重噪声：Stage-A 与人工标签均可能有偏；通过多候选 + 确定性信号 + 反思提案缓解
- 误放行控制：对高风险 mission 设置更高的选择阈值或强制人工复核
- 指标退化监控：持续追踪 semantic_advantage 与 confidence 分布，异常时冻结 guidance 变更并回滚到上个快照
- 词表与格式漂移：严守 `DATA_AND_DATASETS.md` 摘要规范；新增检查点须同步更新属性映射与摘要生成

---

## 术语对照

- Mission：任务/条目（如挡风板安装检查）
- Guidance（经验库）：供模型读取的编号规则片段（[G0]、[G1]…）
- Reflection（反思）：基于近期表现自动提出的规则增量修改
- Verdict（判定）：组级“通过/不通过”与理由
- Stage-A 摘要：每图单行事实归纳，Stage-B 的事实基础

---

## 附：与仓库文档的映射关系

- 属性与层级：`data_conversion/hierarchical_attribute_mapping.json`
- 数据与摘要规范：`docs/DATA_AND_DATASETS.md`
- 训练与推理参考：`docs/REFERENCE.md`
- 业务背景与双阶段关系：`docs/STAGE_A_STAGE_B.md`