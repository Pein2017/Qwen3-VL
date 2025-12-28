## 背景与目标

通信工程竣工验收场景中，我们以 AI 辅助人工坐席进行工单审核。输入为一组与某个“任务/条目（mission）”相关的图片，输出为组级别的“审核通过/审核不通过”以及规范化的理由说明。整体采用“两阶段”流程：
- Stage-A：逐图片的客观事实归纳（摘要）
- Stage-B：基于多图摘要与业务规则的组级判定与反思更新

Stage-B 现为“prompt-only”流程：推理输出**严格两行二分类**（`Verdict: 通过|不通过` + `Reason: ...`），且最终输出中**禁止任何第三状态词面**（例如“需复核/证据不足/待定/need-review”等）。

异常与人工复核的边界：
- Stage‑B 仅运行 rule-search；解析失败的候选会在统计时视为无效样本，不再写入 need-review 队列。
- `manual_review_queue.jsonl` 已移除（不再生成）。

业务目标：
- 最大化与历史人工审核一致性（主指标：Label Match）
- 控制“人工判不通过而 AI 判通过”的比例 < 5%（硬约束）
- 产出可审计的决策与规则演进链路（治理友好）

---

## 数据与属性摘要（原始 detection 到生产摘要的一致性）

原始标注采用“分层属性摘要”，关键类型与层级见 `data_conversion/hierarchical_attribute_mapping.json`（下简称“属性映射”）。要点：
- 对象类型：`bbu`（BBU设备）、`bbu_shield`（挡风板）、`connect_point`（螺丝/光纤插头）、`label`（标签）、`fiber`（光纤，线对象）、`wire`（电线，线对象）
- 几何约束：硬件采用 `poly/bbox_2d`，线缆采用 `line`；训练与推理链路保持几何与文本同步
- 描述语义：desc 使用 `key=value` 逗号分隔；条件属性仅在父属性满足时出现；`备注` 为自由文本备注

生产摘要（Stage-A与Stage-B共用）的格式规范见 `docs/data/DATA_AND_DATASETS.md`：
- 单行 JSON 字符串；包含 `dataset/objects_total/统计`，`异常` 仅在非零时出现（BBU 额外含 `备注`，RRU 可含 `分组统计`）
- `统计` 为类别+属性值计数（`{value: count}`），仅统计可见值，不输出缺失/需复核/遮挡
- 无坐标数组；OCR 保留原文（去空格，保留 `-`/`/`），不可读写 `可读性=不可读`

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
{"group_id": "QC-TEMP-20241206-0015502", "mission": "挡风板安装检查", "label": "fail", "images": ["QC-TEMP-20241206-0015502_4348975.jpeg"], "per_image": {"image_1": "{\"dataset\": \"BBU\", \"objects_total\": 4, \"统计\": [{\"类别\": \"BBU设备\", \"品牌\": {\"华为\": 2}, \"可见性\": {\"完整\": 2}, \"挡风板需求\": {\"未按要求配备\": 2}}, {\"类别\": \"BBU安装螺丝\", \"符合性\": {\"符合\": 3}}, {\"类别\": \"光纤\", \"保护措施\": {\"有保护\": 2}}, {\"类别\": \"挡风板\", \"安装方向\": {\"正确\": 1}}], \"备注\": [\"无法判断品牌和是否足够空间安装挡风板\"]}"}}
```

结合当前 `bbu_full_768_poly` 词表，可以大致把摘要里的对象理解为“类别 + 属性值计数”的模式（示意，非穷举）：
- BBU 设备本体：`{"类别":"BBU设备","品牌":{...},"可见性":{...},"挡风板需求":{...}}`；`备注` 通常说明拍摄是否完整、机柜空间是否可判定、品牌是否可辨认等。
- 挡风板：`{"类别":"挡风板","品牌":{...},"安装方向":{...}}`。
- 螺丝/插头：`{"类别":"BBU安装螺丝","符合性":{...}}`、`{"类别":"BBU端光纤插头","符合性":{...}}`、`{"类别":"ODF端光纤插头","符合性":{...}}`。
- 接地相关：`{"类别":"机柜处接地螺丝","安装状态":{...}}` 与 `{"类别":"地排处接地螺丝","安装状态":{...}}`。
- 线缆布放：`{"类别":"光纤","保护措施":{...},"弯曲半径":{...}}`。
- 电线捆扎：`{"类别":"电线","捆扎":{...}}`。
- 标签与 OCR：可读时使用 `文本` 统计，不可读使用 `可读性=不可读`；不会出现“可以识别/无法识别”归并。

先验业务要求（简化版）：
- BBU安装方式检查（正装）：至少检测到 BBU 设备与 BBU 安装螺丝，且“符合”
- BBU接地线检查：至少检测到机柜处接地螺丝与地排处接地螺丝且“符合”，并检测到电线且“整齐”
- BBU线缆布放要求：至少检测到 BBU 端与 ODF 端光纤插头且“符合”，并检测到光纤“有保护/半径合理”
- 挡风板安装检查：识别 BBU 设备并判断是否需要挡风板；若需要，则判定是否“按要求配备/方向正确”

备注：真实生产中需结合 `备注` 等自由文本进行灵活解释。

### 多图互补先验（数据常见：挡风板安装检查 / BBU安装方式检查（正装））
- 该类任务通常需要“全局 + 局部特写”多图才能覆盖关键点；若仅单图且无法覆盖 G0 关键要点，倾向判不通过（由模型基于证据决定，而非硬编码护栏）。
- 全局图通常对应 Stage‑A 摘要中识别到的物体数量最多（由 `objects_total` 得出）；Stage‑B user prompt 会显式提供 `ImageN(obj=...)` 统计，帮助推断全局候选。
- 判定应以“全局图确认全局要点 + 局部图确认细节”为原则：局部图因视角不足出现“部分/无法判断”不应直接否决已由全局图明确确认的结论（除非局部图出现明确负项触发词）。

---

## Stage-A 流程（已完成）

职责与产物：
- 使用“summary”能力对每张图片进行事实性归纳，输出每图单行摘要
- 产出一份组级 JSONL：包含 `group_id / mission / label（历史人工）/ images / per_image` 的摘要字典
- 作为 Stage-B 的唯一上游输入（事实来源），也是审核可追溯的依据
- Prompt profiles：训练与推理均使用 **summary_runtime**（system prompt = summary 任务基座 + 全局“非现场/图纸”规则；user prompt = 摘要指令 + BBU/RRU 场景提示块 + 任务重点）。

质量注意点：
- 摘要存在识别偏差（漏检/误检/幻觉），但统一格式与词表可显著降低 Stage-B 的解释难度

---

## Stage-B：业务需求与架构拆解（训练后优化，训练自由）

问题本质：
- 双重噪声：Stage-A 摘要有偏，历史人工标签亦有小概率偏差
- 数据量有限：直接 RL 往往不稳定且成本较高

设计选择：采用“training-free GRPO 风格”的 Prompt 优化与反思循环，维护一份“经验/知识库（guidance）”，由同一个底座模型（Qwen3-VL-4B-Instruct）在推理时读取，持续提升判定一致性。
Stage‑B system prompt 不再包含领域提示；领域知识上移到 Stage‑A user prompt（BBU/RRU 场景提示块）。guidance 与 Stage‑A 摘要仅出现在 Stage‑B user prompt。

核心需求（业务视角）：
- 输入：多图摘要 + 任务/条目 + 当前 guidance + 历史人工 label
- 输出：组级“通过/不通过”与可读理由；并在反思环节提议对 guidance 的增量修改
- 约束：误放行率（人工不通过且 AI 通过）< 5%；理由需与摘要要点对齐，便于审核
- 审计：每次 guidance 变更需记录证据（ticket_key=`{group_id}::{gt_label}`）、反思 ID、适用范围与变更理由

高层架构（运行时）：
1) Ingest：读取 Stage-A JSONL，规范化成组级工单（GroupTicket）
2) Rollout：system prompt 固定为两行判决契约 + 规则/软硬信号；user prompt 仅包含 guidance（S*/G0/G*）+ Stage‑A 摘要（领域提示已上移到 Stage‑A user prompt）。按解码网格（温度、top_p、max_new_tokens 等）对每组生成候选判定，每个候选都遵守两行输出协议（Verdict/Reason）。规则默认 AND（S* 与 G* 必须同时满足；仅当规则文本明确写“或/例外条件”才允许 OR），缺证据即判不通过；超长提示直接 drop，不截断。
3) Signals：为候选附加确定性信号（如与历史标签的一致性 `label_match`、在候选集合中的自洽度 self_consistency 等），不再调用 CriticEngine。
4) Selection：基于多数表决 + fail-first 策略选择最终判定；并叠加 **mission-scoped fail-first** 确定性护栏（仅当负项与当前 mission 的 `G0` 相关时触发整组不通过；支持 pattern-first `不符合/<issue>`）。若护栏覆盖采样 verdict，则必须重写最终 `Reason` 以与最终 `Verdict` 一致，且仍不得出现第三状态词面。
5) Reflection：对近期包含标签冲突或部分正确的轨迹进行批处理，比较模型判定与 GT 差异，构造 JSON-only 的 guidance 变更提案（增加/更新/删除经验规则），并先输出可证伪 hypothesis 进入候选池，跨批次证据累积后再晋升为 `G*` 规则。
6) Guidance Repository：对提案进行应用/留存，并以快照形式记录每次变更，支持回滚与再训练使用。
- rule_search 模式：在少量错例上提出 1–N 条候选操作（upsert/update/merge/remove），并对 update/merge/remove 施加 fp/acc 改善与 `max_fp_rate_increase` 约束；train pool gate 通过后才纳入 guidance，eval pool 指标仅用于审计记录。
- LLM 输出协议：Rollout 使用两行判定格式（`Verdict:` / `Reason:`）；rule_search proposer 输出 JSON 对象，包含候选操作列表 `operations`（或兼容 `rules`）。

产物与目录结构（概览）：
- guidance.json + snapshots/：当前生效 guidance 及每次变更的快照
- rule_candidates.jsonl：候选操作与 gate 指标
- benchmarks.jsonl：被接受规则的 train/eval 指标与元数据
- rule_search_hard_cases.jsonl：错例溯源
- rule_search_candidate_regressions.jsonl：回归样本
- distill_chatml.jsonl：rule-search 早停后可选导出（ChatML）

---

## 生产推理与运行建议

运行入口与配置：
- 通过 `bash scripts/stage_b.sh` 或 `python -m src.stage_b.runner --config /abs/path/to/config.yaml --log-level debug` 运行
- 关键配置项（见 `docs/training/REFERENCE.md` “Stage-B” 与仓库 `configs/stage_b/`）：
  - stage_a_paths：Stage-A JSONL 路径列表
  - model：`model_name_or_path / torch_dtype / device_map`
  - rule_search：proposer 提示、train/eval pool、gate 参数、train/eval sampler
  - reflection：proposer prompt 所需配置（decision/ops prompt 仍要求，但不执行）
  - output：根目录与 run_name；按 mission 自动划分子目录

运营守则：
- 高优先 mission：持续跑 Stage-A，Stage-B 小批量高频反思（如每小时 32 组），严格控制误放行 < 5%
- 低频 mission：按需运行，反思提案需人工审批后合并
- 指标看板：Label Match、误放行率、置信度分布、Holdout 前后 uplift、Guidance 变更频率

---

## 判定规则示例（对齐属性映射）

> 说明：以下示例用于提示 LLM 聚焦的关键对象，不再作为硬编码规则执行。Stage-B 以 prompt + guidance 为主，历史标签只作为冲突标记与反思信号；推理阶段**只输出“通过/不通过”两类**，不输出任何第三状态。`备注` 与“无法确认/模糊/部分”等属于软信号：不得作为硬触发词直接否决，但若无法从摘要/备注/多图证据给出覆盖当前 mission `G0` 关键点的通过证据，则按安全约束判为“不通过”。

将“先验要求”具体化为可执行检查（基于 Stage-A 摘要）：
- BBU安装（正装）：必须同时出现“BBU设备/完整”与“螺丝、光纤插头/BBU安装螺丝/完整/符合”；若摘要出现“不符合/未拧紧/露铜/复接/生锈”，则直接“不通过”
- 接地线检查：需出现“机柜处接地螺丝/完整/符合”与“地排处接地螺丝/完整/符合”；且“电线/整齐”；若任一缺失或不合规，则“不通过”
- 线缆布放：需出现“BBU端光纤插头/完整/符合”与“ODF端光纤插头/完整/符合”；出现“光纤/有保护/（蛇形管|铠装|同时有蛇形管和铠装）/半径合理”；否则“不通过”
- 挡风板安装：若“BBU设备/空间充足需安装”，则必须“挡风板/完整/方向正确”或“BBU设备/按要求配备挡风板”；否则“不通过”；若“免装”，以其他要点决定

特殊情况处理：
- Stage-A 摘要中出现 `备注` 时，判定以备注与多图证据为准，仍需输出“通过/不通过”。
- 遇到“无法判断/模糊/遮挡/部分”等不确定表述时，不作为硬 fail-first 触发词；但若其影响到当前 mission `G0` 的关键要素，且其它图片/备注无法提供支持通过的证据，则判为“不通过”，并在 Reason 中明确说明缺失的关键证据。

---

## 风险、边界与质量控制

- 双重噪声：Stage-A 与人工标签均可能有偏；通过多候选 + 确定性信号 + 反思提案缓解
- 误放行控制：通过 **mission-scoped fail-first**（负项需与当前 `G0` 相关才触发）+ 安全约束（无通过证据则不通过）+ 多候选多数表决，把“人工 fail 而 AI pass”的风险压到最低；冲突样本通过 `conflict_flag` 路由给反思与排查，而不是简单用标签覆盖模型判定
- 冲突/不确定反馈：`conflict_flag` 表示与历史标签矛盾；反思阶段默认尝试从错例中学习可泛化规则。
- 指标退化监控：持续追踪 Label Match、误放行率、verdict/Reason 分布，异常时冻结 guidance 变更并回滚到最近的稳定快照
- 词表与格式漂移：严守 `DATA_AND_DATASETS.md` 摘要规范；新增检查点须同步更新属性映射与摘要生成

---

## 术语对照

- Mission：任务/条目（如挡风板安装检查）
- Guidance（经验库）：供模型读取的编号规则片段（[S1]… 为结构不变量只读；[G0] 为检查清单且必须存在；[G1+] 为可学习规则）
- Reflection（反思）：基于近期表现自动提出的规则增量修改，运行时采用“三段式 JSON-only”（summary→critique→batch update），仅对冲突/部分正确批次生效，产物落盘可复用
- Verdict（判定）：组级“通过/不通过”与理由
- Stage-A 摘要：每图单行事实归纳，Stage-B 的事实基础

---

## 附：与仓库文档的映射关系

- 属性与层级：`data_conversion/hierarchical_attribute_mapping.json`
- 数据与摘要规范：`docs/data/DATA_AND_DATASETS.md`
- 训练与推理参考：`docs/training/REFERENCE.md`
- 业务背景与双阶段关系：`docs/runtime/STAGE_A_STAGE_B.md`
