---
name: stageb-guidance-audit
description: 分析 Qwen3-VL Stage-A 摘要 JSONL 和 Stage-B initial_guidance.json，推导规则、识别噪声标签，并优化 Stage-B 指导规则以提升判决准确率，同时保持 Stage-A 摘要不变。当需要手动审核 stage_a.jsonl、initial_guidance.json、通过/不通过标签，或提出 Stage-B 指导更新时使用。
---

# Stage-B 指导规则审计（Stage-A + Stage-B）

使用此技能手动审计 Stage-A 摘要输出与通过/不通过标签，检测可能的噪声，并迭代优化 Stage-B 指导规则。保持 Stage-A 摘要固定，仅调整 `initial_guidance.json` 和推理提示。

## 实际工作流程

### 输入数据
- **Stage-A 摘要**：已存在于 `output_post/stage_a_bbu_rru_summary_12-22/{MISSION}_stage_a.jsonl`（固定不变）
- **Stage-B 配置**：`configs/stage_b/*.yaml`
- **初始指导**：`output_post/stage_b/initial_guidance.json`

### 执行流程
1. 使用 `bash scripts/stage_b.sh config={config_name}` 执行 Stage-B rule-search
2. 输出文件位于 `output_post/stage_b/{MISSION}/{run_name}/` 下
3. **BBU 场景额外步骤**：执行 `bash scripts/run_rule_search_postprocess.sh` 生成带人工审核备注的文件

### Agent 核心任务

#### 1. 噪声数据识别与排除
**目标**：排除疑似噪声数据（人工标注错误、特殊情况、Stage-A 摘要错误），只专注于**可学习和拯救的样本**。

**噪声类型识别**：
- **人工标注错误**：标签与 Stage-A 摘要证据明显冲突（如 `label=pass` 但关键证据缺失）
- **异常工单（去除无关图片后有效图过少但GT=pass）**：参考 `scripts/flip_stage_a_pass_by_irrelevant.py` 的口径：对 `label=pass` 的记录，若存在无关图片，且去除无关图片后有效图片数 `< 3`，则该 `pass` 视为采集/提交噪声（应按 fail 处理）。这类不应参与规则学习或用于“拯救/可学习”结论；建议进入过滤名单，必要时抽样人工复核后整体忽略。
- **不可观测噪声（高频）**：`gt_fail_reason_text` 或人工备注提到的失败原因不在 Stage‑A 摘要的“可观测词表”中（例如站点名/拍照清晰度/补拍要求/线径一致性/颜色类描述等）。这类样本对 Stage‑B 属于不可学习，应优先过滤或 stop‑gradient，而不是在 guidance 中硬编码去拟合。
- **特殊情况**：边界案例、罕见场景、标注规则不明确的情况
  - **重要**：对于不在 Stage-A 语料库和训练 label 中的特殊情况，默认为"噪声"或"不可学习的"，可以忽略
  - 只关注那些模型有可能能判断正确的样本（即 Stage-A 摘要包含足够证据的样本）
- **Stage-A 摘要错误**：摘要遗漏关键信息、误识别、格式异常

**可学习样本特征**：
- Stage-A 摘要包含足够证据支持正确判决
- 当前 guidance 规则存在缺陷或缺失，导致误判
- 通过调整 guidance 规则可以纠正判决

**排除策略**：
- 标记为噪声的样本：提供 `group_id`、标签、怀疑原因
- 建议从训练池中排除或标记为 `no_evidence_group_ids`
- 不在 guidance 规则中针对噪声样本做特殊处理

#### 2. 指导规则优化
**目标**：提出、修改 `output_post/stage_b/initial_guidance.json`，使得整体判决准确率提升。

**优化方向**：
- **G0（任务要点）**：确保清晰、可执行，基于 Stage-A 摘要中可观察到的标记
- **S*（结构不变量）**：定义证据覆盖、视角主次、配对规则等结构性约束
- **G*（可学习规则）**：基于困难案例和误判模式，添加或修改规则

**规则编写原则**：
- 简短、正向、基于可观察证据
- 避免要求 Stage-A 很少输出的特征
- 禁止第三状态词面（需复核、证据不足等）
- 保持规则之间的逻辑一致性

#### 3. Reflection 规则发现分析
**目标**：思考如何让 reflection 模型找到更好的规则；如果没有，分析为何未能找到。

**分析维度**：

**A. 困难案例质量**
- 检查 `rule_search_hard_cases.jsonl` 中的案例是否具有代表性
- 评估案例是否包含足够的证据支持规则推导
- 识别是否存在系统性模式（如跨图配对、去重、视角区分）

**B. Proposer Prompt 有效性**
- 检查 `configs/prompts/stage_b_rule_search_proposer_prompt.txt` 是否清晰
- 评估 prompt 是否引导模型关注关键模式
- 识别 prompt 中可能缺失的指导信息

**C. 候选规则质量**
- 检查 `rule_candidates.jsonl` 中的候选规则
- 分析被拒绝的规则：是否因为 gate 阈值过高？规则本身是否有问题？
- 分析被接受的规则：是否真正改善了指标？

**D. Gate 机制分析**
- 检查 `benchmarks.jsonl` 中的指标趋势
- 分析 `rule_search_candidate_regressions.jsonl` 中的回归案例
- 评估 gate 阈值（RER、changed_fraction、bootstrap）是否合理

**E. 规则发现失败原因**
- **证据不足**：困难案例中证据不足以推导规则 → 建议增加更多高质量案例
- **Prompt 不清晰**：proposer prompt 未引导模型关注关键模式 → 建议优化 prompt
- **规则冲突**：候选规则与 scaffold（S*）冲突 → 建议调整 scaffold 或规则
- **Gate 过严**：规则本身合理但未通过 gate → 建议调整 gate 阈值或规则表述
- **规则重复**：候选规则与现有规则语义重复 → 建议合并或删除冗余规则

## 工作流程

### 1) 加载最小文档（可选但推荐）
如不确定数据契约或流程，请阅读：
- `docs/README.md` 查看文档索引
- `docs/training/REFERENCE.md`
- `docs/runtime/STAGE_B_RUNTIME.md`
- `docs/stage_b/DIAGNOSIS_AND_REVIEW.md`
保持上下文最小化；不要批量加载。

### 2) Stage-A 摘要抽样审计
**目标**：了解模型实际输出的内容。
- 从 `output_post/stage_a_bbu_rru_summary_12-22/{MISSION}_stage_a.jsonl` 中随机读取 5–10 条记录。
- 注意一致字段（如站点距离、组N、RRU/BBU 对象、标签）。
- 提取 Stage-B 可可靠使用的最小“信号词汇表”。

**BBU 场景关注点**：
- BBU设备、挡风板、螺丝、光纤插头、光纤、电线、标签
- 注意：BBU 场景**禁止**输出站点距离、RRU相关内容、紧固件、接地线、分组前缀（组N:）
- 关注 BBU 设备完整性、挡风板需求/方向、螺丝/插头符合性、光纤保护/弯曲半径、电线捆扎

**RRU 场景关注点**：
- RRU设备、紧固件、接地线、尾纤、标签、站点距离
- 注意：RRU 场景**必须**输出站点距离（左上角水印），**禁止**BBU相关内容
- 关注配对关系（组N 或站点距离）、标签可识别性、套管保护

### 2.5) 反射降级：Agent 扫描式规则挖掘（不依赖 reflection）
**目标**：当噪声较多、reflection 难以稳定提出可用规则时，基于 `{MISSION}_stage_a.jsonl` + `initial_guidance.json` 直接做大样本/全量扫描，提炼**最小可观测证据口径**与**精简规则**；尽量不依赖 Stage‑B proposer/reflection。

**核心产物**：
- `saveable_ticket_keys`：可学习/可拯救（缺关键证据或规则缺口导致误判）
- `noise_ticket_keys`：不可观测噪声（含“去除无关图片后有效图过少但GT=pass”与“证据齐全但标 fail”等）
- `uncertain_ticket_keys`：需要人工确认任务边界/口径
- `initial_guidance.json` 的精简更新建议：仅保留任务关键证据口径（G0 + 少量 G*），通用约束上移到共享 prompt

**执行要点**：
1. 先定义该 mission 的 “通过所需关键 token” 与 “明确负项 token”（必须来自 Stage‑A 稳定输出）。
2. 大样本扫描：统计关键 token 缺失/出现与 `label` 的对应关系，优先定位误放行与误拦截。
3. 噪声止损：在形成规则前，先剔除不可学习噪声（尤其是“去除无关图片后有效图过少但GT=pass”）。
4. 规则总结必须“短 + 可执行 + 仅基于可观测 token”；发现不确定或新模式时，应立即向用户汇报并请求确认口径。

**精简指导的原则（推荐）**：
- `initial_guidance.json` 每个 mission 只保留任务相关的关键证据口径与少量强规则（G0 + 少量 G*）；避免在每个 mission 重复通用噪声处理。
- BBU/RRU 的通用约束应优先放在共享 system prompt（例如 `src/prompts/stage_b_verdict.py`），而不是散落在每个 mission 的 experiences 内。

### 3) Stage-B 输出分析
**目标**：理解当前 rule-search 的执行情况和问题。

**关键文件**：
- `rule_search_hard_cases.jsonl`：困难案例（用于 reflection proposer）
- `rule_candidates.jsonl`：候选规则及其 gate 指标
- `benchmarks.jsonl`：每轮迭代的 train/eval 指标
- `rule_search_candidate_regressions.jsonl`：回归案例
- `guidance.json` + `snapshots/`：当前指导规则及历史快照
- **BBU 场景额外文件**：`rule_search_hard_samples.jsonl`（包含人工审核备注）

**jump_reflection（baseline-only 审计模式）**：当模型无法稳定提出可用规则时，建议先跳过 proposer/reflection/gating，只做 baseline rollout + 输出审计材料，然后人工/agent 迭代 `initial_guidance.json`。

- **开启方式**：
  - CLI/ENV：`jump_reflection=true config={config_name} bash scripts/stage_b.sh`（或 `--jump-reflection`）
  - YAML：在 `configs/stage_b/*.yaml` 顶层设置 `jump_reflection: true`
- **行为**：
  - 跳过 proposer/reflection 与 gate；不产生/不更新候选规则
  - 仅执行 rollout（不截断，超长 prompt 直接 drop）并导出与 hard cases 相关的审计文件
  - 每处理 `runner.logging_steps` 个 ticket，rank0 追加一次阶段性 metrics（同时打印到日志）
- **关键输出（位于 `output_post/stage_b/{MISSION}/{run_name}/`）**：
  - `baseline_metrics.json`：一次性汇总指标（acc/fn/fp 等）
  - `baseline_metrics_steps.jsonl`：按 `runner.logging_steps` 记录的阶段性指标快照（便于看数据分布/是否被 drop）
  - `baseline_ticket_stats.jsonl`：逐工单统计（majority_pred/agreement 等）
  - `baseline_wrong_cases.jsonl`：所有错例 join Stage-A `per_image`（用于快速定位证据缺失/噪声）
  - `baseline_np_cases.jsonl`：NP（GT=pass → pred=fail，false block）分桶
  - `baseline_ng_cases.jsonl`：NG（GT=fail → pred=pass，false release）分桶
  - `rule_search_hard_cases.jsonl`：仍会生成（sampler=baseline），用于快速 triage



**BBU 场景额外信息**：
- **Postprocess 流程**：执行 `bash scripts/run_rule_search_postprocess.sh` 后，会生成 `rule_search_hard_samples.jsonl`
- **人工审核备注**：该文件包含 `gt_fail_reason_text` 和 `gt_fail_reason_texts` 字段
  - 这些备注来自标注人员的**非正式口头批注**，用于提醒施工人员执行整改
  - 备注内容反映了人工审核时发现的具体问题和不通过原因
  - 这些备注可以帮助理解为什么某些样本被标记为 `fail`，以及模型应该如何识别这些问题
- **备注利用策略**：
  - 优先关注 `gt_label=fail` 且 `pred_label=pass` 的误放行案例（这些是 postprocess 筛选出的）
  - 结合 `gt_fail_reason_text` 分析模型未能识别的问题模式
  - 基于备注内容优化 guidance 规则，使模型能够识别这些关键问题
  - 注意：备注可能包含非正式用语，需要提炼为可执行的规则表述

**分析步骤**：
1. 检查 `benchmarks.jsonl` 的指标趋势（准确率、误放行率等）
2. 查看 `rule_candidates.jsonl` 中被接受/拒绝的规则及原因
3. 分析 `rule_search_hard_cases.jsonl` 中的困难案例质量
4. **BBU 场景**：分析 `rule_search_hard_samples.jsonl` 中的误放行案例及人工备注
5. 检查 `rule_search_candidate_regressions.jsonl` 中的回归模式

### 4) 噪声数据识别
**目标**：识别并排除不可学习的噪声样本。

**识别模式**：
- `label=pass` 但关键证据缺失（如缺失 RRU、缺失紧固件、无站点距离；或 BBU 场景缺失 BBU设备、缺失螺丝等）
- `label=fail` 但证据似乎充分
- Stage-A 摘要明显错误或遗漏关键信息
- **特殊情况**（边界案例、罕见场景）：
  - **重要原则**：对于不在 Stage-A 语料库和训练 label 中的特殊情况，默认为"噪声"或"不可学习的"，可以忽略
  - 只关注那些模型有可能能判断正确的样本（即 Stage-A 摘要包含足够证据的样本）
  - 如果 Stage-A 摘要中完全没有相关证据，即使标签正确，也应视为不可学习

**BBU/RRU 通用噪声模式（已在 BBU 工单中验证；RRU 往往同类存在）**：
- **备注/整改驱动型噪声**：`gt_fail_reason_text` 常出现“站点名/站名填写”“拍全/拍清晰/遮挡”“补拍某端”“线径一致性”“标签材质/运营商标识”“固定螺丝拍清楚”等。这些多数属于拍照规范或现场整改提示，不等价于 Stage‑A 的结构化摘要信号；若 Stage‑A 不输出对应 token，则 Stage‑B 不可学习。
- **OCR/标签归一化导致不可观测**：Stage‑A 往往将标签文本归一（如“标签/可以识别”“标签/无法识别”），不会稳定保留站点名、运营商标识、PVC 标签等细节；因此任何依赖“具体站点名/具体标签材质/具体字样”的 fail 规则通常不可学习。
- **“证据齐全但标 fail”（unlearnable fail）**：若 Stage‑A 摘要已满足该 mission 的全部必要条件（例如关键对象齐全且均“符合要求”），但 `gt_label=fail` 且该 fail 原因无法在摘要中定位，则应视为不可学习噪声。此类样本会导致 rule_search 反复提出与现有规则同义的候选规则（changed_fraction≈0），无法通过 gate。
- **“去除无关图片后有效图过少但GT=pass”（unlearnable pass for learning）**：参考 `scripts/flip_stage_a_pass_by_irrelevant.py` 的口径：对 `gt_label=pass`，若存在无关图片，且去除无关图片后有效图片数 `< 3`，按采集/提交噪声处理；应从规则学习与评估池中剔除，避免将“无关图片数量/占比”误学成通过条件。
- **同一 `group_id` 多标签并存**：Stage‑A JSONL 可能存在同一个 `group_id` 同时出现 `label=pass` 与 `label=fail` 的记录。审计与 join 必须以 `ticket_key="{group_id}::{label}"` 为主键，而不是只用 `group_id`。
- **第三态/“需复核”注意**：Stage‑A 可能包含 `需复核,备注:` 的结构化 marker，但 Stage‑B prompt 构造会清洗/移除“需复核”字面，仅保留“备注:”内容；因此 guidance 不应依赖“需复核”作为触发词，且 Stage‑B 输出协议也禁止第三态词面。
- **跨任务/跨域串扰 token**：BBU 任务摘要中偶发出现 RRU/站点距离/ODF 等 token，RRU 任务中偶发出现 BBU/机柜等 token；应默认作为干扰项，不可替代本任务关键证据。

**快速判别流程（建议：5–10 分钟内完成）**：
1. 先定义该 mission 的“可观测最小词表”（Stage‑A 会稳定输出的关键 token），并把 guidance 的 `G0/S*` 约束写成“只基于可观测 token”。
2. 从 `rule_search_hard_cases.jsonl` /（BBU 额外）`rule_search_hard_samples.jsonl` 取样，按 `ticket_key` 回查 Stage‑A 摘要，判断 hard case 是否“缺关键 token”还是“关键 token 齐全仍标 fail”。
3. 若 hard case 的主要 fail 原因词面在 Stage‑A 中完全不可见（站点名/拍清楚/补拍/线径/颜色/标签材质等），直接归类为不可学习噪声：不在 guidance 中拟合，优先过滤或 stop‑gradient。
4. 若 hard case “缺关键 token”且该 token 在 Stage‑A 中可观测，则属于可学习：通过新增/更新 `G*` 规则补齐缺失触发（优先降低误放行）。
5. 若 hard case “关键 token 齐全仍标 fail”，优先假设 label 噪声或任务边界缺失维度：加入 unlearnable 过滤名单，避免污染 rule_search。

**过滤/止损机制（推荐优先级）**：
- **优先：config 级过滤（全流程一致）**：在 `configs/stage_b/*.yaml` 中使用 `ticket_filter.exclude_ticket_keys_path`，在 ingest 后直接剔除不可学习/噪声 `ticket_key`（`{group_id}::{label}`）。
- **次选：stop-gradient（只影响反思/学习）**：通过 decision pass 输出 `no_evidence_group_ids`，把不可学习样本从 learnable CASES 中排除（仍保留在评估/审计中）。
- **BBU 场景辅助工具（示例）**：
  - 备注关键词噪声过滤：`scripts/build_stage_b_noise_ticket_filter.py`（从 postprocess 的 `rule_search_hard_samples.jsonl` 生成噪声 `ticket_key` 列表）。
  - 去除无关图片后有效图过少但GT=pass过滤：`scripts/build_stage_b_noise_ticket_filter.py --exclude-pass-min-useful 3`（与 `flip_stage_a_pass_by_irrelevant.py` 默认一致）。
  - “证据齐全但标 fail”过滤：以 Stage‑A 可观测 token 为准定义 unlearnable 规则，再生成 `ticket_key` 过滤文件（例如 BBU 安装方式检查已验证此模式可显著提升 learnability）。

**BBU 场景特殊处理**：
- 结合 `rule_search_hard_samples.jsonl` 中的 `gt_fail_reason_text` 判断：
  - 如果备注中提到的问题在 Stage-A 摘要中完全没有体现 → 视为噪声（Stage-A 摘要错误或特殊情况）
  - 如果备注中提到的问题在 Stage-A 摘要中有部分体现，但模型未能识别 → 可学习样本，需要优化规则
  - 如果备注中提到的问题在 Stage-A 摘要中明确体现，但模型误判 → 可学习样本，需要优化规则

**输出格式**：
- 噪声工单列表：`group_id`、标签、怀疑原因、建议处理方式（排除/标记为 no_evidence）
- **BBU 场景**：对于有 `gt_fail_reason_text` 的样本，说明备注与 Stage-A 摘要的对应关系

### 5) 指导规则优化
**目标**：基于困难案例和误判模式，优化 `initial_guidance.json`。

**优化策略**：
- 保持规则简短、正向，基于 Stage-A 摘要中可观察到的标记
- 避免要求 Stage-A 很少输出的特征
- 当数据有噪声时，添加"忽略"或"延迟"子句，而非强制判不通过
- 确保规则与 scaffold（S*）不冲突

**BBU 场景优化策略**：
- **利用人工审核备注**：结合 `rule_search_hard_samples.jsonl` 中的 `gt_fail_reason_text` 优化规则
  - 分析备注中提到的常见问题模式（如"螺丝松动"、"挡风板方向错误"等）
  - 将这些非正式备注提炼为可执行的规则表述
  - 确保规则能够识别备注中提到的关键问题
- **关注误放行案例**：优先优化导致 `gt_label=fail` 但 `pred_label=pass` 的规则
  - 这些案例在 `rule_search_hard_samples.jsonl` 中被筛选出来
  - 结合备注内容，理解模型未能识别的问题
  - 添加或修改规则以覆盖这些误放行模式

**规则结构**：
- **G0**：任务要点（必须存在，不可删除）
- **S***：结构不变量（只读，不可修改）
- **G***：可学习规则（可增删改）

### 6) Reflection 规则发现分析
**目标**：分析为何 reflection 未能找到更好的规则，并提出改进建议。

**分析步骤**：
1. **困难案例质量评估**：
   - 检查 `rule_search_hard_cases.jsonl` 中的案例是否具有代表性
   - 评估案例是否包含足够的证据支持规则推导
   - 识别是否存在系统性模式

2. **Proposer Prompt 评估**：
   - 检查 `configs/prompts/stage_b_rule_search_proposer_prompt.txt`
   - 评估 prompt 是否清晰引导模型关注关键模式
   - 识别 prompt 中可能缺失的指导信息

3. **候选规则质量分析**：
   - 检查 `rule_candidates.jsonl` 中被拒绝的规则
   - 分析拒绝原因：gate 阈值过高？规则本身有问题？
   - 分析被接受的规则是否真正改善了指标

4. **Gate 机制评估**：
   - 检查 `benchmarks.jsonl` 中的指标趋势
   - 分析 `rule_search_candidate_regressions.jsonl` 中的回归案例
   - 评估 gate 阈值（RER、changed_fraction、bootstrap）是否合理

5. **失败原因诊断**：
   - 证据不足 → 建议增加更多高质量案例
   - Prompt 不清晰 → 建议优化 proposer prompt
   - 规则冲突 → 建议调整 scaffold 或规则
   - Gate 过严 → 建议调整 gate 阈值或规则表述
   - 规则重复 → 建议合并或删除冗余规则

### 7) 呈现编辑建议 + 分析报告
**输出内容**：
- **关键观察摘要**（2–5 条要点）
- **噪声工单列表**：`group_id`、标签、怀疑原因、建议处理方式
- **`initial_guidance.json` 更新建议**（精确 JSON 编辑块）
- **Reflection 规则发现分析**：
  - 困难案例质量评估
  - Proposer Prompt 评估与优化建议
  - 候选规则质量分析
  - Gate 机制评估
  - 失败原因诊断与改进建议
- **可选**：小型测试计划（重跑 Stage-B 验证改进效果）

## 启发式模式（按任务调整）

### 通用模式
- **跨图配对**：当站点距离相等时允许跨图配对（如无更近候选，允许 ±2 容差）。
- **去重**：不重复计算同一对象的多个视角；按站点距离统计存在性。
- **忽略“无关图片”**：除非*所有*图片都无关，否则不自动判不通过（仅在任务定义允许时）。
- **视角区分**（BBU 挡风板任务）：全局视角（显示完整、信息最多）与局部视角（只显示部分）需同时存在；冲突时以全局视角为准。

### 任务特定配对规则

#### RRU 场景
- **RRU安装检查**：RRU设备 + 紧固件（需成对出现）
- **RRU位置检查**：RRU设备 + RRU接地线(标签可识别)（需成对出现）
- **RRU线缆**：尾纤 + 套管保护（需成对出现）

#### BBU 场景
- **BBU安装方式检查（正装）**：BBU设备 + BBU安装螺丝（需符合要求）
- **BBU接地线检查**：机柜处接地螺丝 + 地排处接地螺丝 + 电线（需符合要求且捆扎整齐）
- **BBU线缆布放要求**：BBU端光纤插头 + ODF端光纤插头 + 光纤（需符合要求且有保护措施和弯曲半径合理）
- **挡风板安装检查**：BBU设备（必须存在）+ 挡风板（根据情况判断是否需要及是否符合要求）

### 领域特定规则

#### BBU 领域特点
- **禁止项**：禁止输出站点距离、RRU相关内容、紧固件、接地线、分组前缀
- **关注对象**：BBU设备、挡风板、螺丝、光纤插头、光纤、电线、标签
- **品牌规则**：爱立信品牌 BBU 不安装挡风板；仅当存在两台 BBU 时，在两台 BBU 之间必须安装挡风板
- **关联性**：螺丝/挡风板与本体强关联；BBU端光纤插头多为蓝白色且插在 BBU 设备上

#### RRU 领域特点
- **必须项**：凡非“无关图片”，必须输出站点距离（左上角水印）
- **配对方式**：同图优先用组N配对；跨图用站点距离标识同一安装点
- **禁止项**：禁止 BBU相关内容与品牌/机柜概念（如机柜/机房/挡风板/BBU端/ODF端）
- **标签要求**：标签文本需完整保留，无法识别写“标签/无法识别”

## Rule-Search 机制说明

### 工作流程
1. **Baseline Rollout**：在 train pool 上运行 baseline rollout，建立每个 ticket 的 majority stats
2. **困难案例选择**：从 baseline 中选择 `reflect_size` 个高价值 mismatch tickets（默认 16 个）
3. **Reflection Proposer**：使用 reflection 模型从困难案例中提出 `num_candidate_rules` 个候选规则（默认 3 个）
4. **A/B 验证**：对每个候选规则，在 train pool 上运行 A/B rollout
5. **Gate 评估**：使用 gate 指标评估候选规则：
   - **RER（相对错误率下降）**：`min_relative_error_reduction >= 0.1`
   - **Changed Fraction**：`max_changed_fraction <= 0.05`（覆盖率合理性）
   - **FP Rate Increase**：`max_fp_rate_increase <= 0.01`（误放行率控制）
   - **Bootstrap**：bootstrap 概率验证（默认 `min_prob >= 0.8`）
6. **规则接受**：只有通过 gate 的规则才会被接受并写入 guidance
7. **Early Stop**：如果连续 `patience` 轮（默认 5 轮）没有规则被接受，则早停

### Gate 指标说明
- **RER（Relative Error Reduction）**：`(err_base - err_new) / max(err_base, eps)`
  - 衡量新规则相对于 baseline 的错误率下降
  - 阈值：`>= 0.1`（即至少 10% 的相对错误率下降）
- **Changed Fraction**：`mean_i [majority_pred_base(i) != majority_pred_new(i)]`
  - 衡量新规则改变了多少 ticket 的判决
  - 上限：`<= 0.05`（即最多改变 5% 的 ticket）
- **FP Rate Increase**：`fp_rate_new - fp_rate_base`
  - 衡量新规则是否增加了误放行率
  - 上限：`<= 0.01`（即最多增加 1% 的误放行率）
- **Bootstrap Probability**：通过 bootstrap 重采样验证 RER 的稳定性
  - 阈值：`>= 0.8`（即 80% 的 bootstrap 样本中 RER >= 阈值）

## 输出格式清单
- **关键观察摘要**（2–5 条要点）
- **噪声工单列表**：`group_id`、标签、怀疑原因、建议处理方式
- **`initial_guidance.json` 更新建议**（精确 JSON 编辑块）
- **Reflection 规则发现分析**：
  - 困难案例质量评估
  - Proposer Prompt 评估与优化建议
  - 候选规则质量分析
  - Gate 机制评估
  - 失败原因诊断与改进建议
- **可选**：小型测试计划（重跑 Stage-B 验证改进效果）

## 注意事项
- 当证据与标签冲突时，**不要**假设模型错误；标记供重新审核。
- 保持 Stage-B 提示简洁以匹配 8B 模型容量。
- 在此工作流程中不要更改 Stage-A 摘要生成。
- **BBU 与 RRU 场景差异**：
  - BBU 场景不依赖站点距离，主要关注设备完整性、安装符合性、线缆布放
  - RRU 场景依赖站点距离进行安装点合并，关注跨图配对关系
  - 两个场景的指导规则结构相同（G0/S*/G*），但关注对象和判定逻辑不同
- **Rule-Search 约束**：
  - Reflection 仅作为 rule proposer，不直接修改 guidance
  - 所有规则修改必须通过 gate 验证
  - 只有通过 gate 的规则才会被接受并写入 guidance

## 相关文件与命令
- **Stage-A 摘要**：`output_post/stage_a_bbu_rru_summary_12-22/{MISSION}_stage_a.jsonl`（固定不变）
- **Stage-B 运行**：`bash scripts/stage_b.sh config={config_name}`（使用 `configs/stage_b/*.yaml`）
- **BBU 场景 Postprocess**：`bash scripts/run_rule_search_postprocess.sh`（生成带人工审核备注的文件）
- **指导文件**：`output_post/stage_b/initial_guidance.json`
- **Stage-B 输出目录**：`output_post/stage_b/{MISSION}/{run_name}/`
- **关键输出文件**：
  - `rule_search_hard_cases.jsonl`：困难案例
  - **BBU 场景**：`rule_search_hard_samples.jsonl`：误放行案例（`gt=fail & pred=pass`）+ 人工审核备注（`gt_fail_reason_text`）
  - `rule_candidates.jsonl`：候选规则及 gate 指标
  - `benchmarks.jsonl`：每轮迭代的 train/eval 指标
  - `rule_search_candidate_regressions.jsonl`：回归案例
  - `guidance.json` + `snapshots/`：当前指导规则及历史快照
- **Proposer Prompt**：`configs/prompts/stage_b_rule_search_proposer_prompt.txt`
- **BBU 场景数据源**：`output_post/BBU_scene_latest.xlsx`（包含人工审核备注的 Excel 文件）
