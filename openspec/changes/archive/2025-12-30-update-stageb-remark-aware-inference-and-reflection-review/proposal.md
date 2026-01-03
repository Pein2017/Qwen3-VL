# Change Proposal — update-stageb-remark-aware-inference-and-reflection-review

## 背景
- Stage‑B inference 当前依赖 Stage‑A per‑image summary 文本做组级判定，但 summary 中常出现“需复核,备注: …”结构（见 `data/bbu_full_768_poly-need_review/val.jsonl`），容易被模型误读为“异常/必不通过”，从而把“需复核”硬等价为 fail。
- 当前 prompt 还把“无法确认/无法判断/只显示部分/模糊”等表述当作硬触发词直接否决，导致在“备注提供了补充信息/其他图片提供了证据”时仍可能被过度否决。
- 训练/反思阶段存在 gt‑label 与 Stage‑A summary 不一致的噪声样本风险：如果把“标签=fail 但 summary 找不到负证据”的样本当作学习 fail 规则的证据，会系统性放大误拒（把更多 pass 学成 fail）。

## 目标与约束（必须满足）
1) **Inference 输出协议不变且严格二分类两行**：仅允许两行
   - `Verdict: 通过` 或 `Verdict: 不通过`
   - `Reason: ...`
   不得输出 need‑review，也不允许任何第三状态字样出现在最终输出中（含“需复核/证据不足/待定/通过但需复核/通过但需人工复核”等）。
2) **Prompt 变更：备注软信号**：把输入 summary 里的“需复核 + 备注”视为软信号，鼓励模型读取备注综合判断哪些情况可通过、哪些应不通过；不得把“需复核”硬等价为异常/不通过。
3) **Fail‑first 负项触发词**：提供一组跨 4 个 mission 通用的简短负向触发词（不含名词，仅动词/形容词/短语），用于 fail‑first：
   - 必含：未按要求、错误、缺失、松动、损坏、方向不正确、反向、不符合要求、不合格、不合理、未安装、未配备
   - 规则：任一图片出现**与当前 mission 的 `G0` 相关**的明确负项 ⇒ 整组必须判不通过（mission‑scoped fail‑first）。
4) **待确认信号**：“无法确认/无法判断/只显示部分/模糊”属于待确认信号：
   - 不得作为硬触发词直接否决；
   - 但需要一个通用安全约束：**若无法给出支持通过的证据，则判不通过**（避免弱证据放行；不等价于“需复核=不通过”）。
5) **Reflection/训练更自主且更安全**：
   - need‑review 只存在于 reflection/训练阶段，用于审核可疑的 gt‑label 和 Stage‑A summary；不影响 inference 输出协议。
   - 强化“证据一致性”：当 gt‑label=fail 但 Stage‑A per‑image summary 找不到负向证据、模型倾向通过时，应更倾向把该工单加入 need‑review（标注疑似 gt‑label 错误或 Stage‑A summary 问题），并避免把它当作学习 fail 规则的证据。
   - 同理：当 gt‑label 与 Stage‑A summary 明显矛盾（summary 不支持 gt‑label）时，应进入 need‑review 或标注为 label/stageA noise（以训练数据质量为目标）。
6) **任务要点来源与 prompt 结构**：4 个 mission 的任务要点以 `output_post/stage_b/guidance_back.json` 的各自 `G0` 为准；提示词应支持：
   - mission‑specific `G0`
   - 通用负项触发词（fail‑first）
   - “需复核备注解释”
   - **mission 作用域强调**：只根据当前 mission 的 `G0` 判定；同一组工单可能被不同 mission 审核，不同 mission 下允许出现不同结论
   且不得引入名词词表来强耦合某个 mission。

## 变更概览（What）
### Inference（行为变更）
- **Prompt 结构重排**（`src/stage_b/sampling/prompts.py`）：
  - 显式解释 Stage‑A summary 的 `需复核,备注:`：`需复核` 仅表示需要关注备注；判定以备注与上下文证据为准。
  - 将“无法确认/无法判断/只显示部分/模糊”从硬否决规则移除，改为“待确认信号”；引入“无通过证据则不通过”的通用安全约束（要求 Reason 给出支持通过的证据，否则必须输出不通过）。
  - 移除 prompt 中与 mission 强耦合的名词示例列表（例如“BBU/挡风板/接地线/光纤/螺丝/标签…”），改为仅引用 mission‑specific `G0` 与已有 guidance snippets。
- **Fail‑first 规则落地为确定性护栏**（不仅依赖 LLM 自觉）：
  - 定义跨 mission 通用负项触发词表（仅动词/形容词/短语），并实现 **mission‑scoped** fail‑first：仅当负项与当前 `G0` 相关时才整组 fail（同一组工单在不同 mission 下可出现不同 verdict）；
  - 负项通用化（pattern-first）：对 Stage‑A canonical 表达的结构化负项模式（如 `不符合要求/<issue>`）做覆盖，不依赖穷举所有 issue；
  - 该护栏不得把“需复核/无法确认/无法判断/只显示部分/模糊”等待确认信号当作 fail‑first 触发词。

### Reflection / 训练（架构/数据质量变更）
- 在 reflection/训练阶段引入 **need‑review/噪声标注队列**（不进入 inference 输出）：
  - 当 `gt-label` 与 Stage‑A summary 明显不一致（尤其 label=fail 但 summary 无负证据且模型倾向通过），记录到 need‑review 队列，并标注原因：`label_noise_suspect` / `stageA_noise_suspect` / `insufficient_evidence` 等。
  - 对 need‑review 样本：reflection 不产生“学习 fail 规则”的 operations，避免把噪声学成规则；仅允许输出 wishlist（需要补充的 rule/提示）或 noop。

## 范围（Scope）
- 仅覆盖 Stage‑B training‑free runtime 的 prompt/护栏/反思数据质量策略：
  - Inference prompt：`src/stage_b/sampling/prompts.py`
  - Fail‑first 护栏（确定性触发词检测）：`src/stage_b/signals.py` / `src/stage_b/scoring/selection.py`（或 runner 层的 group‑level override）
  - Reflection/need‑review：`src/stage_b/reflection/engine.py`（与其输出工件）
- 不改变 4 个 mission 的业务要点来源（仍以 `guidance_back.json` 的 `G0` 为基准）。

## 非目标（Non‑Goals）
- 不引入任何 mission‑specific 名词词表/实体字典（避免强耦合、避免跨 mission 迁移失败）。
- 不引入 inference 的第三状态输出（包括但不限于 need‑review、待定、证据不足、通过但需复核）。
- 不在本变更中引入检索/外部知识库/embedding。

## 兼容性（Compatibility）
- **输出协议兼容**：Stage‑B inference 仍严格两行二分类输出，保持 `Verdict:` + `Reason:` 协议。
- **行为变化**：fail‑first 触发词护栏与“无通过证据则不通过”会提高不通过比例；需要在离线集上评估通过率变化与 FP 风险。
- **工件变化（内部）**：reflection/训练阶段可能新增 `need_review_queue.jsonl` 辅助工件；但不得改变 inference 的最终输出协议。

## 风险与缓解（重点：FP 风险 & “备注不足默认不通过”副作用）
- **FP（误拒）风险**：触发词（尤其“错误”）可能在否定语境中出现（如“无错误”）。
  - 缓解：触发词检测做“否定前缀”排除（如“无/未发现/未见/不存在/未见明显”等），并将触发词命中写入 selection warnings 以便审计。
- **“备注不足默认不通过”副作用**：把“待确认信号”改成软信号后，仍要求“无通过证据则不通过”，可能使某些原先会被放行的弱证据样本转为 fail。
  - 缓解：Reason 强制给出支持通过的证据（来自 summary/备注/多图一致性），并在离线评估中监控通过率与典型误拒样本；必要时通过配置开关分阶段 rollout。
- **模型对备注的过拟合/过度依赖**：可能出现“备注一句话盖过多图证据”的情况。
  - 缓解：prompt 明确“备注是补充证据的一部分，但不得与多图明确证据相冲突”；冲突时以明确负项/多图一致证据为准。

## 回滚策略（Rollback）
- 以配置/版本化 prompt 为回滚开关：
  - 禁用 fail‑first 触发词护栏（仅保留 prompt 软规则）
  - 恢复旧的“待确认硬否决”规则（仅用于紧急止损）
  - 禁用 need‑review 队列产出（回退到仅 `no_evidence_for_label` 的不确定标记）
- 回滚不改变输出协议，确保下游稳定。

## 成功标准（Success Criteria）
- Inference 输出始终严格两行二分类，且不包含任何第三状态词面（包括“需复核/证据不足/待定/通过但需复核”等）。
- Fail‑first：任一图片 summary 明确命中**与当前 `G0` 相关**的负项触发词/模式 ⇒ 最终 verdict 必为不通过（同一组工单在不同 mission 下可出现不同 verdict；可通过单元测试与离线回放验证）。
- 备注软信号：含“需复核,备注:”的样本不会因“需复核”本身被硬判 fail；判定由备注与上下文证据驱动。
- Reflection/训练：当 label 与 summary 明显矛盾时，样本进入 need‑review/噪声队列且不产生“学习 fail 规则”的 operations。

## 需要同步更新的文档（至少）
- `docs/training/REFERENCE.md`：Stage‑B 运行/产物/护栏与反思‑need‑review 的数据流说明
- `docs/runtime/STAGE_A_STAGE_B.md`：Stage‑A summary → Stage‑B 判定的协议与 fail‑first/备注处理策略
- `docs/reference/stage-B-knowledge-Chinese.md`：中文业务知识与“需复核备注”“待确认信号”“证据不足默认不通过”的解释
