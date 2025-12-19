# Design — update-stageb-stop-gradient-need-review

## 1) 术语与目标

### 术语
- **ticket / group**：以 `group_id` 为业务单位的一条 Stage‑B 工单（输入包含 Stage‑A 多图摘要 + `gt_label`）。
- **rollout candidates**：同一 ticket 在多 decode / 多采样下产生的多个二分类候选（通过/不通过）。
- **gradient candidate（梯度候选）**：存在“可优化空间”的 ticket：
  - `label_match=False`（最终选择与 `gt_label` 不一致），或
  - rollout 不一致 / 低一致性（候选 verdict 出现 pass+fail 混合，或 vote_strength 低）。
- **stop‑gradient（need‑review）**：给定 `gt_label` 后仍无法产出任何可学习、可验证、可淘汰的 guidance 更新假设的 ticket（噪声/不可判/矛盾）。

### 目标
- `need_review` 等价于 stop‑gradient，且严格 **group_id 级**，**非粘性**（每 epoch 重判）。
- stop‑gradient ticket 不得驱动 guidance 修改（不进入任何已应用操作的 `evidence`，且不进入 ops 生成上下文）。
- 对所有梯度候选强制闭环：每个 group 必须是 stop‑gradient 或对 guidance 更新有建设性贡献（被 evidence 覆盖）；未覆盖样本必须被确定性复判（而不是静默丢弃）。

## 2) 现状：为何称为 “bundle 混合”

reflection 的输入单位是 `ExperienceBundle`：一次反思会把多个 `group_id` 的 CASES 混合在同一上下文中，产出一个 JSON 对象（包含多个 ops）。这种设计的优势是能做跨样本归纳并输出去重后的少量规则更新；但风险是：
- 噪声/不可判样本会在同一上下文中影响规则归纳方向；
- 若未强制 evidence/stop‑gradient 分区，stop‑gradient 样本可能“驱动”规则更新（污染梯度）。

## 3) 目标流程（按 epoch；非粘性）

### 3.1 Rollout → Selection（保持）
1) rollout 生成候选（严格两行二分类，无第三状态词面）。
2) selection 选出最终 verdict，并计算一致性信号（vote_strength / 低一致性等）。

### 3.2 梯度候选筛选（group_id 级）
把 ticket 分为三类：
- **No‑grad**：最终 verdict 正确且候选稳定（不进入 reflection）。
- **Grad‑candidate**：`label_match=False` 或 rollout 不一致/低一致性，或触发 `conflict_flag/needs_manual_review` 等不确定性信号（进入 reflection；不等价于 stop‑gradient）。
- **Hard‑fail**：解析/无候选/selection_error 等（仅写 `failure_malformed.jsonl`；不进入 reflection/need‑review）。

信号对齐（以 group-level 为准，避免实现漂移）：
- `vote_strength`：format_ok 候选的多数票比例；
- `low_agreement`：`vote_strength < manual_review.min_verdict_agreement`；
- `label_match`：最终 verdict（含 deterministic override）是否等于 `gt_label`；
- `conflict_flag`：等价于 `label_match=false`；
- `needs_manual_review`：表达“即便 label_match=true 仍高风险/不确定”的可观测信号，不得被当作 stop‑gradient 本身。

### 3.3 Two-pass Reflection（必须）

为满足 “stop‑gradient 不得驱动规则更新” 与 “bundle 混合不污染梯度” 两个目标，反思拆为两次 LLM 调用：

1) **Decision pass（stop‑gradient 判定）**
   - 输入：同一 mission 下的一批 grad-candidates（bundle）。
   - 输出：严格 JSON，仅用于产生 `no_evidence_group_ids=S`（stop‑gradient 判定）。

2) **Ops pass（仅 learnable 生成规则更新）**
   - 输入：仅包含 learnable groups：`L = G \\ S`。
   - 输出：严格 JSON ops（add/update/delete/merge/none）。
   - 由于 stop‑gradient group 不进入该 pass 的输入上下文，stop‑gradient 无法影响 ops 归纳，隔离最强。

### 3.4 强制闭环：覆盖/互斥不变量（以 two-pass 为基础）

对每个 decision/ops 处理的 batch，定义集合：
- `G`：decision pass 的输入梯度候选 group_id 集合
- `S`：decision pass 输出的 stop‑gradient 集合（`no_evidence_group_ids`）
- `L`：learnable 候选集合（`L = G \\ S`）
- `E`：ops pass 输出中 `operations[*].evidence` 的并集（learnable evidence 覆盖集；只统计通过 validator 的 ops）

必须满足：
1) **互斥**：`S ∩ E == ∅`（stop‑gradient 不得作为 evidence；由 two-pass + validator 双重保证）
2) **learnable 覆盖**：`L == E`（每个 learnable 候选都必须被至少一条 op evidence 覆盖；否则视为“未形成可审计梯度”）

不满足时的安全退化（robust）：
- 对 `S ∩ E != ∅`：该 op MUST NOT 被应用；并记录可观测告警。
- 对 `L \\ E`（learnable 未覆盖）：这些 group MUST 被加入“下一批次复判队列”，在后续**更小 batch**中重复 decision/ops 两段式反思，而不是立即写入 need‑review。
- 为保证终止性，系统 MUST 对每个 group 设置每 epoch 的最大复判次数（retry budget）；默认值：每个 `group_id` 每个 epoch **最多重试 `2` 次**。重试仅指“再次执行 decision+ops 两段式反思”，不得仅为闭环而重跑 rollout；耗尽后仍无法闭环者，进入 need‑review（stop‑gradient）。

成本上界（确定性）：
- 设定 mission-level 的 reflection 调用次数上限（decision+ops 都计数）；超限后将剩余待处理梯度候选路由到 need-review（reason_code=budget_exhausted）。
- 对复判队列采用确定性的 batch 缩小策略（默认：第 k 次复判批大小 = `max(1, floor(batch_size / 2**k))`，并按 group_id 稳定排序分桶），以降低漏写与上下文干扰，同时确保成本可控。

evidence 严格性（避免系统性打穿互斥）：
- ops pass 的 `operations[*].evidence` 缺失/为空/包含非 learnable `group_id` 时，该 op 必须视为无效并拒绝应用；
- 禁止任何 “evidence 缺失 ⇒ 默认全量 bundle” 的回退；否则会系统性破坏 `S∩E==∅` 与闭环语义。

#### 为什么会出现 `L \\ E`（未覆盖 learnable）
常见原因包括：
- LLM 在 ops pass 中对某条 op 仅引用了部分 learnable group 的 `evidence`，遗漏了其它可学习 group_id；
- validator 因 evidence 非法（包含 stop‑gradient、未知 group_id、格式错误等）拒绝应用 op，导致 `E` 变小；
- 模型输出 `operations=[]`（或有效 op 数为 0），但 decision pass 未将这些 group 判为 stop‑gradient，导致它们既不在 `S` 也不在 `E`；
- bundle 太大导致 JSON 截断/解析失败，触发安全回退。

### 3.5 need-review 输出（stop‑gradient；非粘性）
`need_review_queue.jsonl` 记录 stop‑gradient（S 集合），并带 `epoch/reflection_cycle/reflection_id` 等审计字段。
同一 group 在不同 epoch 可重复出现或消失；不得作为后续 epoch 的黑名单。

### 3.6 Guidance lifecycle（反馈隔离 + 缓冲）

由于 stop‑gradient 判定在 decision pass 之后才产生，而 selection/hit-miss 反馈可能在更早发生，系统需要引入反馈缓冲：
- 对每个 ticket，先记录 “候选/最终 verdict 与 gt 的一致性” 以及 “当时最近一次变更 keys（last_applied_rule_keys）” 到 pending buffer；
- 在该 ticket 被 decision pass 判为 stop‑gradient 或 hard‑fail 时：丢弃其 pending 反馈（不计入 hit/miss）；
- 在该 ticket 被确认 learnable 且参与 ops evidence 时：才允许将其 pending 反馈提交到 rule hit/miss 计数。

该缓冲的目标是：stop‑gradient/hard-fail 永不污染 rule 置信度；learnable 样本才贡献梯度闭环。

## 4) Prompt 与 initial guidance（优化面）

### 4.1 Two-pass prompt 的职责分离
- 使用 **两个独立 prompt 文件**：
  - **Decision pass prompt**：尽可能短，只负责 `group_id` 级 stop‑gradient 判定（输出 `no_evidence_group_ids`），避免在同一次调用中混入 “归纳规则/写 ops” 任务，降低上下文污染与输出不确定性。
  - **Ops pass prompt**：只看 learnable groups（`G\\S`），只产出 JSON operations，并显式提示系统会校验 `L==E`，未覆盖将触发复判，从而推动模型更完整地填写 `operations[*].evidence`。
- prompt 侧明确：**只有 `G0` 只读**，其余 `G1+` 均允许 add/update/delete/merge，以保证“梯度”有落点且避免把不可变规则散落到多个 key。

### 4.2 “激发探索但不显式硬编码”的提示策略
为鼓励模型发现（例如数量/频次等）潜在规律，而不把规律手写到初始规则里：
- prompt 用“探索维度 checklist”引导（全局/局部、AND/OR 组合、频次/数量、多图互证、关键证据缺失→保守等），但不写入 mission 专属阈值或具体规则；
- 强制输出为“可泛化、可验证、可淘汰”的假设，避免把 Stage‑A 摘要对象链粘贴进 guidance；
- 通过 two-pass 隔离 stop‑gradient，并用 closure+retry 机制把“遗漏/未覆盖”从静默丢弃变成可恢复流程。
- ops pass 输出可选 `coverage`（learnable/covered/uncovered group_ids）仅用于可观测性；正确性仍以系统计算的 `L/E/L\\E` 为准，以免模型自报覆盖误导闭环。

### 4.3 Initial guidance 的定位
initial guidance 作为 bootstrap，应更偏向：
- mission 定义/硬约束（`G0`，只读）；
- 证据取舍与安全默认（全局/局部互补、多图一致性增强、关键证据缺失时更保守等）；
- 避免把“应由 reflection 学出来”的潜在规律（例如明确数量阈值）直接固化在 seed guidance 中，以免压缩可学习空间。
