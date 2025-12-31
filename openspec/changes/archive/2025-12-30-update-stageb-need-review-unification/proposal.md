# Change Proposal — update-stageb-need-review-unification

> 注（2025-12-18）：`need_review_queue.jsonl` 的语义已进一步收敛为“label-suspect only”（给了 `gt_label` 仍想不明白错哪才入队）。本 proposal 记录的是当时的“入口统一”与工件产出变更；当前行为以 `openspec/specs/stage-b-training-free/spec.md` 为准。

## Why
- 当前 `manual_review_queue.jsonl`/`need_review_queue.jsonl` 语义分裂：前者混入 pipeline 硬故障，后者偏 quarantine，导致“人工复核入口”不清晰。
- 对 `gt=fail & pred=pass` 的错例，现有基于 `G0` 的隔离策略容易提前阻断学习，不符合“拿到参考答案后归纳 fail 场景并学习”的目标。
- 允许反思提出“可能解释/假设规则”后，必须配套更快的 hit/miss 反馈与自动淘汰，否则会长期污染 guidance。

## What Changes
- 将 need‑review 统一为唯一的“人工复核”队列：记录 **候选池无任何候选支持 `gt_label`**（`label_match=True` 的候选数为 0）的工单。
- need‑review 同时产出 `need_review_queue.jsonl`（流式）与 `need_review.json`（run 结束聚合）。
- reflection prompt 强化两类错例方向（fail 场景 vs pass 例外），并允许“关键证据缺失”作为可学习信号（但需可快速淘汰）。
- guidance lifecycle 补齐/强化 hit/miss 闭环，结合现有清理逻辑快速剔除低质量规则。

## Impact
- 影响能力：`stage-b-training-free`（review routing、reflection 学习策略、guidance 生命周期、输出工件）。
- 影响工件：新增 `need_review.json`；调整 `need_review_queue.jsonl` 语义；`manual_review_queue.jsonl` 不再作为人工复核入口。

## 背景
当前 Stage‑B 在每个 mission/run 目录下同时产出：
- `manual_review_queue.jsonl`：混合了 rollout/解析/选择等“硬故障”与反思后的对齐失败样本；
- `need_review_queue.jsonl`：偏“反思训练治理”的隔离队列（label 与 Stage‑A evidence/`G0` 明显矛盾时入队，且通常阻止学习）。

这带来两个问题：
1) 运营侧的“人工复核”目标被稀释：队列里混入大量并非业务复核（而是 pipeline/debug）的样本。
2) 对 `gt=fail & pred=pass` 的错例，现有 “need‑review quarantine” 倾向直接判定为证据不足/疑似噪声并阻止学习；这与期望的“拿到参考答案后总结 fail 场景并学习规则”相冲突。

本变更希望把复核语义统一为：**仅记录“候选池无任何候选支持 `gt_label`”的工单**；其余错例默认进入学习闭环，并配套“快速淘汰瞎猜规则”的机制。

## 目标与约束（必须满足）
1) **仅保留一个“人工复核”语义的 need‑review 队列**：
   - 只记录“候选池无任何候选支持 `gt_label`（`label_match=True` 为 0）”的工单；
   - 不再把 rollout/解析/选择等硬故障写入人工复核队列。
2) **need‑review 同时产出两种形态**：
   - 流式：`need_review_queue.jsonl`（边跑边写，便于在线查看/中断恢复）
   - 汇总：`need_review.json`（run 结束时按 mission 聚合，便于直接阅读与统计）
3) **鼓励“无监督式规则学习”**（以 gt-label 为监督信号）：
   - `gt=fail & pred=pass`：鼓励反思总结 fail 场景/关键证据缺失，不以当前 `G0` 作为唯一标准（允许提出“缺关键证据则 fail”的假设规则）。
   - `gt=pass & pred=fail`：鼓励反思总结通过的例外/特殊情况/可忽略的跨任务负项，帮助学习每个 mission 真正在意的要点。
4) **必须具备“快速抛弃瞎猜规则”的机制**：
   - 如果某条规则只对当前 batch 有益、对后续 batch 负收益，应能通过 hit/miss 与自动清理尽快剔除（近似最小化全局损失的在线闭环）。
5) **Inference 输出协议不变**：仍为严格两行二分类（`Verdict/Reason`），不得引入第三状态词面（含 need‑review）。

## 变更概览（What）
### 输出工件（Artifacts）
- 保留：`need_review_queue.jsonl`，但语义改为“人工复核：候选池无任何候选支持 `gt_label`”。
- 新增：`need_review.json`（从 `need_review_queue.jsonl` 聚合生成）。
- 兼容性处理：`manual_review_queue.jsonl` 不再作为人工复核入口；硬故障仅保留在 `failure_malformed.jsonl` 与日志中（可选：保留空文件以兼容旧脚本）。

### Reflection（行为变更）
- 移除/弱化“基于 `G0` 的 need‑review quarantine 直接阻止学习”的策略，改为：
  - 默认尝试从错例中提出可泛化规则（允许“可能解释”）；
  - need‑review 的路由以“候选池是否存在支持 `gt_label` 的候选”为准；不依赖 `operations=[]`。
- 更新 reflection prompt：强化两种错例方向的归纳目标（fail 场景 vs pass 例外），并允许以“关键证据缺失”作为可学习信号（但需可被快速淘汰）。

### Guidance 生命周期（快速淘汰）
- 补齐/强化 rule 的 hit/miss 闭环：
  - 规则被应用后，后续窗口中的正确/错误将反馈到最近一次变更涉及的 rule keys；
  - 结合现有 `cleanup_low_confidence`，让低质量规则快速出局。

## 范围（Scope）
- 仅覆盖 Stage‑B training‑free runtime（`src/stage_b/`）：
  - reflection 路由与 need‑review 产物
  - reflection prompt（`configs/prompts/stage_b_reflection_prompt.txt`）
  - guidance hit/miss/清理策略
  - run 结束的 need‑review 聚合导出
  - 入口配置：Stage‑B 的入口配置文件为 `bbu_*.yaml`（例如 `configs/stage_b/bbu_line.yaml`），不再使用 `run.yaml`

## 非目标（Non‑Goals）
- 不改变 Stage‑A 的输入/输出契约与摘要生成方式。
- 不引入检索/外部知识库/embedding。
- 不在本变更中重写 Stage‑B 的整体推理协议（仍保持二分类两行输出）。

## 兼容性（Compatibility）
- `need_review_queue.jsonl` 文件名保留，但语义从“训练治理隔离”调整为“人工复核（候选池无任何候选支持 `gt_label`）”。
- `manual_review_queue.jsonl` 可能被废弃或仅保留空文件；下游脚本若仍依赖该文件，需要同步更新为读取新的 `need_review.json`/`need_review_queue.jsonl`。

## 风险与缓解
- 风险：允许“缺证据→fail”的假设规则，可能造成误拒（FP）上升。
  - 缓解：引入更快的 hit/miss 反馈与自动清理；对新规则设置更严格的淘汰阈值/更短窗口（由实现细化）。
- 风险：反思批处理是 bundle 级输出，可能导致一批内不同 ticket 的“可学/不可学”粒度不一致。
  - 缓解：在 runner 侧基于每 ticket 的候选支持情况进行细粒度 need‑review 路由（详见 design）。

## 成功标准（Success Criteria）
- need‑review 只包含“候选池无任何候选支持 `gt_label`”的工单；硬故障不进入该队列。
- 每次 run 结束产出 `need_review.json`，内容可直接用于人工复核与统计。
- 规则学习对两类错例方向均有产出倾向（fail 场景/通过例外），且低质量规则可在短周期内被淘汰。
