# Design — update-stageb-need-review-unification

## 1) 核心语义：need‑review = “候选池无任何候选支持 GT”

本变更将 need‑review 的语义统一为：
- **need‑review（人工复核）**：在给定 `gt_label`（参考答案）后，rollout 候选池中 **没有任何候选支持 GT**（`label_match=True` 的候选数为 0）。

明确排除：
- rollout/解析/选择的硬故障（format/no_candidates/selection_error 等）不进入 need‑review；仅写入 debug/failure 工件与日志。

## 2) 数据对象与粒度

- `GroupTicket`（工单）：一组 Stage‑A per-image summaries + mission + gt-label（你的业务输入单位）。
- `ExperienceRecord`（反思 case）：一个 ticket + 其 rollout 候选（Verdict/Reason）+ signals/元信息。
- `ExperienceBundle`：一次 reflection 批处理的多个 case（`batch_size`）。

目标：need‑review 的路由粒度必须是 **ticket/case 级**（而不是 bundle 级一刀切）。

## 3) 端到端流程（建议实现优先级：先最小闭环，再增强）

### 3.1 Rollout/Selection（保持不变）
- Inference 仍输出严格两行二分类，禁止第三状态词面。
- deterministic fail-first（当前依赖 `extract_mission_evidence(... mission_g0=G0)`）保持作为安全护栏，不作为反思学习的唯一标准。

### 3.2 Reflection：错例的学习目标（prompt 语义升级）

对每个 case，reflection 要求模型在看到 gt-label 后做“归纳式学习”，并允许“可能解释”：
- `gt=fail & pred=pass`：总结 fail 场景或“关键通过证据缺失/无法确认”的必要条件；允许提出保守规则（如缺关键证据则 fail）。
- `gt=pass & pred=fail`：总结通过的例外/特殊情况；明确哪些负项更可能属于其他 mission，不应干扰本 mission。

输出仍沿用现有 strict‑JSON ops（add/update/delete/merge），但 `has_evidence` 的含义应从“硬证据”调整为：
- **能否提出可泛化、可验证、可淘汰的假设规则**（hypothesis/learnability），而不是“是否存在完美证据”。

### 3.3 need‑review 路由（从“quarantine”改为“候选池无支持 GT 即入队”）

现状：reflection 内部存在 `need_review_quarantine`（基于 `G0` evidence 抽取）并可能直接阻止学习与入队。

新策略：**不在反思前置隔离阻断学习**；need‑review 作为“人工复核入口”，使用候选池的可支持性作为最小、可审计的触发条件。

建议的最小可落地判定：
- 对某个 ticket 来说：其 rollout 候选中 **没有任何** `label_match=True` 的候选能支持 `gt_label`
- 则该 ticket 写入 `need_review_queue.jsonl`。

说明：
- 路由在 runner 的 reflection flush 边界执行，便于携带 `reflection_id/reflection_cycle` 等审计字段，但不依赖 `operations=[]`。

## 4) need‑review 工件契约（两种形态）

### 4.1 流式：`need_review_queue.jsonl`

每行一个 ticket 记录，建议字段（最小集合）：
- `ticket_key`（建议与现有一致：`{group_id}::{gt_label}`）
- `group_id`
- `mission`
- `gt_label`
- `pred_verdict`（selection 的最终 verdict）
- `pred_reason`（selection 的最终 reason，单行）
- `reason_code`（例如：`no_candidate_supports_gt`）
- `reflection_id`、`reflection_cycle`
- `epoch`、`epoch_step`、`global_step`
- `note`（可选：来自 reflection 的简短分析文本；不得含第三状态词面）

### 4.2 汇总：`need_review.json`

在 mission run 结束时，由 `need_review_queue.jsonl` 聚合生成，建议结构：
```json
{
  "generated_at": "iso8601",
  "run_dir": "...",
  "missions": {
    "<mission>": {
      "count": 123,
      "tickets": [ /* 与 jsonl 相同字段 */ ]
    }
  }
}
```
实现上可以按“每 mission/run 目录”生成单 mission 文件，或在 run_name 根目录生成汇总；以保持最小改动为准。

## 5) 规则快速淘汰：hit/miss 闭环（最小实现）

目标：允许 reflection 提出“可能解释”的规则，但必须能快速剔除“只对当前 batch 有益、对后续负收益”的规则。

现有基础：
- rule metadata：`hit_count/miss_count/confidence`
- `cleanup_low_confidence(conf<threshold && miss>=k)`（epoch 末自动清理）
- runner 有 `last_applied_rule_keys`（最近一次 reflection 变更涉及的 rule keys）

建议的最小闭环：
1) 增补 `increment_hit_count(mission, keys)`（与 `increment_miss_count` 对称）。
2) 在 runner 中，当 `last_applied_rule_keys` 非空时：
   - 若某 ticket 的最终 verdict 与 `gt-label` 一致（label_match=True）→ 对 `last_applied_rule_keys` 记 hit
   - 若不一致 → 记 miss
3) 仍保留 `cleanup_low_confidence`，使得“猜错的规则”在较短 miss 累积后被删除。

> 注：该归因是“最近一次变更负责”的近似（online optimizer 视角），不保证严格因果，但足够支持快速试错与淘汰。

可选增强（若最小实现不够快再做）：
- 引入“信用窗口”（例如仅在 `N` 个 global_step 内对 last_applied_rule_keys 记 hit/miss），避免长期过度归因。

## 6) 与旧工件的兼容策略

- `manual_review_queue.jsonl` 不再承载人工复核语义；硬故障仅写入 `failure_malformed.jsonl`（及日志）。
- 若存在旧脚本依赖 `manual_review_queue.jsonl`，可选择：
  - 继续创建空文件以兼容读取逻辑；或
  - 在后续变更中同步更新脚本读取 `need_review.json`。
- 入口配置：Stage‑B 的入口配置文件为 `bbu_*.yaml`（例如 `configs/stage_b/bbu_line.yaml`），不再使用 `run.yaml`。

## 7) 验证建议（供 tasks 细化）

- 单元级：构造最小 bundle，验证 “候选池无任何候选支持 gt → 入 need_review”；“候选池存在支持 gt 的候选 → 不入队”。
- 回归级：在已有 Stage‑B run_dir 上离线重建 need_review.json，核对字段与数量。
- 行为级：对比变更前后 `need_review` 的内容纯度（不应包含硬故障原因），以及 rule 的自动淘汰是否发生。
