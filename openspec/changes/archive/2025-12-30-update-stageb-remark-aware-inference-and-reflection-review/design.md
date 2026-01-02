# Design — update-stageb-remark-aware-inference-and-reflection-review

## 设计原则
1) **输出协议稳定**：inference 永远只输出两行二分类（`Verdict:` / `Reason:`），不引入第三状态；need‑review 仅在内部工件中出现。
2) **软信号 vs 硬护栏分离**：
   - “需复核,备注:”与“无法确认/模糊/只显示部分”等属于 **软信号**（提示需要更谨慎/需要找证据），不得直接硬否决；
   - “明确负项”通过 **跨 mission 通用负向触发词表** 做 hard fail‑first（确定性护栏），但必须 **以当前 mission 的 `G0` 为作用域**：只对与 `G0` 相关的负项执行 fail‑first，避免同一组工单在不同 mission 间互相污染。
3) **mission‑specific 只来自 G0**：prompt 不提供任何全局名词词表；关键检查项的语义边界由 `guidance.experiences["G0"]` 决定。
4) **安全约束（弱证据不放行）**：若无法从摘要/备注给出支持通过的证据，则判不通过；这不是把“需复核=不通过”，而是避免“无证据通过”。

## 输入信号（Stage‑A summary 的“需复核 + 备注”）
样例（来自 `data/bbu_full_768_poly-need_review/val.jsonl`）：
- `光纤/需复核,备注:部分未套蛇形管×2`
- `BBU设备/需复核,备注:拍摄范围原因,不能确认挡风板情况×1`

设计要求：
- prompt 中明确：`需复核` 只表示“该条信息需要结合备注/其它图综合判断”，本身不等价于不通过。
- 模型需优先读取 `备注:`，并把备注内容当作证据的一部分（可能支持通过，也可能支持不通过）。

## 负项触发词（Fail‑first, 跨 mission 通用）
### 触发词表（MUST include）
> 只包含动词/形容词/短语；不含名词。
- 未按要求
- 错误
- 缺失
- 松动
- 损坏
- 方向不正确
- 反向
- 不符合要求
- 不合格
- 不合理
- 未安装
- 未配备

### 执行规则
- 在 group‑level 进行 **mission‑scoped fail‑first**：**任一图片 summary 命中任一触发词（且不是被否定语境抵消），并且该负项与当前 mission 的 `G0` 相关 ⇒ 最终 verdict 强制为不通过**。
- “待确认信号”（无法确认/无法判断/只显示部分/模糊/需复核等）不得进入 fail‑first 触发词表。

### 否定语境排除（降低 FP）
为避免 “无错误/未发现错误” 触发误拒，触发词检测应支持最小否定前缀排除（示例，不限于）：
- `无{触发词}`、`未发现{触发词}`、`未见{触发词}`、`不存在{触发词}`、`未见明显{触发词}`

## Prompt 结构（inference）
建议将 system prompt 拆为 3 段（顺序固定）：
1) **任务要点（G0）**：直接注入 `guidance.experiences["G0"]`，作为“关键检查项”的唯一来源。
2) **通用 fail‑first 负项触发词**：列出上面的触发词表与“任一图命中即整组不通过”的规则。
3) **需复核/待确认解释 + 安全约束**：
   - 解释 `需复核,备注:`：需读备注；`需复核` 不等价 fail。
   - 解释“无法确认/模糊/部分可见”等：是待确认信号；不得直接作为硬触发词。
   - **安全约束**：若无法从摘要/备注给出支持通过的证据（覆盖 G0 的关键点），必须判不通过。
   - **mission 作用域强调**：只根据当前 mission 的 `G0` 判定；与本 mission 无关的负项/异常不得影响结论。同一组工单可能被不同 mission 审核，不同 mission 下允许出现不同 verdict。

注意：
- prompt 中不得出现全局名词示例列表（避免强耦合）；如需例子，仅允许用 “关键要点/必需项/安装状态/方向/完整性/合规性” 等抽象词。
- output 保持严格两行：`Verdict:` + `Reason:`；Reason 单行、限制长度、不得包含第三状态措辞；如发生 fail‑first 覆盖，Reason 需重写以与最终 Verdict 一致。

## Reflection / 训练阶段的 need‑review 与证据一致性
### 目标
- 把“标签噪声/Stage‑A summary 噪声/证据不足”的样本从“可学习规则”中隔离出来，避免把噪声学成规则导致系统性误拒。

### 机制（提案）
1) **证据提取（非名词化）**：基于 Stage‑A summaries/备注做轻量检测：
   - `explicit_negative_hits`: 是否命中 fail‑first 触发词（强证据）
   - `pending_signals`: 是否出现 “需复核/无法确认/无法判断/模糊/只显示部分”等（弱信号）
2) **一致性判定**
   - `gt=fail` 且 `explicit_negative_hits` 为空，且模型倾向通过（多候选投票或选中候选为 pass） ⇒ 进入 need‑review（疑似 label_noise 或 stageA_noise），reflection 不生成学习 fail 的 operations。
   - `gt=pass` 但 `explicit_negative_hits` 非空 ⇒ 进入 need‑review（疑似 label_noise），reflection 不生成放行规则（避免把明确负项学成“也能通过”）。
3) **工件输出（仅内部）**
   - 仅新增/使用 JSONL：`need_review_queue.jsonl`，记录 `{group_id, mission, gt_label, evidence_summary, chosen_verdict, reason, tag}`（不再使用 `label_or_stageA_noise.jsonl`）。
   - 这些工件不得回流到 inference 的两行输出中，也不得写入 guidance 文本（guidance 继续禁止第三状态措辞）。

### 与现有 `no_evidence_for_label` 的关系
现有 reflection 三段式提示已支持输出 `no_evidence_for_label`；本提案将其扩展为：
- 细分为 `label_noise_suspect` / `stageA_noise_suspect` / `insufficient_evidence`（用于训练数据治理）
- 并将“噪声样本不学习规则”设为硬约束。
