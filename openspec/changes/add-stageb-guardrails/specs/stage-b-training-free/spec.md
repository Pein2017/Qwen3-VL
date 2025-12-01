## MODIFIED Requirements

### Requirement: Stage-B SHALL evaluate group tickets without gradient updates by orchestrating a training-free sampling and judging loop.
Stage-B SHALL remain training-free and rely on prompt-space guidance plus label alignment（尤其 label=fail 时预测不得通过）而非硬编码必备规则；模型通过提示聚焦任务要点、慎对不确定证据来减少误放行。关键要素/非关键要素的区分 SHALL 由 LLM 按提示自行判断，不做关键词/规则抽取。
#### Scenario: Prompt-guided cautious reasoning without hard rules
- WHEN Stage-B loads mission guidance and Stage-A summaries
- THEN it SHALL build prompts强调：仅使用与当前 mission focus 直接相关的要素判断；对不确定或与 focus 无关的信息给出“建议人工复核/不作为判定依据”的倾向
- AND selection SHALL prioritize label alignment (label=fail 覆盖一切放行可能)，其余由 LLM/critic 的理由与不确定提示驱动；无硬编码 `rule_fail` 列表。

### Requirement: Stage-B SHALL emit reproducible verdict artifacts that satisfy existing downstream contracts.
Selections和轨迹输出 SHALL 携带 conflict/复核信号（无硬规则 verdict），保证审计与下游可解释性。若历史标签为 fail，输出 verdict MUST 仍为 fail（或 fail+manual_review），可以在 Reason 中注明“强烈怀疑标签噪声”但不得放行。
#### Scenario: Verdicts encode rule signals and conflict flags
- WHEN selections are written, THEN each record SHALL include `conflict_flag` and `needs_manual_review` so downstream reviewers can trace为何被阻拦。
- AND Reason text SHALL mention evidence gaps或关键任务要素缺失/不确定时驱动 fail/复核。
- AND IF the historical label is `fail`, THEN the exported verdict SHALL be `fail`（或 fail+manual_review），任何放行倾向 MUST 以 `conflict_flag` 标记并被阻断。
#### Scenario: Conflict-driven reflection prompts and audits
- WHEN a candidate sets `conflict_flag=true` (e.g., label=fail but model wants pass), THEN reflection prompting SHALL explicitly (a) 优先检索 `focus` 关键要点缺失并提出最小 upsert/merge；(b) 若找不到负项则在 `uncertainty_note` 标注疑似标签/摘要噪声并允许仅提出“标记人工复核/提高不确定性”操作；否则返回 noop。
- AND an offline audit tool SHALL surface groups repeatedly marked `conflict_flag=true` that never receive applied reflection ops, so operators can manually review labels或回溯 Stage-A 摘要。

### Requirement: Stage-B critic outputs MAY be half-structured text and SHALL be normalized to canonical JSON downstream.
Critic/rollout prompts MAY ask模型按“键: 值”半结构行（SUMMARY/CRITIQUE/VERDICT/NEEDS_RECHECK/EVIDENCE_SUFFICIENCY/RECOMMENDED_ACTION），系统 MUST 负责宽松解析并落盘为规范 JSON；模型不得因为格式偏差而被强制重试。
#### Scenario: Half-structured critic tolerated with lossless normalization
- WHEN critic returns fenced、双花括号或“SUMMARY: …”行格式，而非严格 JSON 对象，THEN the ingestion layer SHALL strip包装/截取首个平衡花括号或按键名提取字段，填充缺省值后落地为同样字段集合的 JSON，避免因格式噪声丢失保守信号。
- AND prompts SHALL仍强调“只输出结构化字段，不要额外文本”，但解析 SHALL 容忍多余前后缀（如 ```、{{ }}）并优先保留 verdict/recommended_action/needs_recheck 信息。
- AND rollout parsing SHALL operate per-ticket inside the batch loop; if all samples for a group are filtered out, the system MAY inject a single `sampling_failed` placeholder (fail verdict, zero confidence) to keep the pipeline from aborting while logging the anomaly and skipping reflection for that group.

### Requirement: Stage-B SHALL record minimal deterministic signals for each candidate to support LLM reflection.
Stage-B SHALL 仅维护 label_match/self_consistency/confidence 等最小确定性信号；不依赖手写正则或词面规则去强制降置信或自动 fail，不确定性由 LLM Reason/Critic 体现。
#### Scenario: Minimal signals without regex heuristics
- WHEN candidates are attached with deterministic signals, THEN only label_match/self_consistency/confidence SHALL be stored; no regex-based uncertainty flags are required。
- AND any“不确定/模糊/部分可见”等表述 SHALL 由 LLM 在 Reason/critique 中给出，系统可在后续解析时选择性使用，但不做硬规则裁决。

### Requirement: Stage-B SHALL treat LLM reflection-guided experiences updates as the optimizer step with direct application.
Stage-B reflection eligibility SHALL 纳入标签冲突和证据不足案例，仍以现有 guidance 为唯一规则来源，不依赖硬规则列表。
#### Scenario: Reflection eligibility considers label conflicts and evidence sufficiency
- WHEN building reflection bundles, THEN eligibility SHALL include batches containing `conflict_flag=true`, all-wrong groups, or tickets marked `needs_manual_review` due to evidence insufficiency, in addition to existing label_match-based rules.
- AND reflection proposals SHALL preserve the “guidance-first” source: no新外部规则列表；建议仍通过 guidance 文本演化。
- AND mission focus SHALL stay isolated: reflection SHALL evaluate conflicts/insufficiency per mission, allowing the same Stage-A summaries to yield different verdicts across missions without cross-mission leakage。

### Requirement: Stage-B SHALL enforce label-fail safety without hiding model reasoning.
安全兜底不得遮蔽模型原始推理；即使导出 verdict 因标签兜底被改写，原始 verdict/reason 与 conflict/uncertainty 信号 MUST 保留以支持反思与审计。
#### Scenario: Transparent safety override
- WHEN ticket.label=fail but a candidate verdict=pass, THEN the exported verdict MAY be overridden to fail for safety, BUT trajectories SHALL保留候选的原始 verdict/reason，且 `conflict_flag` 与相关保守信号 SHALL 导出供反思/人工复核。
- AND reflection SHALL be allowed to ingest这些冲突样本并提出“补充关键要点”或“疑似标签/摘要噪声、标记人工复核”的经验，而非丢弃样本或压制 prompt 优化空间。
