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
- WHEN selections are written, THEN each record SHALL include `conflict_flag` and `needs_manual_review` derived from uncertainty/insufficient evidence so downstream reviewers can trace为何被阻拦。
- AND Reason text SHALL mention evidence gaps或关键任务要素缺失/不确定时驱动 fail/复核。
- AND IF the historical label is `fail`, THEN the exported verdict SHALL be `fail`（或 fail+manual_review），任何放行倾向 MUST 以 `conflict_flag` 标记并被阻断。
#### Scenario: Conflict-driven reflection prompts and audits
- WHEN a candidate sets `conflict_flag=true` (e.g., label=fail but model wants pass), THEN reflection prompting SHALL explicitly (a) 优先检索 `focus` 关键要点缺失并提出最小 upsert/merge；(b) 若找不到负项则在 `uncertainty_note` 标注疑似标签/摘要噪声并允许仅提出“标记人工复核/提高不确定性”操作；否则返回 noop。
- AND an offline audit tool SHALL surface groups repeatedly marked `conflict_flag=true` that never receive applied reflection ops, so operators can manually review labels或回溯 Stage-A 摘要。

### Requirement: Stage-B SHALL record minimal deterministic signals for each candidate to support LLM reflection.
Stage-B SHALL 在保持最小信号集的前提下，引入对不确定性词面的信号标记，避免误放行，且无需硬规则裁决。
#### Scenario: Uncertainty cues lower confidence and flag manual review
- WHEN summaries contain uncertainty cues（“需复核/无法判断/只显示部分/无法识别”），THEN Stage-B SHALL lower candidate confidence (or set to null), set `needs_manual_review=true`, and carry this flag into signals/selection/trajectories.
- AND such tickets SHALL NOT be auto-pass unless the uncertain cue is for a non-critical attribute of the current mission and no negative evidence is present; critical-attribute uncertainty MUST trigger manual_review and block auto-pass.
- AND uncertainty cues SHALL be filtered by mission relevance defined in guidance/prompt focus, not via硬编码规则。

### Requirement: Stage-B SHALL treat LLM reflection-guided experiences updates as the optimizer step with direct application.
Stage-B reflection eligibility SHALL 纳入标签冲突和证据不足案例，仍以现有 guidance 为唯一规则来源，不依赖硬规则列表。
#### Scenario: Reflection eligibility considers label conflicts and evidence sufficiency
- WHEN building reflection bundles, THEN eligibility SHALL include batches containing `conflict_flag=true`, all-wrong groups, or tickets marked `needs_manual_review` due to evidence insufficiency, in addition to existing label_match-based rules.
- AND reflection proposals SHALL preserve the “guidance-first” source: no新外部规则列表；建议仍通过 guidance 文本演化。
- AND mission focus SHALL stay isolated: reflection SHALL evaluate conflicts/insufficiency per mission, allowing the same Stage-A summaries to yield different verdicts across missions without cross-mission leakage。

### Requirement: Stage-B SHALL enforce label-fail safety without hiding model reasoning.
安全兜底不得遮蔽模型原始推理；即使导出 verdict 因标签兜底被改写，原始 verdict/reason 与 conflict/uncertainty 信号 MUST 保留以支持反思与审计。
#### Scenario: Transparent safety override
- WHEN ticket.label=fail but a candidate verdict=pass, THEN the exported verdict MAY be overridden to fail for safety, BUT trajectories SHALL保留候选的原始 verdict/reason，且 `conflict_flag` 与 `uncertainty_notes` SHALL 导出供反思/人工复核。
- AND reflection SHALL be allowed to ingest这些冲突样本并提出“补充关键要点”或“疑似标签/摘要噪声、标记人工复核”的经验，而非丢弃样本或压制 prompt 优化空间。
