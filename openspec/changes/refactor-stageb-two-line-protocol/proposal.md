# Change Proposal — refactor-stageb-two-line-protocol

## Why
- Current Stage-B rollout enforces四行格式 with Evidence_Positive/Negative JSON数组，频繁因换行/截断触发解析 WARNING，存在格式脆弱性。
- 业务方希望支持更宽泛的任务类型（不一定有“证据列表”），要求用更简洁的输出协议，并去掉旧字段的兼容/遗留处理。
- 需要一次性重构管线，移除旧 evidence 依赖，减少解析风险。

## What (high level)
- 定义并切换到“两行协议”：`Verdict: 通过|不通过` + `Reason: <自然语言>（单行，无证据数组）`。
- 移除 evidence 相关字段/解析/导出/提示；所有处理链路改用原始文本（verdict+reason）驱动 selection 与 reflection。
- 更新 Stage-B prompts、解析、selection、reflection、导出和相关配置/文档；不保留任何旧格式兼容层。

## Impact / Risk
- **Breaking**：`selections.jsonl` / `trajectories.jsonl` / reflection prompts格式变化；下游若依赖 evidence_* 字段需同步调整。
- 风险：反思规则生成可用信号减少（无 evidence）；需调整反思逻辑以仅基于 reason/label/选中候选。
- 验证需要至少一次端到端运行（可用 debug 配置）。

## Success Criteria
- Stage-B 生成和解析不再产生“Rollout parse failed / Salvaged”类 WARNING。
- 产物文件中无 `evidence_positive`/`evidence_negative`/相关空字段；仅含 verdict+reason（和必要元数据）。
- reflection 能运行并产生有效 plan（即便基于 reason），无因字段缺失而报错。
