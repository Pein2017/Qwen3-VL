# Proposal: 强化 Stage-B 训练自由判定的保守护栏

## 背景
- Stage-A 摘要质量较高，但历史人工标签存在偶发粗心或模糊，误放行风险高。
- 现有 Stage-B 选择强依赖 label_match，缺少跨图必备项校验、不确定性处理和“标签冲突”反馈。
- 需求：在保持 training-free 与 guidance 驱动的前提下，引入轻量规则信号与冲突告警，优先防止误放行，并为人工校准提供回路。

## 目标
1) 保持 training-free & prompt-space 优化：不新增硬规则库，主要通过 prompt 提示模型更谨慎、聚焦任务要点，降低误放行。
2) 标签对齐优先，尤其 label=fail 时预测不得通过；当出现不确定信息时，优先提示人工复核而非放行。
3) 不确定性处理：识别“无法判断/只显示部分”等词面，结合任务相关性，在 Reason/输出中标记不确定并建议复核，但允许对非关键属性的软不确定保持通过。
4) 冲突反馈回路：当 LLM/critic 判断与历史标签矛盾时输出告警标记，供人工校准；无需硬编码必备项列表。
5) 反思触发扩展：纳入标签冲突、all-wrong、证据不足样本，依然以 guidance 作为唯一先验，其他信息由数据驱动自省。

## 约束与非目标
- 不新增手写规则文件，初始规则仍来自 `output_post/stage_b/initial_guidance.json`，其余由 LLM 反思迭代。
- 保持三行输出格式和现有文件布局；不引入外部检索/嵌入。
- 格式鲁棒性暂不深挖（qwen 已较稳定），仅在候选全被过滤时兜底。

## 影响面
- 代码：`src/stage_b/sampling/prompts.py`, `signals.py`, `scoring/selection.py`, `reflection/engine.py`（去除硬规则，强化 prompt 与不确定性/冲突标记流）。
- 数据/配置：无新增外部依赖；可复用 guidance 与现有 Stage-A 摘要。
- 文档：`docs/runtime/STAGE_B_RUNTIME.md`, `docs/reference/stage-B-knowledge-Chinese.md` 更新护栏与人工复核策略。

## 风险与缓解
- 过度依赖 LLM：保持 label=fail 兜底和不确定提示，必要时可加轻量警告而非强制规则。
- 误解析中文表述：使用短词表+正则+单测，遇到未识别时标记 unsure 而非放行。
- 反思过度触发：新增触发类型但保持 change_cap/batch_size 控制，避免过频更新。
