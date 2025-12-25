# Stage-B guidance audit (2025-12-24)

## Objective
- 审计 Stage-A 摘要输入（固定不变），结合既有 Stage-B rule-search 的 hard cases，区分 **可学习/可拯救** 与 **不可观测噪声**。
- 在不改 Stage-A 的前提下，提出对 `output_post/stage_b/initial_guidance.json` 的可执行优化点。
- 对 BBU 场景：将通用约束尽量上移到共享 prompt，减少每个 mission guidance 的重复与 token 占用。

## Inputs
- Stage-A JSONL
  - `output_post/stage_a_bbu_rru_summary_12-22/BBU接地线检查_stage_a.jsonl`
  - `output_post/stage_a_bbu_rru_summary_12-22/BBU线缆布放要求_stage_a.jsonl`
  - `output_post/stage_a_bbu_rru_summary_12-22/BBU安装方式检查（正装）_stage_a.jsonl`
  - `output_post/stage_a_bbu_rru_summary_12-22/挡风板安装检查_stage_a.jsonl`
- Stage-B hard cases (来自既有运行产物)
  - `output_post/stage_b/BBU接地线检查/12-19-bbu-line-rule-search-train-eval/rule_search_hard_cases.jsonl`
  - `output_post/stage_b/BBU接地线检查/12-19-bbu-line-rule-search-train-eval/rule_search_hard_samples.jsonl`（含 `gt_fail_reason_text`）
  - `output_post/stage_b/BBU线缆布放要求/12-19-bbu-cable-rule-search-train-eval/rule_search_hard_cases.jsonl`
  - `output_post/stage_b/挡风板安装检查/12-22-ocr/rule_search_hard_cases.jsonl`

## Key findings

### BBU接地线检查
- hard cases 总计 442（缺 11 条 Stage-A 对应 ticket_key，需后续核对数据对齐）。
- 误放行（GT=fail, pred=pass）= 292：
  - **可拯救（可学习缺证据）= 60**：主要模式是“只有机柜处接地螺丝合格/有捆扎，但缺 `地排处接地螺丝,符合要求`”。
  - **疑似噪声/不可观测 = 189**：摘要已满足关键证据，但 GT=fail（`rule_search_hard_samples.jsonl` 的 fail reason 多为“补拍/标签/线径/站点名”等不可观测维度）。
  - **不确定 = 43**：摘要满足“地排+捆扎”，但缺“机柜处接地螺丝合格”；是否应作为通过条件存在语义不确定性。
- 误拦截（GT=pass, pred=fail）= 150：
  - **可拯救（证据齐全但被误判）= 116**：摘要包含“地排处接地螺丝合格 + 电线捆扎整齐”，但 baseline 仍给出不通过，疑似被干扰项/备注/旧规则误导。

### BBU线缆布放要求
- hard cases 总计 500。
- 误放行（GT=fail, pred=pass）= 497：
  - **可拯救（可学习缺证据）= 326**：主要缺口是 **ODF 端插头合格证据缺失**（占 270/326），其次是光纤保护/弯曲半径证据缺失或出现“无保护措施”。
  - **疑似噪声/不可观测 = 171**：摘要满足“BBU端+ODF端插头合格+光纤保护/弯曲半径合理”，但 GT=fail。

### 挡风板安装检查
- hard cases 中以 **GT=pass, pred=fail 的误拦截**为主：可拯救 86，疑似噪声 17。
- 可拯救模式高度集中：Stage-A 已明确给出“按要求配备挡风板 + 安装方向正确”（或“无需安装”），但 Stage-B 仍以“缺少全局判断依据/视角不足/证据不足”判不通过 → 属于 guidance 对“可观测关键短语”的使用缺失。

### BBU安装方式检查（正装）
- 该任务的最新 Stage-B 运行产物不完整（`output_post/stage_b/BBU安装方式检查（正装）/12-19-bbu-line-rule-search-train-eval/baseline_metrics_steps.jsonl` 仅到 batch 78/86，未落盘 wrong_cases/hard_cases），因此这里先做 **label vs Stage-A 可观测证据** 的 learnability 审计。
- 结合现有 `ticket_filter.exclude_ticket_keys_path`（`output_post/stage_b/filters/bbu_install_exclude_ticket_keys.txt`）后：
  - 保留数据中可学习通过（BBU设备 + BBU安装螺丝合格）= 7855
  - 保留数据中可学习不通过（缺关键证据）= 324
  - 仍残留疑似不可学习 pass = 20（多数为“pass 但缺 BBU安装螺丝合格证据”）
  - `需复核` 类备注在可学习 pass 中仍较多（1352），易诱发模型误拦截，需在 guidance 中显式降权。

## Outputs
- 详细可拯救 ticket 列表（含 `per_image` 与特征）
  - `analysis/stageb_guidance_audit_2025-12-24/bbu_line_saveable.jsonl`
  - `analysis/stageb_guidance_audit_2025-12-24/bbu_cable_saveable.jsonl`
  - `analysis/stageb_guidance_audit_2025-12-24/bbu_wind_saveable.jsonl`
- 可拯救 ticket_key 纯列表
  - `analysis/stageb_guidance_audit_2025-12-24/bbu_line_saveable_ticket_keys.txt`
  - `analysis/stageb_guidance_audit_2025-12-24/bbu_cable_saveable_ticket_keys.txt`
  - `analysis/stageb_guidance_audit_2025-12-24/bbu_wind_saveable_ticket_keys.txt`
- 噪声候选（建议用于 `ticket_filter.exclude_ticket_keys_path`）
  - `analysis/stageb_guidance_audit_2025-12-24/bbu_line_noise_ticket_keys.txt`
  - `analysis/stageb_guidance_audit_2025-12-24/bbu_cable_noise_ticket_keys.txt`
  - `analysis/stageb_guidance_audit_2025-12-24/bbu_wind_noise_ticket_keys.txt`
  - `analysis/stageb_guidance_audit_2025-12-24/bbu_install_noise_ticket_keys.txt`
- 汇总统计
  - `analysis/stageb_guidance_audit_2025-12-24/summary.json`

## Guidance updates applied
- 已将 BBU 通用约束上移到共享 prompt：`src/prompts/stage_b_verdict.py` 新增 `【BBU任务通用补充提示】`。
- 已在 `output_post/stage_b/initial_guidance.json` 对四个 BBU mission 做了“精简 + 只保留任务关键证据口径”的更新：
  - **BBU接地线检查**：补充“证据不可替代”（机柜处不能替代地排处）与“明确负项优先”。
  - **BBU线缆布放要求**：补充“明确负项优先（无保护措施/弯曲半径不合理）”与 ODF 端证据口径（包含 `ODF端光纤插头、光纤/无要求,符合要求` 变体）。
  - **BBU安装方式检查（正装）**：补充“证据口径只看 BBU设备 + BBU安装螺丝合格”、显式 pass/fail 规则、并将“品牌/挡风板/接地线/标签/光纤”等定义为干扰项（避免需复核备注误导）。
  - **挡风板安装检查**：补充对 Stage-A 稳定短语（“机柜空间充足需要安装/按要求配备挡风板/未按要求配备挡风板/安装方向正确/无需安装”）的显式判定规则，并增加“矛盾表述→不通过”的一致性约束。

## Noise triage: “大量无关图片但GT=pass”
- 现有数据中存在 `GT=pass` 但同一工单内 “无关图片” 占比较高的情况（例如一半以上图片为 `无关图片×1`），按约定可视为提交/采集噪声，需要单独甄别（建议：从训练/规则学习池剔除，而不是在 guidance 中硬拟合）。
- 已导出候选列表（按 `无关图片` 图片占比阈值分桶）：
  - 汇总：`analysis/stageb_guidance_audit_2025-12-24/bbu_pass_high_irrelevant_candidates.json`
  - 每个 mission 的 ticket_key 列表与样例：见同目录下 `*_pass_irrel_ge_*.txt` 与 `*_examples.jsonl`
## Recommended next actions
1. 使用现有 config 复跑 baseline（建议 `jump_reflection=true`）验证误放行是否下降：
   - `conda run -n ms bash scripts/stage_b.sh config=bbu_line`
   - `conda run -n ms bash scripts/stage_b.sh config=bbu_cable`
   - `conda run -n ms bash scripts/stage_b.sh config=bbu_wind`
2. 对 **BBU安装方式检查（正装）** 建议先补齐一次完整的 baseline 落盘（wrong_cases/hard_cases），便于进一步做“可拯救工单”精确定位（当前目录为中断态）。
3. 若目标是提高可学习性（避免噪声污染 gate），可将 `analysis/.../bbu_*_noise_ticket_keys.txt` 写入 config 的 `ticket_filter.exclude_ticket_keys_path` 再跑一次对比。
