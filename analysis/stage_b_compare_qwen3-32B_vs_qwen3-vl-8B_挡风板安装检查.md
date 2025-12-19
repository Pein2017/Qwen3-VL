# Stage-B 对比报告：挡风板安装检查（qwen3-32B vs qwen3-vl-8B）

- 生成时间：`2025-12-17 01:39:40`
- 32B 输出目录：`output_post/stage_b/new_stage_a/挡风板安装检查`
- 8B 输出目录：`output_post/stage_b/new_stage_a-8b/挡风板安装检查`

> 注：我在当前 workspace 中检查时，发现 `output_post/stage_b/new_stage_a-8b/挡风板安装检查` 目录已不存在（可能被后续运行覆盖）；本报告的数值来自报告生成时刻该目录下的产物快照。

## 1) 执行摘要

- 在该 mission 的**metrics 最末 epoch**（qwen3-32B: 6 / qwen3-vl-8B: 6）上，`include_manual_review.acc`：qwen3-32B=0.8472，qwen3-vl-8B=0.8264。
- `exclude_manual_review.acc`：qwen3-32B=1.0000，qwen3-vl-8B=1.0000（注意 n 可能不同，见下表）。
- epoch=1 的 `label_match`（record-level）匹配率：qwen3-32B=0.8125 (n=144)，qwen3-vl-8B=0.8333 (n=144)。
- Reflection 概览：qwen3-32B=reflection 条目 30（eligible 30 / applied 30），常见 ineligible_reason=`NA`，常见 critique=`json_ops`；qwen3-vl-8B=reflection 条目 30（eligible 18 / applied 3），常见 ineligible_reason=`generation_error: No valid JSON found in reflection response`，常见 critique=`json_ops`。

> 说明：两次运行的 epoch 数量不同（本报告同时对齐对比 epoch=1 与各自最终 epoch）。

## 2) 详细对比

### 2.1 Rollouts / Selections 质量对比

| Model | Epoch | traj_groups | traj_candidates | format_ok | cand.pass% | sel_records | sel_groups | sel_units | sel.pass% | label_match | match_n | vote_p10 | vote_p50 | vote_p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-32B | 1 | 133 | 432 | 100.0% | 86.6% | 144 | 133 | 144 | 86.1% | 0.8125 | 144 | 1.000 | 1.000 | 1.000 |
| qwen3-32B | 6 | 133 | 431 | 100.0% | 89.1% | 144 | 133 | 144 | 89.6% | 0.8472 | 144 | 1.000 | 1.000 | 1.000 |
| qwen3-vl-8B | 1 | 133 | 429 | 100.0% | 91.6% | 144 | 133 | 144 | 91.0% | 0.8333 | 144 | 1.000 | 1.000 | 1.000 |
| qwen3-vl-8B | 6 | 133 | 427 | 100.0% | 93.4% | 144 | 133 | 144 | 93.1% | 0.8264 | 144 | 1.000 | 1.000 | 1.000 |

- `label_match` 为 `selections.jsonl` 的 record-level 指标；同一 `group_id` 可能因 reflection cycle 等出现多条记录。

**候选多样性（verdict 在组内/温度间是否变化）**

| Model | Epoch | Groups | mixed_verdict_groups | temp_sensitive_groups | avg_unique_verdicts | avg_temps |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3-32B | 1 | 133 | 8 (6.0%) | 4 (3.0%) | 1.06 | 3.00 |
| qwen3-32B | 6 | 133 | 8 (6.0%) | 4 (3.0%) | 1.06 | 2.99 |
| qwen3-vl-8B | 1 | 133 | 5 (3.8%) | 5 (3.8%) | 1.04 | 2.98 |
| qwen3-vl-8B | 6 | 133 | 5 (3.8%) | 4 (3.0%) | 1.04 | 2.97 |

### 2.2 Reflection / Guidance 过程对比

| Model | Epoch | n_reflection | eligible | applied | top_ineligible | top_action | top_ops | cache_files(summary/critique/plan) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-32B | 1 | 5 | 5 | 5 | `NA` | `refine` | `upsert:13` | 90 (30/30/30) |
| qwen3-32B | 6 | 5 | 5 | 5 | `NA` | `refine` | `upsert:13` | 90 (30/30/30) |
| qwen3-vl-8B | 1 | 5 | 3 | 0 | `generation_error: No valid JSON found in reflection response` | `noop` | `NA` | 65 (29/18/18) |
| qwen3-vl-8B | 6 | 5 | 3 | 1 | `non_conflict_bundle` | `noop` | `upsert:2` | 65 (29/18/18) |

- `guidance.json`：qwen3-32B：step=31, updated_at=2025-12-16T12:54:39.555188+00:00, experiences=1 (G0), snapshots=5；qwen3-vl-8B：step=4, updated_at=2025-12-16T14:44:26.218005+00:00, experiences=1 (G0), snapshots=5。
- `qwen3-vl-8B` snapshots（按文件名排序）step 序列（截断）：2, 3, 3, 4, 4。

### 2.3 性能指标对比（metrics_epoch.jsonl）

**qwen3-32B**
| epoch | exc.acc | exc.fn | exc.fp | exc.n | inc.acc | inc.fn | inc.fp | inc.n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.0000 | 0 | 0 | 117 | 0.8125 | 15 | 12 | 144 |
| 2 | 1.0000 | 0 | 0 | 116 | 0.8056 | 16 | 12 | 144 |
| 3 | 1.0000 | 0 | 0 | 118 | 0.8194 | 14 | 12 | 144 |
| 4 | 1.0000 | 0 | 0 | 119 | 0.8264 | 13 | 12 | 144 |
| 5 | 1.0000 | 0 | 0 | 119 | 0.8264 | 13 | 12 | 144 |
| 6 | 1.0000 | 0 | 0 | 122 | 0.8472 | 10 | 12 | 144 |

**qwen3-vl-8B**
| epoch | exc.acc | exc.fn | exc.fp | exc.n | inc.acc | inc.fn | inc.fp | inc.n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.0000 | 0 | 0 | 120 | 0.8333 | 10 | 14 | 144 |
| 2 | 1.0000 | 0 | 0 | 121 | 0.8403 | 10 | 13 | 144 |
| 3 | 1.0000 | 0 | 0 | 119 | 0.8264 | 9 | 16 | 144 |
| 4 | 1.0000 | 0 | 0 | 119 | 0.8264 | 9 | 16 | 144 |
| 5 | 1.0000 | 0 | 0 | 119 | 0.8264 | 9 | 16 | 144 |
| 6 | 1.0000 | 0 | 0 | 119 | 0.8264 | 9 | 16 | 144 |

### 2.4 Manual Review / Need Review 队列对比

- qwen3-32B：need_review_queue: 72（top tags: `insufficient_evidence`:42, `label_noise_suspect`:30）；manual_review_queue: 155（top tags: `(no_tag)`:155）。
- qwen3-vl-8B：need_review_queue: 91（top tags: `insufficient_evidence`:48, `label_noise_suspect`:43）；manual_review_queue: 294（top tags: `(no_tag)`:294）。

### 2.5 错误模式（failure_malformed.jsonl 等）

- qwen3-32B malformed：2（top: `(unknown)`:2）。
- qwen3-vl-8B malformed：24（top: `(unknown)`:24）。

## 3) 关键差异样例（各自内部：label_match=false）

**qwen3-32B / epoch=1**

| ticket_or_unit | group_id | gt_label | verdict | vote | warnings | reason |
| --- | --- | --- | --- | --- | --- | --- |
| QC-TEMP-20241026-0015113::pass | QC-TEMP-20241026-0015113 | pass | fail | 1.000 | ['label_mismatch'] | Image1: BBU设备/华为,只显示部分,机柜空间充足需要安装/这个BBU设备未按要求配备挡风板×1（明确负项）; Image2: BBU设备未显示品牌×1（待确认信号）; … |
| QC-TEMP-20241022-0015062::fail | QC-TEMP-20241022-0015062 | fail | pass | 1.000 | ['label_mismatch'] | Image1: 检测到BBU设备，机柜空间充足需要安装挡风板，挡风板显示完整且安装方向正确，符合要求；Image2: 挡风板部分显示，安装方向正确；Image3: 挡风板部分显示… |
| QC-20241217-0035172::fail | QC-20241217-0035172 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备/中兴,显示完整,无需安装; 螺丝、光纤插头/BBU端光纤插头,符合要求; 总结: 全局图明确检测到BBU设备且无需安装挡风板，无相关负项。 |
| QC-TEMP-20250110-0015857::pass | QC-TEMP-20250110-0015857 | pass | fail | 1.000 | ['label_mismatch'] | Image1: 检测到挡风板安装方向正确，但未提及BBU设备；Image2: 只显示部分挡风板，无法判断是否关联BBU；Image3: 未涉及BBU或挡风板安装需求；总结: 所有… |
| QC-TEMP-20241224-0015678::fail | QC-TEMP-20241224-0015678 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备检测到且显示完整，无需安装挡风板；无相关明确负项；总结: 符合要求。 |
| QC-TEMP-20240913-0014643::fail | QC-TEMP-20240913-0014643 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备检测到且机柜空间充足需要安装挡风板，并按要求配备了挡风板，螺丝和光纤插头符合要求；总结: 符合挡风板安装要求。 |
| QC-TEMP-20240926-0014738::fail | QC-TEMP-20240926-0014738 | fail | pass | 1.000 | ['label_mismatch'] | Image1: 检测到BBU设备，机柜空间充足需要安装挡风板，且已按要求配备，符合要求；Image2: 挡风板安装方向正确，BBU设备需安装挡风板，符合要求；Image3: BB… |
| QC-TEMP-20241206-0015501::fail | QC-TEMP-20241206-0015501 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备/中兴,显示完整,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板×1; 总结: 全局图明确检测到BBU设备并配备挡风板，符合要求。 |

**qwen3-vl-8B / epoch=1**

| ticket_or_unit | group_id | gt_label | verdict | vote | warnings | reason |
| --- | --- | --- | --- | --- | --- | --- |
| QC-TEMP-20241026-0015113::pass | QC-TEMP-20241026-0015113 | pass | fail | 1.000 | ['label_mismatch'] | Image1: BBUR设备/华为,只显示部分,机柜空间充足需要安装/这个BBU设备未按要求配备挡风板×1,备注:后续图片无显示,无法框选问题;BBU设备/华为,只显示部分,机柜… |
| QC-TEMP-20241022-0015062::fail | QC-TEMP-20241022-0015062 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备和挡风板皆只显示部分,但根据BBU设备可以判断机柜空间充足需要安装/这个BBU设备按要求配备了挡风板,且挡风板安装方向正确×1;总结: 由于无法判断BB… |
| QC-20241217-0035172::fail | QC-20241217-0035172 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备需安装/中兴,只显示部分,无需安装×1 |
| QC-20241105-0034263::pass | QC-20241105-0034263 | pass | fail | 1.000 | ['label_mismatch'] | 中兴RRU设备,只显示部分,机柜空间充足需要安装/这个RRU设备未按要求配备挡风板×1;无法判断Image1中的螺丝是否足够、BBU设备是否足够空间安装挡风板,所以无法判断中兴R… |
| QC-TEMP-20241224-0015678::fail | QC-TEMP-20241224-0015678 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备/爱立信,显示完整,无需安装×2 |
| QC-TEMP-20240913-0014643::fail | QC-TEMP-20240913-0014643 | fail | pass | 1.000 | ['label_mismatch'] | Image1: 这个BBU设备按要求配备了挡风板 |
| QC-TEMP-20240926-0014738::fail | QC-TEMP-20240926-0014738 | fail | pass | 1.000 | ['label_mismatch'] | Image1: 这个BBU设备按要求配备了挡风板; Image2: 机柜空间充足需要安装/这个BBU设备按要求配备了挡风板,显示完整,符合要求; Image3: 挡风板/华为,只… |
| QC-TEMP-20241206-0015501::fail | QC-TEMP-20241206-0015501 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备(中兴,显示完整),机柜空间充足需要安装/这个BBU设备按要求配备了挡风板×1 |

**qwen3-32B / epoch=6**

| ticket_or_unit | group_id | gt_label | verdict | vote | warnings | reason |
| --- | --- | --- | --- | --- | --- | --- |
| QC-TEMP-20241206-0015501::fail | QC-TEMP-20241206-0015501 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备/中兴,显示完整,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板×1; BBU设备/中兴,只显示部分,机柜空间充足需要安装/这个BBU设备按要求… |
| QC-TEMP-20240926-0014738::fail | QC-TEMP-20240926-0014738 | fail | pass | 1.000 | ['label_mismatch'] | Image1: 检测到BBU设备，机柜空间充足需要安装挡风板，且已按要求配备；Image2: 挡风板安装方向正确，BBU设备需安装挡风板；Image3: BBU设备与挡风板存在，… |
| QC-TEMP-20241018-0015024::pass | QC-TEMP-20241018-0015024 | pass | fail | 1.000 | ['label_mismatch'] | Image1: 检测到BBU设备且存在矛盾描述（“按要求配备了挡风板”与“未按要求配备挡风板”），但明确提及“需要安装”；Image2: 无法判断是否足够位置安装挡风板；Imag… |
| QC-TEMP-20241202-0015446::fail | QC-TEMP-20241202-0015446 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备/中兴,显示完整,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板×1; 总结: 全局图明确检测到BBU设备并已按要求安装挡风板，符合要求。 |
| QC-20241217-0035172::fail | QC-20241217-0035172 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备显示完整且无需安装挡风板，相关配件符合要求；总结: 全局图已明确无需安装挡风板且无负项，符合通过条件。 |
| QC-TEMP-20241224-0015688::fail | QC-TEMP-20241224-0015688 | fail | pass | 1.000 | ['label_mismatch', 'fail_first_irrelevant_hit'] | Image1: 无关图片; Image2: BBU设备存在但无法判断是否足够空间安装挡风板; Image3: BBU设备存在但无法判断是否足够空间安装挡风板; Image4: 显… |
| QC-TEMP-20250123-0016019::fail | QC-TEMP-20250123-0016019 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备检测到且空间充足需安装挡风板，挡风板安装方向正确且符合要求；总结: 全局图明确满足安装需求且无负项。 |
| QC-TEMP-20250102-0015779::pass | QC-TEMP-20250102-0015779 | pass | fail | 1.000 | ['label_mismatch'] | Image1: BBU设备检测到且按要求配备了挡风板; Image2: 挡风板安装方向正确，BBU设备部分显示且按要求配备挡风板; Image3: BBU设备部分显示但未按要求配… |

**qwen3-vl-8B / epoch=6**

| ticket_or_unit | group_id | gt_label | verdict | vote | warnings | reason |
| --- | --- | --- | --- | --- | --- | --- |
| QC-TEMP-20241028-0015135::fail | QC-TEMP-20241028-0015135 | fail | pass | 1.000 | ['label_mismatch'] | 无法判断BBU设备是否需要安装挡风板以及BBU设备是否符合要求,建议框选完整BBU设备拍摄近景照片,方便后续评估是否需要安装挡风板 |
| QC-TEMP-20241206-0015501::fail | QC-TEMP-20241206-0015501 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备华为,显示完整,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板,备注:由于品牌原因,无法判断上方及后方螺丝是否符合要求 |
| QC-TEMP-20240926-0014738::fail | QC-TEMP-20240926-0014738 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备只显示部分,但能判断品牌华为,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板,BBU端光纤插头符合要求;BBU设备/华为,只显示部分,机柜空间充… |
| QC-TEMP-20241018-0015024::pass | QC-TEMP-20241018-0015024 | pass | fail | 1.000 | ['label_mismatch'] | Image1: 无法判断品牌和是否足够位置安装BBU设备/这个BBU设备按要求配备了挡风板",备注:无法判断品牌和是否足够位置安装挡风板×1,无法判断品牌和是否足够空间安装挡风板… |
| QC-TEMP-20241202-0015446::fail | QC-TEMP-20241202-0015446 | fail | pass | 1.000 | ['label_mismatch'] | Image1:标签/可以识别×1，光纤/有保护措施,弯曲半径合理/蛇形管×4，螺丝、光纤插头/BBU安装螺丝,符合要求×2，螺丝、光纤插头/机柜处接地螺丝,符合要求×1，BBU设… |
| QC-20241217-0035172::fail | QC-20241217-0035172 | fail | pass | 1.000 | ['label_mismatch'] | Image1: BBU设备需安装/这个BBU设备按要求配备了挡风板×1 |
| QC-TEMP-20240802-0014359::pass | QC-TEMP-20240802-0014359 | pass | fail | 1.000 | ['label_mismatch'] | Image1: BBU设备需安装/这个BBU设备按要求配备了挡风板,但BBU设备未显示完整,无法判断是否符合要求×1,备注:无法判断品牌×1,Image2: BBU设备需安装/这… |
| QC-TEMP-20241224-0015688::fail | QC-TEMP-20241224-0015688 | fail | pass | 1.000 | ['label_mismatch', 'fail_first_irrelevant_hit'] | Image1: 无关图片; Image2: BBU设备/华为,只显示部分,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板,备注:无法判断品牌和是否足够空间安装挡风板; I… |

## 4) 关键发现

- 该 mission 上，qwen3-32B 最终 `include_manual_review.acc` 高于 qwen3-vl-8B（0.8472 vs 0.8264）。
- qwen3-vl-8B reflection 存在 `generation_error`，需检查反思 prompt/解析稳定性。
- 最终 `guidance.step` 不同：qwen3-32B=31，qwen3-vl-8B=4（结合 snapshots 可进一步追踪演化路径）。
- 队列规模差异：need_review qwen3-32B=72 vs qwen3-vl-8B=91；manual_review qwen3-32B=155 vs qwen3-vl-8B=294。

## 5) 建议

- 就该 mission 的这次对比结果而言，qwen3-vl-8B 暂不建议直接替代 qwen3-32B（最终 `include_manual_review` 指标落后）。
- 替代决策建议至少覆盖多个 missions/数据切片（不同品牌/遮挡/只显示部分/无关图比例），避免单 mission 偏差。
- 若继续用 Stage-B training-free 迭代：优先修复/降低 reflection 的解析失败（例如 `generation_error`），否则 guidance 学习几乎不可用。
- 对齐对比时建议固定相同 epoch 数/相同 reflection 策略（或显式关闭 reflection），否则“最终表现”会混入训练回合数差异。

## 6) 附录：文件存在性与规模

| artifact | qwen3-32B | qwen3-vl-8B |
| --- | --- | --- |
| failure_malformed.jsonl | Y | Y |
| group_report.jsonl | Y | Y |
| guidance.json | Y | Y |
| manual_review_queue.jsonl | Y | Y |
| metrics_epoch.jsonl | Y | Y |
| need_review_queue.jsonl | Y | Y |
| reflection.jsonl | Y | Y |
| reflection_cache/ | Y | Y |
| selections.jsonl | Y | Y |
| snapshots/ | Y | Y |
| trajectories.jsonl | Y | Y |
