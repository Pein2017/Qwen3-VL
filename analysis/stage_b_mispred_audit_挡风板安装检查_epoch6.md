# Stage-B 误判审计：挡风板安装检查（epoch=6）

- 生成时间：`2025-12-17 01:26:04`
- 32B run：`output_post/stage_b/new_stage_a/挡风板安装检查`
- 8B run：`output_post/stage_b/new_stage_a-8b/挡风板安装检查`
- Stage-A 输入：`output_post/stage_a/挡风板安装检查_stage_a.jsonl`
- 原始图片：`group_data/bbu_scene_2.0_order/挡风板安装检查`

> 注：我在当前 workspace 中检查时，发现 `output_post/stage_b/new_stage_a-8b/挡风板安装检查` 目录已不存在（可能被后续运行覆盖）；本报告的数值来自报告生成时刻该目录下的产物快照。

## 0) Prompt / 业务逻辑（用于解释误判）

- Stage-B rollout prompt 入口：`src/stage_b/sampling/prompts.py`。
- 该 prompt 的关键点：
  - **fail-first**：若摘要出现与当前 G0 相关的明确负项（如“未按要求/未配备/未安装/方向不正确/不符合要求”等），整组判不通过。
  - **多图协同**：`obj` 最大倾向全局图；局部图“只显示部分/无法判断”本身不应否决全局已明确确认的要点；若关键要点在所有图片都无法明确确认则判不通过。
  - **不允许第三态**：输出只能是“通过/不通过”，但 pipeline 会把部分样本送入 `manual_review_queue.jsonl`。

> 你提到的 `guidance.jsonl` 在这两次输出目录里未找到；本次运行的 guidance 以 `guidance.json` + `snapshots/guidance-*.json` 的形式存在。

## 1) epoch=6 总体误判概览

| model | units(n) | wrong(n) | acc | pass->fail(fn) | fail->pass(fp) |
| --- | --- | --- | --- | --- | --- |
| 32B | 144 | 22 | 0.8472 | 10 | 12 |
| 8B | 144 | 25 | 0.8264 | 9 | 16 |
| union | - | 29 |  -  | - | - |
| intersection | - | 18 |  -  | - | - |

## 2) 自动对账：GT vs 标注 JSON（粗略）

对 epoch=6 的误判集合（union=29）做了一个“**只看标注 JSON 里 bbu_equipment/bbu_sheild_direction**”的粗略对账：

- `ann_pass_gt_pass`：标注倾向通过 & GT=pass（但模型判错）
- `ann_pass_gt_fail`：标注倾向通过 & GT=fail（高度怀疑 GT/标注口径不一致，或存在未编码的 fail 条件）
- `ann_fail_gt_fail`：标注明确有“未配备挡风板/方向错误”等 & GT=fail（更像 Stage-A 摘要遗漏负项导致 Stage-B 判错）
- `ann_none_*`：该 ticket 的图片没有对应 .json 标注文件（只能靠图片 + 摘要人工判断）

- `ann_pass_gt_pass`: 10
- `ann_none_gt_fail`: 7
- `ann_pass_gt_fail`: 7
- `ann_none_gt_pass`: 3
- `ann_fail_gt_fail`: 2

## 3) 误判明细表（epoch=6 union=29）

| ticket_key | gt | pred32 | pred8 | nimg | unc | neg_shield | ann_files | ann_expected | ann_missing | ann_dir_bad | ann_no_need | ann_has_shield | ann_need_install |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QC-20241031-0034159::pass | pass | fail | fail | 3 | 1 | 1 | 2 | pass | 0 | 0 | 2 | 2 | 2 |
| QC-20241203-0034864::fail | fail | fail | pass | 8 | 0 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |
| QC-20241217-0035172::fail | fail | pass | pass | 1 | 0 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |
| QC-TEMP-20240802-0014359::pass | pass | pass | fail | 3 | 1 | 0 | 2 | pass | 0 | 0 | 0 | 4 | 4 |
| QC-TEMP-20240913-0014643::fail | fail | pass | pass | 1 | 0 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |
| QC-TEMP-20240926-0014738::fail | fail | pass | pass | 4 | 1 | 0 | 2 | pass | 0 | 0 | 0 | 2 | 2 |
| QC-TEMP-20241018-0015024::pass | pass | fail | fail | 3 | 1 | 1 | 1 | pass | 0 | 0 | 0 | 0 | 0 |
| QC-TEMP-20241022-0015062::fail | fail | pass | pass | 3 | 1 | 0 | 1 | fail | 2 | 0 | 0 | 2 | 4 |
| QC-TEMP-20241022-0015062::pass | pass | fail | fail | 1 | 1 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |
| QC-TEMP-20241026-0015113::pass | pass | pass | fail | 3 | 1 | 1 | 2 | pass | 0 | 0 | 4 | 6 | 6 |
| QC-TEMP-20241028-0015135::fail | fail | fail | pass | 3 | 0 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |
| QC-TEMP-20241031-0015171::pass | pass | fail | fail | 3 | 1 | 1 | 2 | pass | 0 | 0 | 2 | 6 | 6 |
| QC-TEMP-20241101-0015173::pass | pass | fail | fail | 3 | 1 | 1 | 1 | pass | 0 | 0 | 2 | 0 | 0 |
| QC-TEMP-20241102-0015177::pass | pass | fail | fail | 3 | 1 | 1 | 1 | pass | 0 | 0 | 0 | 2 | 2 |
| QC-TEMP-20241107-0015210::fail | fail | pass | pass | 3 | 1 | 0 | 3 | pass | 0 | 0 | 0 | 6 | 6 |
| QC-TEMP-20241111-0015238::fail | fail | fail | pass | 4 | 0 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |
| QC-TEMP-20241122-0015316::fail | fail | fail | pass | 2 | 1 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |
| QC-TEMP-20241202-0015446::fail | fail | pass | pass | 1 | 0 | 0 | 1 | pass | 0 | 0 | 0 | 2 | 2 |
| QC-TEMP-20241203-0015455::pass | pass | pass | fail | 3 | 1 | 0 | 2 | pass | 0 | 0 | 0 | 4 | 4 |
| QC-TEMP-20241206-0015501::fail | fail | pass | pass | 1 | 1 | 0 | 1 | pass | 0 | 0 | 0 | 2 | 2 |
| QC-TEMP-20241206-0015501::pass | pass | fail | pass | 2 | 1 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |
| QC-TEMP-20241206-0015502::fail | fail | pass | pass | 1 | 1 | 0 | 1 | pass | 0 | 0 | 4 | 0 | 0 |
| QC-TEMP-20241209-0015519::fail | fail | pass | pass | 3 | 1 | 0 | 1 | pass | 0 | 0 | 0 | 2 | 2 |
| QC-TEMP-20241212-0015545::pass | pass | fail | pass | 3 | 1 | 1 | 2 | pass | 0 | 0 | 4 | 2 | 2 |
| QC-TEMP-20241224-0015678::fail | fail | pass | pass | 1 | 0 | 0 | 1 | pass | 0 | 0 | 2 | 0 | 0 |
| QC-TEMP-20241224-0015688::fail | fail | pass | pass | 5 | 1 | 0 | 4 | fail | 2 | 0 | 6 | 2 | 4 |
| QC-TEMP-20250102-0015779::pass | pass | fail | pass | 3 | 1 | 1 | 2 | pass | 0 | 0 | 0 | 4 | 4 |
| QC-TEMP-20250123-0016019::fail | fail | pass | pass | 1 | 0 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |
| QC-TEMP-20250123-0016019::pass | pass | fail | pass | 2 | 1 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |

## 4) 重点样例（带证据）

### QC-TEMP-20241022-0015062::fail

- GT: `fail` | 32B: `pass` | 8B: `pass`
- 32B warnings: `['label_mismatch']`
- 32B reason: Image1: 检测到BBU设备，机柜空间充足需要安装挡风板，且挡风板安装方向正确，符合要求；Image2: 挡风板显示部分，安装方向正确；Image3: 挡风板显示部分，安装方向正确；总结: 全局图明确需安装挡风板且符合要求，局部图无明确负项，符合通过条件。
- 8B warnings: `['label_mismatch']`
- 8B reason: Image1: BBU设备和挡风板符合要求;BBU设备和挡风板符合要求;BBU设备和挡风板符合要求;BBU设备和挡风板符合要求;BBU设备和挡风板符合要求;BBU设备和挡风板符合要求;BBU设备和挡风板符合要求。

**Stage-A 摘要（per_image）**

- image_1: 标签/无法识别×1，BBU设备/中兴,只显示部分,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板×2，螺丝、光纤插头/BBU安装螺丝,符合要求×2，螺丝、光纤插头/BBU端光纤插头,符合要求×7，挡风板/中兴,显示完整,安装方向正确×1，光纤/有保护措施,弯曲半径合理/蛇形管×2，BBU设备/中兴,显示完整,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板×2
- image_2: 挡风板/中兴,只显示部分,安装方向正确×1
- image_3: 挡风板/中兴,只显示部分,安装方向正确×1

**图片路径（便于你复核）**

- QC-TEMP-20241022-0015062_4176693.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241022-0015062/QC-TEMP-20241022-0015062_4176693.jpeg`
- QC-TEMP-20241022-0015062_4176694.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241022-0015062/QC-TEMP-20241022-0015062_4176694.jpeg`
- QC-TEMP-20241022-0015062_4176695.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241022-0015062/QC-TEMP-20241022-0015062_4176695.jpeg`

**标注 JSON 摘要（bbu_equipment / bbu_sheild_direction）**

- `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241022-0015062/QC-TEMP-20241022-0015062_4176693.json`
  - bbu_equipment: {'bbu_equipment_true/bbu_equipment_true_true': 2, 'bbu_equipment_true/bbu_equipment_true_false': 2, 'bbu_equipment_true': 4}
  - bbu_sheild_direction: {'bbu_sheild_direction_true': 2}

**初步判断**

- 更像：**Stage-A 摘要遗漏了“未配备挡风板”等负项**（因为标注 JSON 明确存在 true_false）。

### QC-TEMP-20241224-0015688::fail

- GT: `fail` | 32B: `pass` | 8B: `pass`
- 32B warnings: `['label_mismatch', 'fail_first_irrelevant_hit']`
- 32B reason: Image1: 无关图片; Image2: BBU设备存在但无法判断是否足够空间安装挡风板; Image3: BBU设备存在但无法判断是否足够空间安装挡风板; Image4: 显示完整BBU设备且明确标注无需安装挡风板; Image5: 显示完整BBU设备并配备挡风板，安装方向正确; 总结: 全局图Image4和Im…
- 8B warnings: `['label_mismatch', 'fail_first_irrelevant_hit']`
- 8B reason: Image1: 无关图片; Image2: BBU设备/华为,只显示部分,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板,备注:无法判断品牌和是否足够空间安装挡风板; Image3: BBU设备/华为,只显示部分,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板,备注:无法判断品牌和是否足够空间安装挡风…

**Stage-A 摘要（per_image）**

- image_1: 无关图片
- image_2: 标签/4G-BBU-电源线×2，标签/4G-BBU-电源线×1，BBU设备/需复核,备注:无法判断品牌和是否足够空间安装挡风板
- image_3: 标签/4G-BBU-电源线×2，标签/4G-BBU-电源线×1，BBU设备/需复核,备注:无法判断品牌和是否足够空间安装挡风板×1，螺丝、光纤插头/BBU端光纤插头,符合要求×2
- image_4: 标签/无法识别×4，光纤/有保护措施,弯曲半径合理/铠装×2，螺丝、光纤插头/BBU端光纤插头×2，BBU设备/华为,显示完整,无需安装×1，光纤/有保护措施,弯曲半径合理/蛇形管×3，螺丝、光纤插头/BBU端光纤插头×1，标签/4G-AAU1-光纤×1，标签/4G-BBU-传输光纤×1，螺丝、光纤插头/BBU端光纤插头×1，光纤/有保护措施,弯曲半径合理/蛇形管×4，螺丝、光纤插头/BBU端光纤插头×1，标签/4G-BBU-电源线×1，标签/4G-BBU-电源线×1，螺丝、光纤插头/机柜处接地螺丝,符合要求×1，螺丝、光纤插头/BBU安装螺丝,符合要求×1
- image_5: 标签/无法识别×4，螺丝、光纤插头/BBU安装螺丝,符合要求×2，螺丝、光纤插头/BBU端光纤插头,符合要求×7，BBU设备/华为,显示完整,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板×2，光纤/有保护措施,弯曲半径合理/铠装×3，光纤/有保护措施,弯曲半径合理/蛇形管×3，挡风板/华为,显示完整,安装方向正确×2，螺丝、光纤插头/BBU安装螺丝,符合要求×1，螺丝、光纤插头/BBU端光纤插头,符合要求×1，光纤/有保护措施,弯曲半径合理/蛇形管×5，螺丝、光纤插头/BBU端光纤插头,符合要求×1，光纤/有保护措施,弯曲半径不合理(弯曲半径4cm处应该套蛇形管保护)/蛇形管×3，螺丝、光纤插头/BBU端光纤插头,符合要求×1，光纤/有保护措施,弯曲半径合理/铠装×1，光纤/有保护措施,弯曲半径合理/蛇形管×3，螺丝、光纤插头/BBU端光纤插头,符合要求×2

**图片路径（便于你复核）**

- QC-TEMP-20241224-0015688_4404200.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241224-0015688/QC-TEMP-20241224-0015688_4404200.jpeg`
- QC-TEMP-20241224-0015688_4405335.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241224-0015688/QC-TEMP-20241224-0015688_4405335.jpeg`
- QC-TEMP-20241224-0015688_4405382.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241224-0015688/QC-TEMP-20241224-0015688_4405382.jpeg`
- QC-TEMP-20241224-0015688_4405427.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241224-0015688/QC-TEMP-20241224-0015688_4405427.jpeg`
- QC-TEMP-20241224-0015688_4405858.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241224-0015688/QC-TEMP-20241224-0015688_4405858.jpeg`

**标注 JSON 摘要（bbu_equipment / bbu_sheild_direction）**

- `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241224-0015688/QC-TEMP-20241224-0015688_4405335.json`
  - bbu_equipment: {'bbu_equipment_false': 2}
- `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241224-0015688/QC-TEMP-20241224-0015688_4405382.json`
  - bbu_equipment: {'bbu_equipment_false': 2, 'bbu_equipment_true/bbu_equipment_true_false': 2, 'bbu_equipment_true': 2}
- `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241224-0015688/QC-TEMP-20241224-0015688_4405427.json`
  - bbu_equipment: {'bbu_equipment_false': 2}
- `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241224-0015688/QC-TEMP-20241224-0015688_4405858.json`
  - bbu_equipment: {'bbu_equipment_true/bbu_equipment_true_true': 2, 'bbu_equipment_true': 2}
  - bbu_sheild_direction: {'bbu_sheild_direction_true': 4}

**初步判断**

- 更像：**Stage-A 摘要遗漏了“未配备挡风板”等负项**（因为标注 JSON 明确存在 true_false）。

### QC-TEMP-20241101-0015173::pass

- GT: `pass` | 32B: `fail` | 8B: `fail`
- 32B warnings: `['label_mismatch']`
- 32B reason: Image1: BBU设备/华为,只显示部分,机柜空间充足需要安装/这个BBU设备未按要求配备挡风板×1; Image2: BBU设备/华为,只显示部分,无需安装×1; Image3: 无相关证据; 总结: Image1 明确指出“未按要求配备挡风板”，属于明确负项，导致整组不通过。
- 8B warnings: `['label_mismatch']`
- 8B reason: Image1: BBU设备/华为,只显示部分,机柜空间充足需要安装/这个BBU设备未按要求配备挡风板×1，BBU设备/华为,只显示部分,机柜空间充足需要安装/无法判断是否符合要求×1

**Stage-A 摘要（per_image）**

- image_1: 标签/无法识别×3，螺丝、光纤插头/BBU安装螺丝,符合要求×1，螺丝、光纤插头/BBU端光纤插头,符合要求×3，光纤/有保护措施,弯曲半径合理/铠装×2，光纤/有保护措施,弯曲半径合理/同时有蛇形管和铠装×1，BBU设备/华为,只显示部分,机柜空间充足需要安装/这个BBU设备未按要求配备挡风板×1
- image_2: 标签/无法识别×1，BBU设备/华为,只显示部分,无需安装×1，螺丝、光纤插头/BBU安装螺丝,符合要求×1
- image_3: 标签/无法识别×1

**图片路径（便于你复核）**

- QC-TEMP-20241101-0015173_4231176.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核通过/QC-TEMP-20241101-0015173/QC-TEMP-20241101-0015173_4231176.jpeg`
- QC-TEMP-20241101-0015173_4231177.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核通过/QC-TEMP-20241101-0015173/QC-TEMP-20241101-0015173_4231177.jpeg`
- QC-TEMP-20241101-0015173_4231178.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核通过/QC-TEMP-20241101-0015173/QC-TEMP-20241101-0015173_4231178.jpeg`

**标注 JSON 摘要（bbu_equipment / bbu_sheild_direction）**

- `group_data/bbu_scene_2.0_order/挡风板安装检查/审核通过/QC-TEMP-20241101-0015173/QC-TEMP-20241101-0015173_4231176.json`
  - bbu_equipment: {'bbu_equipment_false': 2}

**初步判断**

- 更像：**Stage-A 摘要把“未配备挡风板”写进来了**（但标注 JSON/GT 都倾向通过），导致 Stage-B 按 fail-first 误杀。

### QC-20241031-0034159::pass

- GT: `pass` | 32B: `fail` | 8B: `fail`
- 32B warnings: `['label_mismatch']`
- 32B reason: Image1: BBU设备/中兴,只显示部分,无需安装; Image2: BBU设备/华为,只显示部分,无需安装,但存在“这个BBU设备未按要求配备挡风板”的明确负项; Image3: BBU设备/华为,显示完整,无需安装; 总结: Image2中存在与G0相关的明确负项“未按要求配备挡风板”，导致整组不通过。
- 8B warnings: `['label_mismatch']`
- 8B reason: Image2: 无法判断是否足够空间安装挡风板;BBU设备未拍摄完整,无法判断是否需要安装以及是否符合要求×1

**Stage-A 摘要（per_image）**

- image_1: BBU设备/中兴,只显示部分,无需安装×1
- image_2: BBU设备/华为,只显示部分,无需安装×1，螺丝、光纤插头/BBU安装螺丝,符合要求×2，BBU设备/华为,只显示部分,机柜空间充足需要安装/这个BBU设备未按要求配备挡风板×1
- image_3: 标签/无法识别×3，BBU设备/华为,显示完整,无需安装×1，螺丝、光纤插头/BBU安装螺丝,符合要求×2，螺丝、光纤插头/BBU端光纤插头,符合要求×2，光纤/有保护措施,弯曲半径合理/蛇形管×1，螺丝、光纤插头/ODF端光纤插头,符合要求×1

**图片路径（便于你复核）**

- QC-20241031-0034159_4291387.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核通过/QC-20241031-0034159/QC-20241031-0034159_4291387.jpeg`
- QC-20241031-0034159_4291388.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核通过/QC-20241031-0034159/QC-20241031-0034159_4291388.jpeg`
- QC-20241031-0034159_4291389.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核通过/QC-20241031-0034159/QC-20241031-0034159_4291389.jpeg`

**标注 JSON 摘要（bbu_equipment / bbu_sheild_direction）**

- `group_data/bbu_scene_2.0_order/挡风板安装检查/审核通过/QC-20241031-0034159/QC-20241031-0034159_4291388.json`
  - bbu_equipment: {'bbu_equipment_true/bbu_equipment_true_true': 2, 'bbu_equipment_true': 2}
  - bbu_sheild_direction: {'bbu_sheild_direction_true': 2}
- `group_data/bbu_scene_2.0_order/挡风板安装检查/审核通过/QC-20241031-0034159/QC-20241031-0034159_4291389.json`
  - bbu_equipment: {'bbu_equipment_false': 2}

**初步判断**

- 更像：**Stage-A 摘要把“未配备挡风板”写进来了**（但标注 JSON/GT 都倾向通过），导致 Stage-B 按 fail-first 误杀。

### QC-TEMP-20241206-0015502::fail

- GT: `fail` | 32B: `pass` | 8B: `pass`
- 32B warnings: `['label_mismatch']`
- 32B reason: Image1: BBU设备/华为,显示完整,无需安装×2; 总结: 全局图明确检测到BBU设备且无需安装挡风板，无相关明确负项。
- 8B warnings: `['label_mismatch']`
- 8B reason: Image1:标签/无法识别×4，光纤/有保护措施,弯曲半径合理/蛇形管×3，螺丝、光纤插头/BBU安装螺丝,符合要求×2，螺丝、光纤插头/BBU安装螺丝,符合要求×2，BBU设备/华为,显示完整,无需安装×2，BBU设备/华为,只显示部分,无需安装×2

**Stage-A 摘要（per_image）**

- image_1: 标签/无法识别×4，光纤/有保护措施,弯曲半径合理/蛇形管×3，螺丝、光纤插头/BBU安装螺丝,符合要求×2，螺丝、光纤插头/BBU安装螺丝,符合要求×2，BBU设备/华为,显示完整,无需安装×2，BBU设备/华为,只显示部分,无需安装×2

**图片路径（便于你复核）**

- QC-TEMP-20241206-0015502_4348975.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241206-0015502/QC-TEMP-20241206-0015502_4348975.jpeg`

**标注 JSON 摘要（bbu_equipment / bbu_sheild_direction）**

- `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241206-0015502/QC-TEMP-20241206-0015502_4348975.json`
  - bbu_equipment: {'bbu_equipment_false': 4}

**初步判断**

- 更像：**GT label/口径可能有问题**（标注 JSON 倾向通过，但 GT=fail）。建议人工复核是否有未编码的 fail 条件（如必须出现全局图/两端固定等）。

### QC-TEMP-20241224-0015678::fail

- GT: `fail` | 32B: `pass` | 8B: `pass`
- 32B warnings: `['label_mismatch']`
- 32B reason: Image1: BBU设备检测到且显示完整，明确标注“无需安装”挡风板，无相关明确负项；总结: 符合通过条件。
- 8B warnings: `['label_mismatch']`
- 8B reason: Image1: BBU设备/爱立信,显示完整,无需安装×1

**Stage-A 摘要（per_image）**

- image_1: 标签/无法识别×1，标签/无法识别×1，BBU设备/爱立信,显示完整,无需安装×1，螺丝、光纤插头/BBU端光纤插头,符合要求×2，光纤/有保护措施,弯曲半径合理/铠装×1，光纤/有保护措施,弯曲半径合理/铠装×2，螺丝、光纤插头/BBU端光纤插头,符合要求×1，光纤/有保护措施,弯曲半径合理/铠装×2，螺丝、光纤插头/BBU端光纤插头,符合要求×1，光纤/有保护措施,弯曲半径合理/铠装×1，螺丝、光纤插头/BBU端光纤插头,符合要求×1，光纤/有保护措施,弯曲半径合理/铠装×1，BBU设备/爱立信,显示完整,无需安装×1

**图片路径（便于你复核）**

- QC-TEMP-20241224-0015678_4399607.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241224-0015678/QC-TEMP-20241224-0015678_4399607.jpeg`

**标注 JSON 摘要（bbu_equipment / bbu_sheild_direction）**

- `group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241224-0015678/QC-TEMP-20241224-0015678_4399607.json`
  - bbu_equipment: {'bbu_equipment_false': 2}

**初步判断**

- 更像：**GT label/口径可能有问题**（标注 JSON 倾向通过，但 GT=fail）。建议人工复核是否有未编码的 fail 条件（如必须出现全局图/两端固定等）。

### QC-TEMP-20241206-0015501::pass

- GT: `pass` | 32B: `fail` | 8B: `pass`
- 32B warnings: `['label_mismatch']`
- 32B reason: Image1: 只显示部分挡风板，无法确认整体是否符合要求; Image2: 未提及挡风板或BBU设备，无法判断是否需安装挡风板; 总结: 未明确检测到BBU设备及挡风板安装情况，无法满足关键检查项要求。
- 8B warnings: `[]`
- 8B reason: Image1: 无法判断品牌,无法判断是否需要安装/这个BBU设备按要求配备了挡风板×1

**Stage-A 摘要（per_image）**

- image_1: 挡风板/中兴,只显示部分,安装方向正确×1
- image_2: 螺丝、光纤插头/BBU安装螺丝,符合要求×1，螺丝、光纤插头/机柜处接地螺丝,符合要求×1，标签/5G-避雷器接地线×1

**图片路径（便于你复核）**

- QC-TEMP-20241206-0015501_4455238.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核通过/QC-TEMP-20241206-0015501/QC-TEMP-20241206-0015501_4455238.jpeg`
- QC-TEMP-20241206-0015501_4455239.jpeg: `group_data/bbu_scene_2.0_order/挡风板安装检查/审核通过/QC-TEMP-20241206-0015501/QC-TEMP-20241206-0015501_4455239.jpeg`

**初步判断**

- 更像：**证据覆盖不足**（只有局部/不清晰），需要明确你的业务口径：这种情况应当算 pass 还是应转人工。

## 5) 建议（下一步怎么验证“GT vs 模型”）

- 先优先人工复核 `ann_fail_gt_fail` 的 2 个样本：这些最可能是**真实 fail**，但 Stage-A 摘要没暴露负项给 Stage-B。
- 对 `ann_pass_gt_fail` 的 7 个样本：建议对齐“挡风板任务”的口径（是否必须有全局图？是否多 BBU 时任一不合格即 fail？）后再决定是否改 GT。
- 对 `ann_pass_gt_pass` 的 10 个样本：建议回溯 Stage-A 摘要生成逻辑（为什么会生成“未按要求配备挡风板”），否则 Stage-B 再强也会被输入噪声带偏。
- 如果你愿意，我可以再做一份“带缩略图”的审计 Markdown/HTML，把这 29 个 ticket 的图片直接嵌进报告里，方便你/标注同学快速复核。
