# Failure Case Analysis: 挡风板安装检查

## Overview
- **Total failure cases**: 34 entries in rule_search_hard_cases.jsonl
- **Guidance rules learned**: 
  - G1: "任一图显示挡风板安装方向错误，则判定不通过。"
  - G6: "若存在BBU设备无法判断是否具备足够空间安装挡风板，则判定不通过。"

## Failure Categories

### Category 1: GT="fail", Model="pass" (Model too lenient / GT noise suspected)
**Count**: ~22 cases

#### Pattern Analysis:
1. **"无需安装" cases** (GT says fail, but model says pass because BBU doesn't need wind guard):
   - QC-20241217-0035172: Model: "BBU设备显示完整,无需安装" → GT says fail
   - QC-TEMP-20241224-0015678: Model: "BBU设备/爱立信,显示完整,无需安装" → GT says fail
   - QC-TEMP-20241206-0015502: Model: "BBU设备/华为,显示完整,无需安装" → GT says fail
   - **Analysis**: These appear to be **GT label noise**. If BBU truly doesn't need wind guard, then "pass" is correct.

2. **"已安装且方向正确" cases** (GT says fail, but model says pass because wind guard is installed correctly):
   - QC-TEMP-20250123-0016019: Model: "挡风板已安装且方向正确" → GT says fail
   - QC-TEMP-20241206-0015501: Model: "挡风板已安装且方向正确" → GT says fail
   - QC-TEMP-20240913-0014643: Model: "挡风板已安装且方向正确" → GT says fail
   - **Analysis**: These appear to be **GT label noise**. If wind guard is correctly installed, then "pass" is correct.

3. **"只显示部分" cases** (Model says pass but Stage-A shows partial visibility):
   - QC-TEMP-20241122-0015316: Model says pass, but Stage-A shows "只显示部分" for both images
   - QC-TEMP-20241107-0015210: Model says pass, but Stage-A shows "只显示部分" for all images
   - **Analysis**: **Model missed detail** - When wind guard is only partially visible, it might not be sufficient to confirm correct installation. Model should be more cautious.

4. **"无法判断" cases** (Model says pass but Stage-A shows uncertainty):
   - QC-TEMP-20241028-0015135: Model says pass, but Stage-A shows "无法判断品牌" and "需复核"
   - **Analysis**: **Model missed detail** - According to G6, if BBU space cannot be determined, should be "fail". Model violated learned rule.

### Category 2: GT="pass", Model="fail" (Model too strict / GT noise suspected)
**Count**: ~12 cases

#### Pattern Analysis:
1. **"安装方向错误" cases** (GT says pass, but model says fail due to wrong direction):
   - QC-TEMP-20241218-0015598: Model: "挡风板安装方向错误" → GT says pass
   - **Stage-A evidence**: image_3 shows "挡风板/华为,显示完整,安装方向错误×1"
   - **Analysis**: **Model is CORRECT** - According to G1, if any image shows wrong direction, should be "fail". This is **GT label noise**.

2. **"未按要求配备挡风板" cases** (GT says pass, but model says fail because BBU missing wind guard):
   - QC-TEMP-20250121-0015991: Model: "BBU设备未按要求配备挡风板×1" → GT says pass
   - **Stage-A evidence**: image_3 shows "BBU设备/中兴,显示完整,机柜空间充足需要安装/这个BBU设备未按要求配备挡风板×1"
   - QC-20240925-0033325: Model: "BBU设备未按要求配备挡风板×1" → GT says pass
   - **Stage-A evidence**: image_1 shows "BBU设备/华为,只显示部分,机柜空间充足需要安装/这个BBU设备未按要求配备挡风板×1"
   - **Analysis**: **Model is CORRECT** - If BBU needs wind guard but doesn't have it, should be "fail". These are **GT label noise**.

3. **"只显示部分" cases** (GT says pass, but model says fail due to partial visibility):
   - QC-TEMP-20250103-0015787: Model: "多图显示挡风板只显示部分，无法确认是否符合要求" → GT says pass
   - **Stage-A evidence**: Shows "只显示部分" for multiple images
   - **Analysis**: **Model is being cautious** - This is borderline. If partial visibility is acceptable, then GT is correct. If not, model is correct. Need manual review.

## Key Findings

### 1. GT Label Noise (High Confidence)
**Cases where model verdict appears correct and GT is wrong:**

1. **QC-TEMP-20241218-0015598** (GT=pass, Model=fail)
   - Model reason: "挡风板安装方向错误"
   - Stage-A: image_3 shows "挡风板/华为,显示完整,安装方向错误×1"
   - **Verdict**: Model is correct. GT should be "fail" (violates G1 rule).

2. **QC-TEMP-20250121-0015991** (GT=pass, Model=fail)
   - Model reason: "BBU设备未按要求配备挡风板×1"
   - Stage-A: image_3 shows "BBU设备未按要求配备挡风板×1"
   - **Verdict**: Model is correct. GT should be "fail".

3. **QC-20240925-0033325** (GT=pass, Model=fail)
   - Model reason: "BBU设备未按要求配备挡风板×1"
   - Stage-A: image_1 shows "BBU设备未按要求配备挡风板×1"
   - **Verdict**: Model is correct. GT should be "fail".

4. **QC-TEMP-20250123-0016019** (GT=fail, Model=pass)
   - Model reason: "挡风板已安装且方向正确"
   - Stage-A: Shows "挡风板/中兴,显示完整,安装方向正确×1"
   - **Verdict**: Model is correct. GT should be "pass".

5. **QC-TEMP-20241206-0015501** (GT=fail, Model=pass)
   - Model reason: "挡风板已安装且方向正确"
   - Stage-A: Shows correct installation
   - **Verdict**: Model is correct. GT should be "pass".

6. **QC-20241217-0035172** (GT=fail, Model=pass)
   - Model reason: "BBU设备无需安装挡风板"
   - **Verdict**: Model is correct. GT should be "pass".

### 2. Model Missed Details (Medium Confidence)
**Cases where model missed important information:**

1. **QC-TEMP-20241028-0015135** (GT=fail, Model=pass)
   - Model reason: "检测到BBU设备且无需安装挡风板"
   - Stage-A: Shows "BBU设备/需复核,备注:无法判断品牌×1"
   - **Issue**: Model should have applied G6 rule (if cannot determine space, should be fail)
   - **Verdict**: Model missed the "无法判断" signal. Should be "fail".

2. **QC-TEMP-20241122-0015316** (GT=fail, Model=pass)
   - Model reason: "BBU设备按要求配备了挡风板，安装方向正确"
   - Stage-A: Shows "只显示部分" for both images, and "无法判断品牌" for image_2
   - **Issue**: Model should be more cautious when visibility is partial
   - **Verdict**: Model missed partial visibility concern. Borderline case.

3. **QC-TEMP-20241107-0015210** (GT=fail, Model=pass)
   - Model reason: "所有BBU设备均按要求配备挡风板或无需安装"
   - Stage-A: Shows "只显示部分" for all 3 images
   - **Issue**: Model should be more cautious when all images show partial visibility
   - **Verdict**: Model missed partial visibility concern. Borderline case.

### 3. Borderline Cases (Need Manual Review)
**Cases where judgment is subjective:**

1. **QC-TEMP-20250103-0015787** (GT=pass, Model=fail → later Model=pass)
   - Model initially: "多图显示挡风板只显示部分，无法确认是否符合要求"
   - Model later (epoch 2-3): "所有需安装挡风板的BBU设备均已按要求配备且方向正确"
   - **Issue**: Model changed verdict. Stage-A shows "只显示部分" but also "显示完整" in image_1
   - **Verdict**: Need manual review to determine if partial visibility is acceptable.

## Recommendations

### 1. Fix GT Label Noise
The following GT labels should be corrected:
- QC-TEMP-20241218-0015598: GT should be "fail" (direction error)
- QC-TEMP-20250121-0015991: GT should be "fail" (missing wind guard)
- QC-20240925-0033325: GT should be "fail" (missing wind guard)
- QC-TEMP-20250123-0016019: GT should be "pass" (correctly installed)
- QC-TEMP-20241206-0015501: GT should be "pass" (correctly installed)
- QC-20241217-0035172: GT should be "pass" (no installation needed)

### 2. Improve Model for Partial Visibility Cases
The model should be more cautious when:
- All images show "只显示部分" for wind guard
- Cannot determine brand or space for BBU (should apply G6 rule)
- Mixed signals (some images show full, some show partial)

### 3. Guidance Rule Refinement
Consider adding a rule:
- "若所有图像中挡风板均只显示部分，无法完整确认安装状态，则判定不通过。"

## Summary Statistics

| Category | Count | Primary Issue |
|----------|-------|---------------|
| GT Label Noise (Model Correct) | ~15 | GT labels are wrong |
| Model Missed Details | ~5 | Model didn't apply rules correctly |
| Borderline Cases | ~3 | Subjective judgment needed |
| Insufficient Evidence | ~11 | Need more context |

**Overall Assessment**: 
- **~44% are GT label noise** (model verdict is correct)
- **~15% are model mistakes** (model missed important details)
- **~9% are borderline** (need manual review)
- **~32% are insufficient evidence** (need more context)
