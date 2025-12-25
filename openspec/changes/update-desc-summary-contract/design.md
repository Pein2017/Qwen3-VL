# Design: update-desc-summary-contract

## Context
This change formalizes a new description and summary contract for BBU/RRU conversion outputs. The goal is to remove slash-delimited descs, eliminate 需复核, and emit summaries as JSON strings that are easier for the model to learn. The plan preserves OCR content characters '-' and '/' and keeps geometry JSON formatting stable (with spaces).

## Design Goals
- Deterministic, parseable desc format: key=value, comma-separated, no spaces.
- Clear BBU vs. RRU separation: BBU has remarks, no groups; RRU has optional groups, no remarks.
- JSON summary strings that enumerate statistics without ×N tokens.
- Preserve OCR content (keep '-' and '/') while keeping desc parse-safe.
- Keep geometry JSON formatting stable (spaces in JSON text) to avoid tokenizer drift.

## Description Grammar
### General
- Format: key=value pairs separated by ASCII comma.
- Multi-values use '|'.
- No spaces in desc.
- Values must not include ',', '|', '=' (replace with '，', '｜', '＝').
- Values keep '-' and '/' as-is.

### BBU desc fields
Order: 类别 → object-specific keys → 备注
- BBU设备: 类别=BBU设备,品牌,可见性,挡风板需求,挡风板符合性,备注
- 挡风板: 类别=挡风板,品牌,可见性,遮挡,安装方向,备注
- 螺丝/插头: 类别 is the subtype (BBU安装螺丝 / 机柜处接地螺丝 / 地排处接地螺丝 / ODF端光纤插头 / BBU端光纤插头), then 可见性,符合性,问题,备注
- 光纤: 类别=光纤,遮挡,保护措施,保护细节,弯曲半径,备注
- 电线: 类别=电线,遮挡,捆扎,备注
- 标签: 类别=标签,文本,可读性,备注

### RRU desc fields
Order: 类别 → object-specific keys → 组
- RRU设备: 类别=RRU设备,组
- 紧固件: 类别=紧固件,安装状态,组
- RRU接地端: 类别=RRU接地端,安装状态,组
- 地排接地端: 类别=地排接地端,安装状态,组
- 尾纤: 类别=尾纤,标签,套管保护,组
- 接地线: 类别=接地线,标签,组
- 标签: 类别=标签,文本,可读性,组
- 站点距离: 类别=站点距离,距离

## OCR Rules
- Prefer contentZh['请输入标签上的文字内容'] as 文本.
- If that field is empty and contentZh['能否阅读标签上的文字内容'] is neither empty nor '不能', treat it as 文本.
- If readability is '不能' or text is empty after cleaning: output 可读性=不可读 and omit 文本.
- Remove all whitespace; preserve '-' and '/'; replace ',', '|', '=' in text with safe variants.

## Conflict Resolution
- If any negative choice appears, choose the negative branch.
- Example: connect-point compliance list includes 符合要求 and 不符合要求/露铜 → output 符合性=不符合要求, 问题=露铜.
- Multiple negative issues aggregate with '|'.

## Summary JSON String
- summary is a JSON string (minified or spaced consistently).
- BBU includes per-category stats and a global 备注 list; RRU omits 备注 and adds optional 组统计.
- No ×N tokens.
- Only counts observed values; do not compute missing counts.

## Geometry Formatting
- Assistant JSON output MUST retain spaces in separators (,
