# Design Notes — Two-Line Stage-B Protocol

## Goals
- Eliminate brittle四行/Evidence JSON协议，改为最简 `Verdict + Reason`，降低解析失败概率并适配无证据类任务。
- Remove legacy fields to avoid drift/partial updates.

## Target Protocol
- Response (assistant → system):
  - Line 1: `Verdict: 通过` or `Verdict: 不通过` (case-insensitive alias allowed)
  - Line 2: `Reason: <简体中文单行，<=120字，列出关键判定依据，可用分号分隔>`
- No Evidence_Positive / Evidence_Negative / confidence fields. No extra lines/markdown.

## Pipe Changes
- **Prompts**: System prompt enforces两行输出；user prompt unchanged except removal of evidence examples; keep Chinese, brevity, no markdown.
- **Parsing**: Rollout parser reads first verdict-looking line + first reason-looking line; treats remaining text as noise and discards. No JSON decoding needed.
- **Types**: `ParsedTrajectory` drops evidence fields; downstream uses `verdict`, `reason`, `format_ok`.
- **Selection**: Majority vote on verdict; reasons carried for logging/export; label_match computed from verdict vs GT.
- **Reflection**: Build bundles from verdict+reason only; summary/critique text to reference reason instead of evidence; ops generation unchanged semantically.
- **Export**: `trajectories.jsonl` / `selections.jsonl` schemas drop evidence_* and confidence; docs updated accordingly.

## Compatibility
- No backward compatibility; old four-line outputs considered invalid.

## Validation
- One debug run (`configs/stage_b/debug.yaml`) must complete with zero parse warnings.
- Spot-check artifacts: selections.jsonl, reflection.jsonl, guidance.json unchanged format-wise, but without evidence keys.
