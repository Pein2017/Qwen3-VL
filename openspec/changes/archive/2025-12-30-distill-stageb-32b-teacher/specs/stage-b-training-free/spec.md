# distill-stageb-32b-teacher → stage-b-training-free deltas

## ADDED Requirements

### Requirement: Stage-B runner SHALL support a Teacher distillation mode (on by default) that logs chatml conversations without changing training-free semantics.
Stage-B runner SHALL expose a configuration flag for a Teacher distillation mode (default **enabled** in canonical configs) that can switch to an alternative checkpoint（例如 Qwen3‑32B 文本模型）做 rollout，同时仍遵守 training-free 约束（不在该流程中更新任何模型权重），并在收敛后的最后一轮 rollout 额外写出 chatml 对话日志供后续 SFT 使用。

#### Scenario: Distillation mode uses a Teacher checkpoint, runs until guidance stops changing, and preserves training-free guarantees
- GIVEN a Stage-B config with `stage_b_distillation.enabled: true` (default) and `model.model_name_or_path` pointing to a Teacher checkpoint（例如 `model_cache/models/Qwen/Qwen3-32`）
- WHEN the Stage-B runner continues epochs until the mission guidance no longer receives any updates (reflection applies zero operations in an epoch)
- THEN it SHALL treat that guidance-stable epoch as the **final distill epoch**, use the Teacher checkpoint only for forward passes（rollout/selection），without performing any optimizer steps or checkpoint updates
- AND it SHALL continue to emit standard training-free artifacts（`trajectories.jsonl`, `selections.jsonl`, `guidance.json`, `reflection.jsonl`）under `{output.root}/{run_name}/{mission}/`
- AND it SHALL additionally emit a distillation log file（例如 `distill_chatml.jsonl`）containing the selected verdict per group from that final epoch.

### Requirement: Distillation logs SHALL capture complete chatml-style conversations aligned with existing chat datasets.
When Stage-B distillation mode is enabled, the runner SHALL record conversations in a shape that can be consumed directly as `dataset: chat` with `template: chatml` (consistent with coig_cqia) so subsequent SFT only needs to point fusion configs at the generated JSONL files.

#### Scenario: Logging the selected Teacher verdict from the convergence epoch as a single-turn chatml conversation
- GIVEN Stage‑A summaries for a group (`per_image` dict), mission name, the **converged** guidance text (the epoch where no guidance updates occurred), and the Teacher-selected `{verdict, reason}` pair for that epoch
- WHEN distillation logging is enabled
- THEN the runner SHALL construct a `messages` array containing exactly:
  - a `system` message that encodes Stage‑B instructions（Verdict/Reason 两行协议、mission focus、guidance 片段）;
  - a `user` message that summarizes the ticket context（mission、历史 label、按图片编号串联的 Stage‑A 摘要文本，以及必要的 task description）;
  - a single `assistant` message whose content corresponds to the **selected** Teacher Verdict/Reason output for that ticket（保持两行协议形式）。
- AND the runner SHALL write a JSONL record for each logged verdict containing `{group_id, mission, label, messages}` only (no extra decode/epoch metadata), where `messages` is compatible with the existing `chatml` template（roles限定在 `system|user|assistant`，content 为字符串），无需后处理即可用于 `dataset: chat`。

### Requirement: Distillation logging SHALL not alter the semantics or schema of existing Stage-B artifacts.
Enabling distillation mode SHALL be additive-only: standard training-free Stage-B artifacts must remain backward compatible so that downstream evaluation and reflection workflows continue to function unchanged.

#### Scenario: Distillation mode coexists with standard trajectories and selections
- GIVEN a Stage‑B run with `stage_b_distillation.enabled: true`
- WHEN the run completes
- THEN the system SHALL:
  - write `trajectories.jsonl`, `selections.jsonl`, `guidance.json`, `reflection.jsonl`, and `manual_review_queue.jsonl` under the mission directory with the same schema and semantics as before;
  - write a fresh `distill_chatml.jsonl` for the convergence epoch, **overwriting any existing file** in that mission directory on rerun;
  - avoid introducing new mandatory fields into existing artifacts that would break current readers; any distillation-specific metadata SHALL live in the new log file(s) or optional fields that existing tooling can ignore safely.
- AND if distillation logging fails（e.g., due to disk full or schema bug），the runner MAY log a warning and continue producing standard Stage‑B artifacts, but MUST NOT corrupt or partially overwrite existing guidance/selections.
