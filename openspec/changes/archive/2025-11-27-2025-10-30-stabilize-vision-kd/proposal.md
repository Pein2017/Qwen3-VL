# Proposal: Stabilize Vision KD for Qwen3-VL

## Problem Statement
- Stage-3 GKD runs presently compute token-level KL across the full language sequence. Vision/aligner weights (visual merger + deepstack mergers) are still free to drift when the student leaves the teacher distribution.
- On out-of-domain imagery, the student begins hallucinating BBU-scoped content even when the scene differs; coordinates stay anchored to legacy formatting because the KL term clamps every token back to the teacher.
- We need an additional regularizer that anchors the multimodal pathway (image patch embeddings + aligner outputs) to the frozen teacher, while allowing the language stack (LoRA heads) to explore new coordinate formatting.

## Desired Outcomes
- Optional feature-level KD that operates on visual tokens before they enter the language model.
- Config-driven weights so the vision KD strength can be tuned per stage without code edits.
- Telemetry (per-mode `vision_kd_loss`) to confirm the regularizer is active and finite.
- Backwards compatibility: when the knob is disabled the training loop behaves exactly as today.

## Scope
- ms-swift powered SFT/GKD training in this repository.
- Qwen3-VL-4B vision encoder + aligner stack (final merger + deepstack mergers).
- YAML config schema, trainer wrapper, tests, and docs.

## Non-Goals
- No architectural changes to ms-swift core or upstream transformers.
- No changes to dataset format or augmentation pipeline.
- Avoid altering teacher CE/KL computation beyond adding the vision KD term.

## Proposed Approach (high level)
1. **Config knob**: introduce `custom.visual_kd` with fields `enabled`, `weight`, `targets` (`merger`, `deepstack`), and `distance` (`mse`, possibly `cosine`). Validate in config loader.
2. **Feature capture**: register forward hooks on `model.model.visual` (final merger output) and its `deepstack_merger_list` modules for both student and teacher so we can reuse activations from the normal forward pass (no duplicate vision compute).
3. **Loss computation**: when enabled, compute the chosen distance between teacher and student features for each requested target, normalize by token count, multiply by `weight`, and add to `total_loss`. Keep gradients on student only; teacher stays `no_grad`.
4. **Telemetry**: append `train/vision_kd_loss`, `eval/vision_kd_loss` to the metrics aggregator. Ensure NaNs surface as warnings (consistent with existing KL logging).
5. **Docs & Recipes**: add a Stage-3 overlay example showing how to activate the feature and document trade-offs in `docs/REFERENCE.md`.

## Validation Plan
- Unit tests that run a tiny forward pass with synthetic multimodal batch to assert hooks fire, the loss is computed, and gradients flow to aligner parameters only.
- Update integration test in `tests/test_gkd_monitor_integration.py` to confirm metrics include `vision_kd_loss` when enabled.
- Smoke training run (≤2k samples) comparing Stage-3 with and without the feature to ensure loss is finite and language outputs show improved coordinate adaptation (documented in proposal follow-up).


