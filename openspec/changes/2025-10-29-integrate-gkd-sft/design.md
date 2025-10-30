# Design Notes

## Current State
- SFT launches call `swift.llm.train.sft.SwiftSft` via `scripts/train.sh` with YAML describing datasets, LoRA targets, etc.
- KL anchoring is absent. We rely on regular CE plus minor repetition penalties.
- ms-swift already ships a multimodal-compatible `GKDTrainer` that:
  - Loads frozen teacher + policy (student) models.
  - Balances KL(teacher || student) with CE (via `sft_alpha`) and optional student sampling (`lmbda`).
  - Surfaces telemetry (losses, KL) via trainer log history.

## Proposed Integration
1. **Configuration-first**
   - Introduce `configs/stage_2_llm_lora_gkd.yaml` / `configs/stage_3_gkd.yaml` overlays inheriting from existing stage configs.
   - New keys under `rlhf:` block (mirroring ms-swift examples) to point at `teacher_model`, `beta`, `sft_alpha`, `seq_kd`, etc.
   - Keep standard `scripts/train.sh` entry; supply `train_type=gkd` (ms-swift uses `rlhf_type: gkd`).

2. **Launcher Glue**
   - Use the existing `scripts/train.sh` (config-first). The loader builds `RLHFArguments` and `src/sft.py` selects the wrapper via `resolve_trainer_cls` when `custom.trainer_variant: gkd_monitor` is set.
   - No upstream ms-swift edits; our wrapper subclasses the ms-swift GKD trainer.

3. **Telemetry**
   - Wrapper logs `train/loss`, `train/kl_loss`, `train/sft_loss`, `train/token_accuracy` plus `eval/*` counterparts, with prefixes emitted exactly once.
   - Evaluation skips the teacher forward (CE only) to keep validation inexpensive; document keys so monitoring can alert on KL spikes and CE regressions.

4. **Docs & Recipes**
   - Update dense caption guide with: when to choose GKD, recommended `beta`/`sft_alpha`, forward-only KD recipe, and evaluation checklist.
   - Provide sample command lines (with teacher path) and expected wall-clock cost.

## Alternatives Considered
- **Custom KL-SFT Loss**: Adding a bespoke loss inside SFT trainer. Rejected for now: higher maintenance, duplicates ms-swift features.
- **Vision-only KD**: Distill projector/vision outputs. Consider as follow-up if hallucinations persist after LM anchoring.

## Risks & Mitigations
- **Teacher weight mismatch**: Ensure teacher & student share tokenizer/template; verify by running processor compatibility check before launch.
- **Performance hit**: GKD doubles forward passes. Mitigate via gradient accumulation or reduced batch size; document expected GPU memory.
- **Config drift**: Provide single source overlay; avoid manual flag toggles across multiple configs.
- **Token alignment regressions**: Earlier drafts used `torch.roll`, which wrapped BOS tokens into the loss window. Mitigate by enforcing slice-based (`[:-1]` vs `[1:]`) alignment, adding unit coverage, and failing fast on vocab mismatches.
- **DeepSpeed autocast assertions**: DeepSpeed rejects nested `torch.autocast`. Guard the teacher forward so autocast only runs when DeepSpeed is inactive (or explicitly enabled via config) and up-cast teacher logits to the student dtype.

## Post-Implementation Notes (2025-10-29)
- Implemented hard alignment guarantees: student/teacher logits now share identical masked indices, and vocab mismatches raise explicit errors instead of padding.
- Added integration tests covering first-token supervision and DeepSpeed guard behavior to detect regressions.
