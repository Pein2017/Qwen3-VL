# Design: Vision/Aligner Feature Distillation

## Background & Constraints
- **Stage context**
  - Stage-3 (`configs/stage_3_gkd.yaml`) currently runs LoRA on ViT last-6 + LLM last-2 under GKD, launched via `scripts/train.sh`. Any new knobs must plug into the YAML-only workflow so existing launch commands remain unchanged.
  - Dataset pipeline (described in `docs/DATA_AND_DATASETS.md`) always supplies `pixel_values`, `image_grid_thw`, and packed conversation tensors; hooks must respect packing and augmentation side-effects.
- **Visual stack layout** (Qwen3-VL-4B)
  - `model.model.visual` returns two tensors during a forward pass: the merged image tokens (`Qwen3VLVisionPatchMerger`) and a list of deepstack features from the intermediate `deepstack_merger_list` ([ref](https://github.com/??) but specifically `transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel.forward` lines 714-753 in the installed package).
  - During the causal LM forward, these merged tokens populate `<image>` placeholders via `masked_scatter`, and the deepstack features feed into the first few language layers through `_deepstack_process`.
- **Trainer wrapper** (`src/trainers/gkd_monitor.py`)
  - Already overrides `compute_loss` to run the teacher forward under `torch.no_grad`, align logits, and accumulate metrics.
  - Student inputs (`model_inputs`) contain `pixel_values` and `image_grid_thw`; the forward executes the full vision stack exactly once per batch.
- **Hook requirements**
  - Avoid recomputing vision features (costly) or disrupting DeepSpeed wrapping.
  - Teacher features must stay detached; student features require gradients.
  - Handles packed batches (`logits_to_keep` path) without relying on sequence order from the language model—vision embeddings sit before packing, so we anchor them prior to packing.

## Proposal Overview
We add optional feature-level KD controlled via `custom.visual_kd` in YAML. When enabled, the trainer collects vision/aligner activations from both student and teacher, computes an alignment loss, and adds it to the total objective. The change is internal to `src/trainers/gkd_monitor.py`; `scripts/train.sh` and higher-level entry points stay untouched apart from consuming the new config knob.

### Feature Capture Strategy
1. **Student hooks**: register `forward_hook`s at trainer init on `student.model.visual` and its `deepstack_merger_list` modules (accessed via `accelerator.unwrap_model`). Hooks store references in `self._student_visual_cache` keyed by module name. Because hooks run inside the existing forward, they avoid re-computation and receive tensors with gradient attached.
2. **Teacher hooks**: register identical hooks but store tensors detached (`.detach()`) in `self._teacher_visual_cache` because the teacher stays frozen.
3. **Lifecycle**: hooks are installed once at trainer construction and removed in `__del__` (or via context manager) to avoid reference leaks when the trainer is destroyed. Guard against multiple registrations by checking sentinel flags (matters when restarting from checkpoints).
4. **Distributed/ZeRO safety**: use `accelerator.unwrap_model` to locate the base module before attaching hooks so we work regardless of DeepSpeed wrapping.

### Loss Computation
- **Targets**: configurable list; supported values at MVP are `merger` (final aligner tokens) and `deepstack` (full list aggregated token-wise). Additional slots can be added later. Respect the shape emitted by Qwen3-VL: merger output shape `(visual_seq, hidden)`, deepstack list of tensors with identical shapes per level.
- **Distance**: default `mse`; we expose `cosine` as an alternative but keep the implementation simple. Each distance operates on matching tensors:
  - `merger`: `student_cache['visual.merger']` vs `teacher_cache['visual.merger']` (shape `(visual_tokens, hidden_dim)`).
  - `deepstack`: iterate over `deepstack_merger_list.{i}` entries; compute loss per level and average.
- **Normalization**: divide by token count to avoid scale growth with varying grids; final per-target loss averaged then multiplied by `weight`.
- **Aggregation**: add to `total_loss` right after KL/CE composition. Also push to metrics under `vision_kd_loss` so logs show contribution magnitude. When batches lack images (e.g., text-only eval), skip the loss gracefully.

### Config Schema
```yaml
custom:
  visual_kd:
    enabled: true
    weight: 0.5           # float > 0
    targets: [merger, deepstack]
    distance: mse         # enum {mse, cosine}
```
- Schema lives in `src/config/custom_schema.py` (or equivalent) with validation to ensure the feature can’t run without teacher features. Default is disabled; `stage_3_gkd.yaml` can opt in via overlay (e.g., `stage_3_gkd_visual.yaml`). Guard cases: missing images, no `<image>` tokens, or disabled targets must short-circuit the loss path.

### Logging & Telemetry
- Extend `_metrics` accumulator with `vision_kd_loss` (train/eval). During eval we still compute the loss because hooks fire; KL stays disabled per spec but feature KD runs under `no_grad` teacher forward (already executed for CE). Skip metric logging when the cached tensors are empty.
- Warn if NaN/Inf appears in `vision_kd_loss` similar to KL handling.

### Compatibility
- **DeepSpeed**: hooking the unwrapped model ensures ZeRO stage 2/3 works. The additional tensor storage is per batch and modest; clear caches every iteration to prevent stale references.
- **LoRA**: gradient flows only through student visual parameters; language LoRAs remain unaffected.
- **Inference**: feature is train-only; evaluation inside trainer uses same forward but no teacher step during pure eval (teacher forward already skipped). We'll compute the KD loss using cached teacher features only when they exist; otherwise metric is zero and skipped.
- **Mixed precision**: ensure teacher caches are cast to student dtype before distance computation to avoid dtype mismatch.

## Alternatives Considered
- **Re-run visual encoder manually** per `compute_loss`: simpler but doubles vision compute (unacceptable for 20-epoch runs). Rejected.
- **Hook inside transformers model** via monkey patching: higher maintenance risk; we can do it from the trainer with local hooks instead.
- **Distill language-hidden states**: explicitly out of scope because we want the language tower to drift.

## Open Questions / Follow-ups
- Should we expose per-target weights (e.g., heavier on merger than deepstack)? MVP uses shared weight but schema can evolve.
- Evaluate whether cosine distance is needed; spec allows it but we can drop during implementation if unnecessary.
- If future work needs patch-token weighting (e.g., favor foreground regions) we can extend the mask in dataset pipelines.


