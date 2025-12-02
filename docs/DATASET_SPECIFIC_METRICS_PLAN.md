# Dataset-Specific Metrics Implementation Plan

## Problem Statement

Training logs currently show only global metrics (`loss`, `token_acc`, `grad_norm`, `learning_rate`) but missing dataset-specific metrics like `bbu_loss`, `lvis_token_acc` that would help monitor per-dataset performance during fusion training.

The existing `GroupedMetricsMixin` + `GroupedPackingDataset` solution is experimental and should be deprecated, so we need alternative approaches.

## Current Architecture

### What Already Exists

1. **Metadata Annotation** (`src/datasets/unified_fusion_dataset.py:223-236`)
   - Fusion dataset already annotates each record with:
     - `metadata._fusion_source`: dataset name (bbu, rru, lvis, lang_chat)
     - `metadata._fusion_template`: template name
     - `metadata._fusion_domain`: "target" or "source"

2. **Custom Metrics Infrastructure** (`swift/trainers/mixin.py:96-100, 854-859`)
   - `SwiftMixin` provides `custom_metrics` dict with train/eval modes
   - Uses `MeanMetric` for aggregation across steps
   - Automatically syncs keys across distributed ranks
   - Logged via `compute_custom_metrics()` → `log()`

3. **Channel Loss Pattern** (`swift/megatron/trainers/trainer.py:66-75`)
   - Megatron trainer supports per-channel metrics via `channels` field in batch
   - Uses `packed_seq_params.cu_seqlens` to slice token-level losses
   - Updates `custom_metrics[mode][f'loss_{channel}']`

## Solution Options

### Option 1: Channel-Based Tracking (Recommended - Minimal Changes)

**Approach**: Reuse the existing `channel` field pattern that Megatron trainer already supports.

**Implementation**:
1. Add `dataset_source` field to collated batches (alongside `input_ids`, `labels`)
2. Modify `SwiftMixin.compute_loss` to extract per-sample metrics grouped by `dataset_source`
3. Update custom_metrics: `self.custom_metrics[mode][f'{source}_loss'].update(...)`

**Pros**:
- Minimal code changes (~50 lines)
- Reuses proven infrastructure
- Works with standard padding (no packing required)
- No breaking changes to dataset/collator contracts

**Cons**:
- Requires per-sample tracking (works best with `per_device_batch_size=1` or small batches)
- Loses granularity when batches mix datasets (reports batch-level average)

**Files to modify**:
- `src/sft.py`: Wrap collator to extract & attach `dataset_source` from metadata
- `swift/trainers/trainers.py`: Override `compute_loss` to log per-source metrics

---

### Option 2: Callback-Based Sampling Metrics

**Approach**: Use a training callback to periodically sample and log per-dataset metrics on separate eval batches.

**Implementation**:
1. Create `DatasetMetricsCallback(TrainerCallback)`
2. On `on_step_end` (every N steps):
   - Sample a small batch from each dataset directly (bypass main dataloader)
   - Run forward pass (no backward)
   - Compute & log per-dataset loss/accuracy
3. Log metrics with `dataset_{name}_loss` / `dataset_{name}_token_acc`

**Pros**:
- Zero impact on training loop performance
- Works with any batch size; packing removed (padding-only)
- Clean separation of concerns

**Cons**:
- Metrics are sampled, not exhaustive (may not reflect true training distribution)
- Requires maintaining separate mini-dataloaders
- Additional forward passes add ~5% overhead

**Files to create**:
- `src/trainers/callbacks/dataset_metrics.py`

---

### Option 3: Post-Training Log Analysis

**Approach**: Don't change training; analyze logs offline to infer per-dataset metrics.

**Implementation**:
1. Enable debug logging to emit `dataset` field per sample (already exists in `unified_fusion_dataset.py:555-576`)
2. Create `scripts/analyze_dataset_metrics.py`:
   - Parse `logging.jsonl` + debug logs
   - Match samples to their source datasets
   - Compute windowed per-dataset metrics
3. Output per-dataset curves for TensorBoard/plotting

**Pros**:
- Zero training code changes
- Can retrospectively analyze existing runs
- No performance overhead

**Cons**:
- Metrics not visible during training (can't react to issues in real-time)
- Requires post-hoc correlation (sample_id → dataset → loss)
- Debug logs are verbose and may bloat storage

---

### Option 4: Custom Data Collator with Group Tracking

**Approach**: Create a lightweight collator wrapper that tracks dataset groups without full packing.

**Implementation**:
1. Wrap template collator to preserve `metadata._fusion_source` → `dataset_group` list
2. In `compute_loss`, iterate over batch samples:
   ```python
   for i, group in enumerate(dataset_groups):
       sample_loss = loss_unreduced[i].mean()
       self.custom_metrics[mode][f'{group}_loss'].update(sample_loss)
   ```
3. Compute per-sample token accuracy similarly

**Pros**:
- Exact per-sample metrics (no sampling)
- Works with standard padding
- Reuses existing `custom_metrics` infrastructure

**Cons**:
- Requires `reduction='none'` loss (more memory if batch is large)
- Per-sample iteration may add ~2-3% overhead
- Slightly more invasive than Option 1

**Files to modify**:
- `src/datasets/collators.py`: Add `DatasetTrackingCollator`
- `src/sft.py`: Wrap collator conditionally
- `swift/trainers/trainers.py`: Extract groups, compute per-sample losses

---

## Recommendation

**Start with Option 1 (Channel-Based Tracking)** because:

1. **Proven Pattern**: Megatron trainer already uses this for channel loss
2. **Minimal Risk**: ~50 lines, no API changes, easily reversible
3. **Immediate Value**: Metrics visible in real-time during training
4. **Incremental**: Can layer Option 2 (sampling) later for validation

If Option 1 proves insufficient due to batch mixing, escalate to **Option 4** (per-sample tracking with custom collator).

Reserve **Option 2** for auxiliary monitoring and **Option 3** for retrospective analysis of existing runs.

---

## Next Steps

1. **Prototype Option 1**:
   - Add collator wrapper in `src/sft.py` (lines ~873-877)
   - Override `compute_loss` in custom trainer mixin
   - Test with `configs/fused_data/debug.yaml`

2. **Validation**:
   - Check logs show `bbu_loss`, `rru_loss`, `lvis_loss`, `lang_chat_token_acc`
   - Verify metrics sync correctly in multi-GPU (DeepSpeed) setup
   - Compare global loss = weighted average of per-dataset losses

3. **Documentation**:
   - Update `docs/TRAINING_PLAYBOOK.md` with new metrics
   - Add example to `docs/REFERENCE.md`
