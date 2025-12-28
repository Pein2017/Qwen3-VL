> Note (2025-12): Stage‑B now runs rule-search only. Reflection/selection references below are legacy and kept for historical context.

# Design Optimal Distributed Training Architecture for Stage-B

## Problem Statement

I need to accelerate Stage-B training by leveraging 8 GPUs with data-parallelism, while maintaining correctness and handling the sequential reflection constraint.

**Current State:**
- **Entry point**: `scripts/stage_b.sh` → `src/stage_b/runner.py`
- **Single-GPU flow**: `rollout (batched) → selection → reflection (sequential, updates guidance) → next epoch`
- **Key components**:
  - `RolloutSampler.generate_for_batch()` in `src/stage_b/rollout.py` generates candidates for ticket batches
  - `ReflectionEngine` runs sequentially and updates `GuidanceRepository` (shared state)
  - Guidance updates affect subsequent rollouts (shared state dependency)
- **Model**: Full model fits on single GPU; 8 GPUs available with free memory
- **Config**: `configs/stage_b/bbu_*.yaml` (e.g., `bbu_line.yaml`); `runner.rollout_batch_size` controls batch size

**Constraints:**
1. Reflection MUST be sequential (updates shared `guidance.json` state)
2. All ranks must use identical guidance for deterministic results
3. Results must be deterministic (same seed → same outputs as single-GPU)
4. Only rank 0 should write artifacts to disk
5. Must preserve backward compatibility (single-GPU mode should still work)

## Task: Design Optimal Architecture

**Please analyze the codebase and design the optimal distributed training architecture for Stage-B.**

### Analysis Required

1. **Architecture Exploration**:
   - Analyze the current rollout/reflection/guidance update flow in `src/stage_b/runner.py`
   - Understand how batches are processed and how guidance state propagates
   - Identify all synchronization points and shared state dependencies
   - Review existing distributed patterns in the codebase (e.g., `src/sft.py`, `src/utils/logger.py`)

2. **Design Space Analysis**:
   - **Rollout distribution strategies**: How should batches be split across GPUs?
     - Per-batch splitting (split each `rollout_batch_size` batch across GPUs)
     - Batch-level parallelism (each GPU processes full batches from different ranges)
     - Other strategies?
   - **Reflection coordination**: How to collect eligible bundles and coordinate reflection?
     - Gather all bundles on rank 0 → reflect → broadcast guidance
     - Alternative coordination patterns?
   - **Model loading**: Load per rank vs. load once and broadcast?
   - **Guidance synchronization**: When and how to synchronize guidance state?
     - After each reflection cycle?
     - At epoch boundaries?
     - Other synchronization strategies?
   - **Artifact aggregation**: How to collect trajectories/selections from all ranks?
     - Gather on rank 0 and write?
     - Write per-rank and merge?
     - Other approaches?

3. **Tradeoff Analysis**:
   - **Performance**: Expected speedup, communication overhead, load balancing
   - **Complexity**: Implementation complexity, maintenance burden
   - **Correctness**: Determinism guarantees, state consistency
   - **Scalability**: How does design scale to different GPU counts?

4. **Implementation Strategy**:
   - Which distributed backend? (PyTorch DDP, torchrun, accelerate, custom?)
   - How to integrate with existing config system (`src/stage_b/config.py`)?
   - How to maintain single-GPU compatibility?
   - What changes are needed in `scripts/stage_b.sh` for distributed launch?

### Deliverables

1. **Architecture Design Document**:
   - Recommended architecture with rationale
   - Tradeoff analysis of considered alternatives
   - Sequence diagrams or flow charts showing the distributed execution flow
   - Synchronization points and state management strategy

2. **Implementation Plan**:
   - Files to modify with high-level change descriptions
   - Config schema additions needed
   - Testing strategy (correctness, performance, edge cases)

3. **Code Implementation** (after design approval):
   - Modified code following the designed architecture
   - Updated config schema
   - Updated launch script
   - Documentation updates

### Success Criteria

- **Correctness**: Multi-GPU run produces identical results to single-GPU run (same seed, after aggregation)
- **Performance**: Significant speedup for rollout phase (target: 5-8x for large batches, accounting for overhead)
- **Maintainability**: Clean integration with existing codebase patterns
- **Backward compatibility**: Single-GPU mode continues to work without changes

### Context Files to Review

- `src/stage_b/runner.py` - Main training loop
- `src/stage_b/rollout.py` - Rollout sampling logic
- `src/stage_b/reflection.py` - Reflection engine (sequential constraint)
- `src/stage_b/config.py` - Configuration schema
- `src/utils/logger.py` - Existing distributed utilities
- `scripts/train.sh` - Reference for distributed setup patterns
- `docs/runtime/STAGE_B_RUNTIME.md` - Runtime documentation

**Please start by analyzing the codebase, then propose the optimal architecture design with clear rationale.**

