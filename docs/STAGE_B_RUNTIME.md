# Stage-B Runtime Guide

#### Stage-B Runner (`src/stage_b/runner.py`, `scripts/stage_b_run.sh`)

Purpose: Training-free verdict loop that performs ingest → rollout → selection → reflection with mission-specific guidance updates.

```bash
config=configs/stage_b/run.yaml gpus=0 log_level=logging \
  bash scripts/stage_b_run.sh
```

- `GuidanceRepository` copies the global guidance file into `{output.root}/{output.run_name}/{mission}/guidance.json` so edits stay isolated until you manually promote them back.
- The shared Qwen3-VL model (defined under `model.*`) is reused by the sampler, critic, and reflection engine to minimize VRAM footprint.

##### Config Breakdown (`src/stage_b/config.py`)

- `model`: HF checkpoint path, dtype (`bfloat16` recommended), and device map (`auto` for single GPU, or explicit slices).
- `sampler`: Array of decode configs (temperature/top_p/max tokens/stop tokens) plus `samples_per_decode` and optional format filter. `RolloutSampler` iterates this grid deterministically per batch.
- `signals`: Enables deterministic metrics — `store_confidence` propagates candidate confidence; `enable_consistency` computes consensus ratios for manual review gating.
- `critic`: Controls `CriticEngine` (prompt template, decode knobs, per-candidate caps, char budgets). Disable this block to skip LLM critiques.
- `selection`: Policy + tie-breaker for `select_for_group` (e.g., `label_match_then_confidence`).
- `manual_review`: Thresholds for deferring to humans when the majority verdict is low-confidence despite agreement or when label match flips happen.
- `reflection`: Prompt path, batch size, eligibility policy, and change budgets. `rapid_mode` removes guardrails for quick dry runs.
- `runner`: Epoch count (per mission). Combined with `seed` and `_shuffle_indices`, this determines ticket order per epoch.
- `output`: Root/run_name plus legacy paths for consumers expecting `trajectories.jsonl` / `selections.jsonl`.
- `guidance`: Global guidance file and snapshot retention count.

##### Outputs & Directory Layout

Each mission writes to `{output.root}/{output.run_name}/{mission}/`:
- `guidance.json` + `snapshots/` — Mission-specific guidance plus retained history.
- `trajectories.jsonl` — One entry per candidate with signals, critic output, reflection metadata, epoch number.
- `selections.jsonl` — Final verdict per ticket, referencing the selected candidate index, reflection cycle, manual-review flag, and eligibility notes.
- `reflection.jsonl` — Chronological log of reflection bundles, proposals, applied status, and the evidence groups they cite.

> Tip: raise `sampler.samples_per_decode` when you need multiple attempts per group under the same decode hyperparameters; drop it to `1` for deterministic sweeps.
> Configure multi-epoch sweeps via `runner.epochs`. The runner shuffles tickets using `(seed + epoch)` so repeated runs with the same seed stay reproducible.

##### Critic & Manual Review

- `CriticEngine` consumes parsed candidates + deterministic signals and emits JSON critiques with `summary`, `critique`, `issues`, and optional `uncertainty_note`. Configure prompts under `configs/stage_b/prompts/critic*.md`.
- Manual review gating uses `manual_review.*` thresholds plus critic signals; when triggered, the selection record sets `manual_review_recommended=true` and the candidate is withheld from reflection batches.
- Deterministic signals ensure consistent telemetry even when the critic is disabled; use them to monitor verdict consensus and label alignment.

### Stage-B GRPO Experiments

`scripts/run_grpo.py` is an experimental launcher for LoRA-based GRPO on Stage-B style datasets. It:
- Loads Stage-A JSONL via `src.stage_b.dataset.load_stage_a_for_grpo`.
- Targets only the last-K transformer blocks (`lora_last_k_blocks`) while freezing vision + aligner stacks.
- Uses the reward functions in `src/stage_b/rewards.py` (`label_reward`, `format_reward`) with configurable weights.
- Shares the Qwen3-VL processor/model path with Stage-B inference to stay prompt-compatible.

Treat it as scaffolding: you still need ms-swift GRPO support plus curated text-only datasets, but the script documents the expected knobs and reward composition.
