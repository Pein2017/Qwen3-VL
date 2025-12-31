# Proposal: Add CHORD (SFT-mixed) fallback for summary GRPO post-training

## Problem
Summary GRPO can produce many prompt groups with zero reward variance (`reward_std == 0`) when all rollouts are identical (often identically wrong). In that case, GRPO advantages collapse to ~0 and the update provides little or no task-learning signal for those prompts. Dynamic sampling/resampling mitigates some cases but does not guarantee non-zero-variance groups.

## Goal
Provide an easy-to-toggle mechanism to inject supervised learning signal during summary GRPO post-training so that training can still make progress even when rollouts are identical.

## Proposed Change
- Add a **config-driven** toggle to enable CHORD-style loss mixing for `rlhf_type=grpo` summary post-training.
- When enabled, compute a combined loss:
  - `loss = (1 - mu) * grpo_loss + mu * chord_sft_loss`
  - where `chord_sft_loss` is a teacher-forced token-level cross-entropy loss over an expert dataset.
- Default expert dataset source: reuse the in-memory GRPO training dataset (fusion summary dataset) so the expert targets match the two-line summary contract and irrelevant handling.

## Toggle UX
- A single YAML switch SHALL enable/disable the feature.
- When disabled, training behavior remains unchanged.

### Proposed config surface (YAML)
```yaml
custom:
  grpo_chord:
    enabled: true            # default false
    per_device_train_batch_size: 1
    mu_warmup_steps: 100
    mu_decay_steps: 1000
    mu_peak: 0.10
    mu_valley: 0.00
    enable_phi_function: false
```

## Impact / risks
- Adds compute cost (extra forward pass per step when `mu > 0`).
- Introduces a supervised bias; `mu` schedule needs conservative defaults to avoid overpowering GRPO.

## Out of scope
- Changing reward functions, rollout generation, or advantage estimation.
- Conditional gating (e.g., only apply SFT loss when `reward_std == 0`) unless explicitly requested in a follow-up.
