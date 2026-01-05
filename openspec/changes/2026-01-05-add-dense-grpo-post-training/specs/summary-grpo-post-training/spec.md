# summary-grpo-post-training Specification (Change Proposal)

## ADDED Requirements

### Requirement: Summary reward ORMs are compatible with ms-swift GRPO calling convention
Summary GRPO reward implementations SHALL accept the ms-swift GRPO calling convention `reward_func(completions, **kwargs)` and SHALL treat `completions` as the primary completion list input.

#### Scenario: Summary reward accepts positional completions
- **WHEN** ms-swift invokes a `summary.*` reward during GRPO training
- **THEN** the reward receives `completions` as a positional list of strings
- **AND** the reward computes per-sample rewards aligned to that list length

### Requirement: Summary rewards no-op on dense samples in mixed-mode GRPO
When summary rewards are included in a mixed-mode GRPO run that also includes dense samples, `summary.*` rewards SHALL return a neutral reward value of `0.0` (and MUST NOT raise) for samples where `metadata._fusion_mode == "dense"`.

#### Scenario: Summary reward skips dense sample
- **GIVEN** a dense-mode sample with `metadata._fusion_mode == "dense"`
- **WHEN** a `summary.*` reward function is invoked
- **THEN** it returns `0.0`
- **AND** it does not attempt summary header mapping or summary JSON parsing
