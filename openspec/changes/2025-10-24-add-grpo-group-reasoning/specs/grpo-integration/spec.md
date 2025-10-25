## ADDED Requirements

### Requirement: Trainer & Programmatic Launcher
- Use ms-swift GRPO trainer via a Python launcher (no CLI); expose a function `run_grpo(config)` that constructs trainer, datasets, and rewards.
- Set `num_generations >= 2` and satisfy batch divisibility.
- Load rewards via Python modules (external_plugins optional); use only label and format rewards initially.

#### Scenario: Weight length mismatch
- If 2 reward funcs and 3 weights → raise ValueError at startup

### Requirement: LLM-only LoRA on last-K blocks
- Freeze ViT and Aligner; apply LoRA only to the last K transformer blocks of the LLM (K is configurable; default K=4).
- Verify wrapped model and `modules_to_save` reflect no-vision tuning.

#### Scenario: Sanity print
- Logs show `freeze_vit=true`, `freeze_aligner=true`, `freeze_llm=false`, `lora_last_k=4`, and Peft/SwiftModel wrapping

### Requirement: Dataset passthrough
- Inputs include `messages`, `stage_a_summaries`, `group_label`, `task_type` for reward functions.
- Rewards receive these via kwargs.

### Requirement: Minimal runnable example (Python)
- Provide a Python example that loads base model and optional adapters, dataset path, registers rewards, and runs a short training dry-run.

#### Scenario: Dry run
- Run 1–2 steps with tiny dataset; trainer produces reward logs and writes completions.jsonl
