- [ ] Update OpenSpec: add a requirement to `summary-grpo-post-training` for optional CHORD mixing and config toggle.
- [ ] Add config schema + validation for a single enable/disable switch and required CHORD schedule fields.
- [ ] Wire `src/sft.py` to pass `chord_sft_dataset` (default: current GRPO train dataset) to the GRPO trainer when enabled.
- [ ] Ensure CHORD schedule params are applied to the GRPO trainer args (`chord_mu_*`, `chord_sft_per_device_train_batch_size`, `chord_enable_phi_function`).
- [ ] Add an example config under `configs/grpo/` demonstrating the toggle and safe default mu schedule.
- [ ] Update docs (`docs/training/GRPO_MS_SWIFT_PIPELINE.md`) with usage, tradeoffs, and debugging checklist.
- [ ] Run `openspec validate add-summary-grpo-chord-toggle --strict`.

