# Tasks

- [ ] Inventory current config usage and identify duplication hotspots (fusion_train, dataset_mix, grpo, debug, smoke, distill).
- [ ] Define the new config hierarchy (components, train presets, fusion configs) and document the composition rules.
- [ ] Add OpenSpec deltas for `sft-training` and `multi-dataset-fusion`.
- [ ] Implement fusion-config inheritance with name-based merge for `targets` and `sources`.
- [ ] Create component YAMLs and migrate SFT presets (dense/summary, 1024/2048, debug/smoke).
- [ ] Create GRPO presets and migrate Stage-B distill config to the new hierarchy.
- [ ] Migrate fusion configs to new base + overlay structure and update all references.
- [ ] Update `scripts/validate_sft_config.py` to validate the new layout (including fusion config resolution).
- [ ] Update docs (`docs/training/TRAINING_PLAYBOOK.md`, `docs/training/REFERENCE.md`, `scripts/README.md`, `docs/README.md`).
- [ ] Validate all configs with `scripts/validate_sft_config.py` and spot-check core presets.
