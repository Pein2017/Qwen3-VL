## 1. Implementation
- [ ] 1.1 Create `src/datasets/augmentation/` with `base.py` and `registry.py`
- [ ] 1.2 Implement built-in ops: hflip, vflip, rotate, scale, color_jitter
- [ ] 1.3 Add config schema and validation for augmentation settings
- [ ] 1.4 Integrate registry into `AugmentationPreprocessor` with RNG injection
- [ ] 1.5 Add unit tests for transform correctness and geometry updates
- [ ] 1.6 Add docs in `docs/qwen3vl.md` and config examples under `configs/`
- [ ] 1.7 Wire feature flag into `src/sft.py` loading path
- [ ] 1.8 Run smoke training with augmentation enabled and verify logs/vis outputs

## 2. Validation
- [ ] 2.1 `openspec validate add-image-augmentation-plugin --strict`
- [ ] 2.2 Lint/type-check changed modules
- [ ] 2.3 Confirm determinism with fixed seed
