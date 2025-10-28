# Tasks: Dense Augmentation Telemetry & Safety

- [x] Align canvas before scaling; enforce `max_pixels` with floor-to-multiple logic; expose padding telemetry.
- [x] Replace AABB coverage with polygon clipping for quads; wire into `RandomCrop` and completeness updates.
- [x] Emit crop telemetry (skip counters, coverage, padding ratios) via `Compose` and `apply_augmentations`.
- [x] Document safeguards/telemetry in `docs/AUGMENTATION.md`.
- [x] Add regression tests for polygon coverage and pixel-cap enforcement.
- [ ] Follow-up: inspect telemetry during next dense-caption smoke run.

