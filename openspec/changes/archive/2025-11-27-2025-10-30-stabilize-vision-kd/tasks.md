- [x] Schema: add `custom.visual_kd` to config loader with validation (enabled flag, weight > 0, enum distance, allowed targets) and default disabled; surface through `TrainArguments` used by `stage_3_gkd.yaml`.
- [x] Trainer hooks: register vision/aligner feature capture for student & teacher in `GKDTrainerWithMetrics`, manage lifecycle, and expose caches.
- [x] Loss & metrics: compute feature KD term (merger + optional deepstack), integrate with total loss, add `vision_kd_loss` logging, and ensure dtype/device handling.
- [x] Tests: extend `tests/test_gkd_monitor_integration.py` (or new unit) to assert hooks fire, loss contributes, gradients reach vision params, and metrics surface.
- [x] Docs & configs: document the knob in `docs/training/REFERENCE.md` + `DATA_AND_DATASETS.md` (how feature KD interacts with pixel pipelines), provide Stage-3 overlay example, and update changelog.
- [x] Validation: plan smoke run using `configs/stage_3_gkd_visual.yaml` with `custom.sample_limit: 64`, monitor `train/vision_kd_loss` in `logging.jsonl`, and compare against the teacher-only baseline once resources are available.


