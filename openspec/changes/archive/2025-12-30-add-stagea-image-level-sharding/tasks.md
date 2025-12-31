# Tasks — Add Stage-A Image-Level Distributed Sharding

- [x] Add CLI flag `--sharding_mode {per_group,per_image}` (default: `per_group`) and validate arguments.
- [x] Remove/disable `--batching_mode cross_group` from Stage-A public surface (keep `per_group` semantics only).
- [x] Add an option to retain intermediate per-image outputs (default: delete after successful merge).
- [x] Implement `per_group` execution path (retain current within-group batching behavior).
- [x] Implement `per_image` execution path:
  - [x] Flatten sampled groups into deterministic image jobs with `(group_seq, image_index, path)`.
  - [x] Shard image jobs across ranks deterministically (default: `jobs[rank::world_size]`).
  - [x] Batch per rank with at most `batch_size` images in flight.
  - [x] Write per-rank per-image JSONL intermediate outputs.
  - [x] Rank 0 merge: rebuild group-level records in deterministic group order; enforce strict coverage; mark incomplete groups as failed and continue.
  - [x] Cleanup intermediate files after successful merge.
- [x] Update `scripts/stage_a.sh` to use `--sharding_mode` and remove `batching_mode` wiring.
- [x] Add targeted tests or a lightweight validation script (if an existing test harness exists for Stage-A) to validate merge coverage and deterministic ordering.
- [x] Update runtime docs if Stage-A/B runbooks mention batching modes (scope: `docs/runtime/STAGE_A_STAGE_B.md` if applicable).
- [x] Emphasize “batched generation requires left padding” (top urgency) in Stage-A docs/specs to prevent future regressions.
