# Stage-A Runtime Guide

Single-image summarization runbook for generating the evidence JSONL that Stage-B consumes. Use this file as the engineer-facing reference for commands, inputs, outputs, and common flags.

---

## Purpose
- Produce per-image summaries with strong rare/long-tail coverage for downstream group-level verdicts.

## Entrypoint
- `scripts/stage_a_infer.sh` wraps `python -m src.stage_a.cli` with mission defaults and device selection.

```bash
mission=挡风板安装检查 gpu=0 verify_inputs=true \
  bash scripts/stage_a_infer.sh
```

## Inputs
- Layout: `<root>/<mission>/{审核通过|审核不通过}/<group_id>/*.{jpg,jpeg,png}`
- Labels inferred from parent directory: `审核通过` → `pass`, `审核不通过` → `fail`.

## Outputs
- JSONL at `<output_dir>/<mission>_stage_a.jsonl` with keys `group_id`, `mission`, `label`, `images`, and normalized `per_image` entries (`image_1`, `image_2`, ...).
- Format aligns with the canonical schema in `../data/DATA_JSONL_CONTRACT.md`.

## Key Flags / Env Vars
- `mission` (required) — must match `SUPPORTED_MISSIONS`.
- `verify_inputs` — logs first-chunk hashes and grid/token counts; optional.
- `no_mission` — skip mission focus text for smoke tests.
- `gpu` / `device` — device selection (`cuda:N` or `cpu`).

## Checkpoint Choices
- LoRA adapters for iteration; merged checkpoints for production (see `scripts/train.sh` exports or `swift export --merge_lora true`).

## See Also
- Stage-B runtime (group verdicts): `./STAGE_B_RUNTIME.md`
- Business overview of the two-stage pipeline: `./STAGE_A_STAGE_B.md`
