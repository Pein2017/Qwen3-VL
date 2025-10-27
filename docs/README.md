# Qwen3‑VL Docs

Status: Active — Internal Engineering

## Quick Navigation
- Architecture → `ARCHITECTURE.md`
- Data (schema, normalization, verification) → `DATA.md`
- Datasets (builders, preprocessors, augmentation, packing) → `DATASETS.md`
- Training (config‑light, modes, two‑stage, health checks) → `TRAINING.md`
- Inference (dense, Stage‑A/B, deployment) → `INFERENCE.md`
- Advanced/FAQ → `REFERENCE.md` (optional)

## Doc ↔ Code Map
- Pipeline: `ARCHITECTURE.md` ↔ `src/sft.py`, `src/README.md`
- Data contract: `DATA.md` ↔ `src/datasets/data_details.md`, `src/datasets/geometry.py`
- Datasets: `DATASETS.md` ↔ `src/datasets/*`
- Training: `TRAINING.md` ↔ `src/sft.py`, `configs/*`
- Inference: `INFERENCE.md` ↔ `src/stage_a/*`, `src/stage_b/*`, `scripts/*`

## Notes
- Flat structure (no subfolders) for easy maintenance
- Keep pages short (≤100 lines target); use anchors and cross‑links
- Archived content stays in `docs/archive/` with redirect notes

Last Updated: October 26, 2025

