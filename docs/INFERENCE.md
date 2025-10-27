# Inference

Source of truth: `docs/INFERENCE_GUIDE.md`, `src/stage_a/README.md`, `src/stage_b/README.md`, `scripts/*`

## Checkpoints
- Adapter-based: load base + LoRA adapter (small, flexible)
- Merged: single, self-contained checkpoint (faster inference)

## Dense captioning (standard)
- Messages: user embeds all images; assistant returns grouped JSON
- Deterministic output favored for deployment (low temperature)

## Stage‑A (per-image summaries)
- Generates one-line Chinese summaries per image; grouped by scene
- Output JSONL consumed by Stage‑B

## Stage‑B (group-level verdict)
- Builds prompts from Stage‑A summaries (no images)
- Output: two lines — 第一行`通过|不通过`; 第二行以`理由:`开头

## Deployment tips
- Prefer merged checkpoints for latency
- Keep processor/template aligned with the base model

## See also
- Training: `docs/TRAINING.md`  ·  Data: `docs/DATA.md`  ·  Architecture: `docs/ARCHITECTURE.md`
