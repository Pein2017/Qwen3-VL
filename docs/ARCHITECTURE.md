# Architecture

Source of truth: `src/sft.py`, `src/README.md`, `src/datasets/dense_caption.py`, `src/datasets/dynamic_pair.py`, `src/datasets/builders/jsonlines.py`, `src/datasets/geometry.py`, `src/utils/README.md`, `src/callbacks/save_delay_callback.py`

## End-to-end pipeline (config-driven)
- YAML → `ConfigLoader` → `SwiftSft` → `DenseCaptionDataset` → `DynamicPairDataset` → Trainer (ms‑swift)
- Single knob for length: `global_max_length` (proxies both model/template)
- Adapters applied before trainer: `sft.prepare_model(...)`

## Model components and token flow
- Vision encoder (ViT) → Aligner (projector) → LLM
- The chat_template inserts image placeholders automatically; do not hand‑craft `<|image_pad|>`.
- Placeholder count scales with `image_grid_thw`; vision embeddings replace placeholders at runtime.

## Data flow at a glance
1) Load JSONL record (images, objects, width, height, optional summary)
2) Group/pair (epoch‑seeded RNG) → optional augmentation
3) Build one‑turn messages with `JSONLinesBuilder` (user embeds all images; assistant returns grouped JSON)
4) Template encodes: adds vision tokens, normalizes top‑level objects bbox to norm1000, tokenizes
5) Trainer consumes tensors: `input_ids`, `labels`, `pixel_values`, `image_grid_thw`, `objects`

## Datasets and builders (where shaping happens)
- `DenseCaptionDataset`: selects dense vs summary per pairing group; configures augmentation
- `DynamicPairDataset`: engine for pairing and per‑item orchestration
- `JSONLinesBuilder`: formats grouped JSON for dense mode, one‑line per image for summary mode

## Logging and callbacks
- Unified, rank‑aware logging: see `src/utils/README.md`
- Early save throttling: `SaveDelayCallback` blocks saves until a warmup step threshold

## Health checks (fail‑fast)
- Image count in user turn matches placeholders
- `image_grid_thw` aligns with `pixel_values`
- Assistant spans end correctly; image tokens masked; non‑target tokens labeled −100
- Geometry kept at top level in pixel space; template normalizes to norm1000 during encoding


