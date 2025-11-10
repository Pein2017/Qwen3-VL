## 2025-02-14 — JSON format variants

- Added `custom.json_format` (required) to select Type A/B/C/D layouts for dense-caption outputs; retired the old `json_indent` knob.
- `DenseCaptionDataset` and `JSONLinesBuilder` now render all four shapes, keep canonical payloads for ms-swift templates, and cover them with new unit tests.
- Updated configs (`configs/base.yaml`, `configs/json_format/stage_1_gkd.yaml`) plus `format.md` so ablation runs can flip the format via YAML only.
- Dense system prompt now auto-patches to the selected `json_format`, with a shorter instruction set to reduce prompt length/OOM risk.
