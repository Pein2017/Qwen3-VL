# Qwen3-VL Training Stack

Qwen3-VL is a multimodal dense-captioning and inspection toolkit built on top of
[ms-swift](https://github.com/modelscope/ms-swift) and Hugging Face
[transformers](https://github.com/huggingface/transformers). It provides a single
repository for supervised fine-tuning, data preparation, augmentation, visual
inspection, and production-ready inference flows for telecom BBU installation
scenes.

## Highlights

- **Config-first workflow** – every experiment is expressed as YAML under
  `configs/`, consumed by the lightweight launcher in `src/sft.py`.
- **Geometry-aware augmentation** – rotation-safe canvas expansion, smart crop
  with label filtering, and polygon/line clipping tuned for dense detection.
- **Domain-aligned datasets** – JSONL schema with hierarchical attributes for
  BBU equipment, shields, labels, and cabling. Conversion utilities keep dense
  captions and Stage-A/B summaries synchronized.
- **Production stages** – Stage-A generates per-image summaries, Stage-B
  aggregates multi-image verdicts for deployment scenarios.
- **Visualization tooling** – `vis_tools/` scripts to sanity-check augmentation
  pipelines and raw annotations.

## Repository Layout

```
├── configs/             # Experiment YAMLs (stage 1–4, summary, debug)
├── docs/                # Project documentation (data, augmentation, upstream deps)
├── scripts/             # Launch helpers (train.sh, LoRA merge, GRPO)
├── src/                 # Training stack, datasets, geometry, prompts
├── vis_tools/           # Visualization CLI utilities
├── data/, data_conversion/  # Sample datasets and attribute taxonomies
└── openspec/            # Spec-driven change workflow (proposals & tasks)
```

See [`docs/README.md`](docs/README.md) for navigation across the full manual.

## Quick Start

1. Create or activate the `ms` Conda environment and install project
   dependencies (`ms-environment.yaml` is provided as a reference).
2. Prepare training JSONL files that follow the schema documented in
   [`docs/DATA_AND_DATASETS.md`](docs/DATA_AND_DATASETS.md).
3. Pick or customize a config in `configs/` (e.g., `stage_3_vision_last6_lora.yaml`).
4. Launch training:

   ```bash
   conda run -n ms bash scripts/train.sh config=/abs/path/to/config.yaml gpus=0
   ```

5. Inspect augmentation results with the visualization tools if you modify
   geometry ops:

   ```bash
   conda run -n ms python vis_tools/vis_augment_compare.py --config /abs/config.yaml
   ```

## Documentation

- [Data & Datasets](docs/data/DATA_AND_DATASETS.md)
- [Augmentation Guide](docs/data/DATA_AUGMENTATION.md)
- [Training & Inference Reference](docs/training/REFERENCE.md)
- [Upstream Dependencies](docs/platform/UPSTREAM_DEPENDENCIES.md)

## Contributing

We track feature work and larger changes through the OpenSpec workflow in
[`openspec/`](openspec/). See [`AGENTS.md`](AGENTS.md) for the global guide and
[`openspec/AGENTS.md`](openspec/AGENTS.md) for proposal & tasks instructions.

Please run formatting, linting, and tests locally before opening a pull request.

## License

This repository inherits the licenses of ms-swift and Hugging Face transformers.
See upstream projects for details. Project-specific licensing will be added when
formalized.
