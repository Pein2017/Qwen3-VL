# Upstream Dependencies

Background on the two primary libraries this project builds on: Hugging Face's Qwen3-VL implementation and the ms-swift training framework. Use this as a quick reference when you need to trace behavior into upstream code or reason about configuration limits.

---

## Hugging Face Qwen3-VL Model

_Source: `transformers.models.qwen3_vl` (installed under `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/`)._

### Architecture Highlights
- **Config class**: `Qwen3VLConfig` glues together a text config (`Qwen3VLTextConfig`) and a vision config (`Qwen3VLVisionConfig`). It also stores multimodal token ids (`vision_start_token_id`, `vision_end_token_id`, `image_token_id`, `video_token_id`).
- **Model class**: `Qwen3VLForConditionalGeneration` wraps a `Qwen3VLModel` backbone plus a tied `lm_head`. The backbone exposes convenience accessors:
  - `model.visual` — vision encoder + aligner stack.
  - `model.language_model` — the LLM (Qwen3-style decoder with rotary embeddings).
  - `get_image_features(pixel_values, image_grid_thw)` — run only the visual branch.
- **Vision pathway**: ViT encoder → deepstack aligner. Vision features are projected through
  `model.visual.merger` and `model.visual.deepstack_merger_list.{0,1,2}` before replacing `<|image_pad|>` placeholders.
- **Rope handling**: the model caches `rope_deltas` to keep multimodal rotary embeddings aligned. `image_grid_thw` (time/height/width) is used to adjust rope positions.
- **Mask expectations**: image tokens are masked out in labels (`-100`), while attention masks can arrive as 4D tensors (prefill vs. decode). The Qwen code auto-flattens 4D masks.

### Template & Token Mechanics
- Chat templates insert start/end placeholders using the ids above. Do not manually insert `<|image_pad|>` tokens; use the HF-provided template (mirrored in this repo).
- Placeholder count must equal the product of the grid dimensions provided in `image_grid_thw`.
- During mixed image/video input the model distinguishes token ids (`image_token_id` vs `video_token_id`) and expects matching `pixel_values` / `pixel_values_videos` tensors.

### Common Integration Points
- **Freezing / LoRA**: the project’s YAMLs freeze `model.visual` components selectively. When adding LoRA targets, reference the actual module names (`model.visual.merger`, `model.visual.deepstack_merger_list.X`) from the HF implementation.
- **Generation config**: `SwiftSft` replaces `model.generation_config` per run. The default config from HF sets repetition penalties and default image token ids; rely on `prepare_generation_config` to keep these in sync.

---

## ms-swift Training Framework

_Source: `/data/ms-swift/swift/` (notably `swift/llm/train/sft.py` and `swift/llm/argument/train_args.py`)._

### SwiftSft Pipeline
- **Entry point**: `SwiftSft` (inherits `SwiftPipeline` + `TunerMixin`) orchestrates loading the model/processor, template, datasets, collator, and trainer.
- **Model loading**: `args.get_model_processor()` returns `(model, processor)`; padding-free or packing modes require flash attention kernels.
- **Template**: `args.get_template(processor)` loads the multimodal chat template and verifies packing compatibility (`template.support_padding_free`). Templates can hook into the model for multimodal masking.
- **Dataset preparation**:
  1. Load JSONL via `load_dataset()` (returns HF Dataset or iterable dataset).
  2. Encode using `LazyLLMDataset` + template.encode (handles multimodal fields).
  3. Optional packing via `PackingDataset` / `IterablePackingDataset`.
  4. Cache handling via `args.cached_dataset` (on-disk Arrow splits).
- **Collation**: `template.data_collator` is partially applied with `padding_to` when `train_type == 'longlora'`. For other regimes the template decides padding/packing logic.
- **Model preparation**: `TunerMixin.prepare_model(...)` applies LoRA/full tuning rules, sets `modules_to_save`, and ensures adapter checkpoints if requested.
- **Trainer factory**: `TrainerFactory.get_trainer_cls(args)` selects a Hugging Face Trainer subclass (standard or DeepSpeed/sequence parallel variants). The resulting trainer receives the template to keep multimodal masking consistent.

### TrainArguments
- Consolidated dataclass that merges base arguments, tuning options, and Seq2Seq overrides.
- Handles path normalization (`resume_from_checkpoint`, `adapters`), DeepSpeed config resolution (named presets → JSON), and padding-free checks (requires flash attention implementation).
- Builds `training_args` (HF `Seq2SeqTrainingArguments`) through `TrainerFactory.get_training_args(self)` with `remove_unused_columns = False` to preserve multimodal tensors.
- Enforces dataset presence: either `dataset` YAML entries or cached datasets must be provided—this is why the repo always supplies `custom.train_jsonl` / `custom.val_jsonl`.

### Packing & Streaming Modes
- `padding_free` or `packing` set in YAML propagate to `TrainArguments`, which in turn select appropriate dataset wrappers.
- Packing requires flash attention; ms-swift will raise if incompatible (`attn_impl` must be `flash_attn`/`flash_attention_*`).
- Streaming datasets use `EncodePreprocessor` to tokenize on the fly while respecting template logic.

### Callbacks & Metrics
- Extra callbacks (`swift.plugin.extra_callbacks`) added via `SwiftSft._prepare_callbacks()` include logging helpers, checkpoint throttling, and optional visualization.
- `SwiftSft._save_val_dataset()` saves the validation split to `val_dataset.jsonl` on rank 0 when the validation set is carved from training data—useful for reproduction.

---

## Practical Tips
- When debugging a mismatch, trace whether it originates upstream (HF template/model) or in-repo glue code. The modules above are the first places to inspect.
- If adding new LoRA targets or freezing logic, confirm the actual module names in `modeling_qwen3_vl.py` and update YAML accordingly.
- For new training modes, verify ms-swift already supports them (`swift/llm/train/` contains SFT, PT, RLHF, KTO). Extend SwiftSft only if the functionality is absent.
- Keep this document updated when upstream versions change (e.g., Transformer release bumps or ms-swift upgrades).

**Last Reviewed:** 2025-10-28
