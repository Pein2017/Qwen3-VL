# Qwen3-VL Training Documentation

Status: Active — Internal Engineering

Welcome to the Qwen3-VL training documentation. This guide covers end-to-end workflows for fine-tuning Qwen3-VL on dense captioning and quality control tasks.

## Quick Navigation

### I want to...

**Train a model**
→ Start with [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- Configuration (YAML structure, inheritance)
- Training modes (full, LoRA, mixed)
- Stage 1 & 2 workflows
- Troubleshooting

**Prepare training data**
→ Read [DATA_FORMATS.md](DATA_FORMATS.md)
- JSONL schema and examples
- Dense captioning vs summary modes
- Geometry formats (bbox, quad, line)
- Data verification tools

**Run inference**
→ See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
- Loading checkpoints and adapters
- Stage-A: Image-level summarization
- Stage-B: Group-level judgment
- Merging adapters for deployment

**Understand advanced topics**
→ Check [REFERENCE.md](REFERENCE.md)
- Vision encoder architecture
- Learning rate schedules
- DeepSpeed configuration
- Augmentation pipeline
- Template mechanics

### Quick Start

```bash
# 1. Prepare your data (JSONL format)
# Review docs/DATA_FORMATS.md. Ensure each record has
# images, objects, width, height (and summary if used).

# 2. Configure training (edit YAML)
cp configs/base.yaml configs/my_experiment.yaml
# Edit configs/my_experiment.yaml with your paths and hyperparameters

# 3. Run training
python -m src.sft --config configs/my_experiment.yaml

# 4. Run inference
# With adapter (lightweight)
CUDA_VISIBLE_DEVICES=0 conda run -n ms swift infer \
  --model /path/to/base/Qwen3-VL \
  --adapters output/my_experiment/checkpoint-XXX

# Or merge adapter into base for deployment (recommended for prod)
CUDA_VISIBLE_DEVICES=0 conda run -n ms swift export \
  --model /path/to/base/Qwen3-VL \
  --adapters output/my_experiment/checkpoint-XXX \
  --merge_lora true \
  --output_dir output/merged/checkpoint-XXX \
  --safe_serialization true \
  --max_shard_size 5GB
```

## Document Structure

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | Complete training workflows and configuration | Setting up or debugging training |
| [DATA_FORMATS.md](DATA_FORMATS.md) | Data schemas and preparation | Preparing datasets |
| [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) | Running inference and deployment | After training, deploying models |
| [REFERENCE.md](REFERENCE.md) | Advanced topics and internals | Deep dives, optimization |

## Key Concepts

**Dense Captioning**: Structured object detection with geometry (bounding boxes, quads, lines) and hierarchical descriptions.

**Summary Mode**: Lightweight per-image text summaries without geometry, useful for group-level reasoning.

**Dynamic Pairing**: Epoch-seeded multi-image grouping for training on image groups rather than individual images.

**LoRA Training**: Parameter-efficient fine-tuning that saves only adapter weights (~240MB) instead of full model (~9.6GB).

**Two-Stage Training**: First train aligner only, then train LLM+aligner while keeping vision tower frozen for efficiency.

## Project Structure

```
Qwen3-VL/
├── src/                    # Core training code
│   ├── config/             # YAML loading, prompts, missions
│   ├── datasets/           # Data loading, preprocessing, augmentation
│   ├── stage_a/            # Image-level summarization
│   ├── stage_b/            # Group-level judgment
│   └── sft.py              # Training entry point
├── configs/                # Experiment YAML files
├── scripts/                # Operational helpers
│   ├── train.sh
│   ├── stage_a_infer.sh
│   ├── merge_stage2_lora.sh
│   ├── run_grpo.py
│   └── inspect_lora_ckpts.py
├── docs/                   # Documentation (you are here)
└── data/                   # Training data (JSONL format)
```

## Getting Help

- **Errors during training**: Check troubleshooting section in [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Data format issues**: Run verification tools from [DATA_FORMATS.md](DATA_FORMATS.md)
- **Inference problems**: See common issues in [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
- **Performance tuning**: Consult [REFERENCE.md](REFERENCE.md) for optimization tips

## Additional Resources

- **Archived documentation**: See `docs/archive/` for older guides and historical context
- **Code architecture**: See `src/README.md` for implementation details
- **Key source files**: `src/sft.py`, `src/config/prompts.py`, `src/datasets/builders/jsonlines.py`, `src/datasets/geometry.py`
- **ms-swift library**: `/data/ms-swift` (upstream training framework)
- **HF transformers**: `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`

---

**Last Updated**: October 25, 2025  
**Target Model**: Qwen3-VL-4B-Instruct  
**Use Case**: BBU Equipment Quality Control

