# Qwen3‑VL Inference & Stage-A Guide

## Inference

### Checkpoints

**Adapter-Based** (Recommended for Development):
```bash
# Load base + LoRA adapter
CUDA_VISIBLE_DEVICES=0 swift infer \
  --model /path/to/Qwen3-VL-4B-Instruct \
  --adapters /path/to/checkpoint-XXX \
  --stream true --max_new_tokens 2048
```

**Benefits**:
- Small adapter file (~240MB)
- Flexible (swap adapters easily)
- Easy to version control

**Merged** (Recommended for Production):
```bash
# Merge adapter into base model
CUDA_VISIBLE_DEVICES=0 swift export \
  --model /path/to/Qwen3-VL-4B-Instruct \
  --adapters /path/to/checkpoint-XXX \
  --merge_lora true \
  --output_dir /path/to/merged \
  --save_safetensors true
```

**Benefits**:
- Single self-contained checkpoint
- Faster inference (no adapter overhead)
- Easier deployment

### Dense Captioning (Standard)

**Input Format**:
```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "img1.jpg"},
        {"type": "text", "text": "请描述图片中的所有物体"}
    ]
}]
```

**Output Format**:
```json
{
  "object_1": {"bbox_2d": [100, 200, 300, 400], "desc": "BBU设备/品牌:华为/型号:5900"},
  "object_2": {"line_points": 4, "line": [50, 60, 80, 120, 130, 180, 180, 220], "desc": "光纤/颜色:黄色/保护:有保护"}
}
```

**Best Practices**:
- Use deterministic generation (low temperature) for deployment
- Set `max_new_tokens` based on expected output length
- Stream responses for better UX

### Stage-A & Stage-B

#### Stage-A CLI (`src.stage_a.cli`, `scripts/stage_a_infer.sh`)

- Purpose: Emit single-line summaries per image for Stage-B ingestion.
- Input layout: `<root>/<mission>/{审核通过|审核不通过}/<group_id>/*.{jpg,jpeg,png}`. Labels are inferred from the parent directory name (`审核通过` → `pass`, `审核不通过` → `fail`).
- Validation: `StageAConfig` checks mission choices, checkpoint path, and positive batch/max token parameters. Optional `--verify_inputs` logs image size/hash plus grid/token counts for the first item in each processed chunk; no `stage_a_complete` marker is required.
- Launch with the script wrapper to keep mission defaults, verification flags, and device selection uniform:

```bash
mission=挡风板安装检查 gpu=0 verify_inputs=true \
  bash scripts/stage_a_infer.sh
```

Key flags/env vars:
- `mission` — Mission name (appears in output JSONL and Stage-B guidance; must be one of `SUPPORTED_MISSIONS`).
- `verify_inputs` — Enable per-chunk logging of image hashes/grid sizes.
- `no_mission` — Skip mission focus instructions for generic smoke tests.
- `gpu` / `device` — Device selection (`cuda:N` or `cpu`).

Output format (per group JSONL):

```json
{
  "group_id": "QC-0001",
  "mission": "挡风板安装检查",
  "label": "pass",
  "images": ["img1.jpg", "img2.jpg"],
  "per_image": {
    "image_1": "BBU设备×1，光模块×3，线缆×2",
    "image_2": "BBU设备×1，光模块×2"
  }
}
```
Output path: `<output_dir>/<mission>_stage_a.jsonl` (defaults baked into the script).

#### Stage-B Runtime

Stage-B ingest/selection/reflection is documented in [STAGE_B_RUNTIME.md](STAGE_B_RUNTIME.md). That guide covers sampler grids, CriticEngine, manual-review gating, and mission-specific guidance workflows.

### Deployment Tips

**Latency Optimization**:
1. Use merged checkpoints (faster than adapter loading)
2. Enable Flash Attention 2 (default for Qwen3-VL)
3. Use bfloat16 precision (balance speed/quality)
4. Batch multiple images when possible

**Memory Optimization**:
1. Use quantization (int8/int4) for large-scale deployment
2. Lower `max_model_len` if not needed
3. Use gradient checkpointing during training (no impact on inference)

**Quality Optimization**:
1. Set temperature=0 for deterministic output
2. Use appropriate `top_p` (0.9 default)
3. Validate output format with regex/JSON parsing
4. Keep processor/template aligned with base model

---

## Stage‑A implementation notes

- Uses batched processing within each group (`batch_size` controls chunking over images).
- Summaries are sanitized to single strings per image; `per_image` keys are normalized to `image_1`, `image_2`, ... even if the model returns JSON objects.
- Native Hugging Face chat template is used; no custom formatting beyond mission focus text.

## Dense/Summary mode (training datasets)

Current datasets run in a **single mode** per config: `custom.use_summary: true` enables summary-only mode; otherwise dense JSON mode is used. Mixed dense/summary sampling is not implemented.
