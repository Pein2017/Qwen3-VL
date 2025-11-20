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

- Purpose: Emit single-line summaries per image that Stage-B (and downstream BI tooling) consume.
- Validation: `StageAConfig` enforces checkpoint paths, mission names, token budgets, and verifies input folders before any inference begins.
- Launch with the script wrapper to keep mission defaults, verification flags, and device selection uniform:

```bash
mission=挡风板安装检查 gpu=0 verify_inputs=true \
  bash scripts/stage_a_infer.sh
```

Key flags:
- `mission` — Mission focus string (appears in output JSONL and Stage-B guidance).
- `verify_inputs` — Checks `stage_a_complete` markers + directory contents before running.
- `no_mission` — Skip mission focus instructions for generic smoke tests.

Output format (per group JSONL):

```json
{
  "group_id": "QC-0001",
  "mission": "挡风板安装检查",
  "label": "审核通过",
  "stage_a_complete": true,
  "images": ["img1.jpg", "img2.jpg"],
  "per_image": {
    "image_1": "BBU设备×1，光模块×3，线缆×2",
    "image_2": "BBU设备×1，光模块×2"
  }
}
```

#### Stage-B Runtime

Stage-B ingest/selection/reflection is documented in [STAGE_B_RUNTIME.md](STAGE_B_RUNTIME.md). That guide covers sampler grids, CriticEngine, manual-review gating, mission-specific guidance workflows, and the GRPO experimentation script.

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

## Stage‑A implementation notes (from archive)

- Hybrid batching gives ~4–5× throughput vs sequential
- Strict validation: non‑empty summaries; contiguous `object_{n}` indices; deterministic ordering
- Native chat_template via HF; no custom wrapper required
- Flat JSONL per mission enables easy downstream GRPO loading

## Dense/Summary mixed mode design (from archive)

- Mode is chosen per sample via epoch-seeded RNG (dense vs summary)
- Summary mode requires valid `summary` on all records
- Selection is deterministic per epoch (seeded RNG)
- Dataset temporarily injects the appropriate system prompt per group during encoding
