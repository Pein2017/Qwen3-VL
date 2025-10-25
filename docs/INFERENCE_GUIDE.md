# Inference Guide

Status: Active — Internal Engineering

Guide to running inference with trained Qwen3-VL models for dense captioning and quality control tasks.

## Table of Contents

- [Loading Checkpoints](#loading-checkpoints)
- [Dense Captioning Inference](#dense-captioning-inference)
- [Stage-A: Image-Level Summarization](#stage-a-image-level-summarization)
- [Stage-B: Group-Level Judgment](#stage-b-group-level-judgment)
- [Merging Adapters](#merging-adapters)

## Loading Checkpoints

### With LoRA Adapter (Lightweight)

Load base model + adapter weights (~240MB):

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ms swift infer \
  --model /path/to/base/Qwen3-VL-4B-Instruct \
  --adapters output/stage_2/checkpoint-XXX \
  --stream true \
  --max_new_tokens 2048
```

**Pros:**
- Fast loading
- Small storage footprint
- Easy to swap adapters

**Cons:**
- Slight inference overhead (adapter merging at runtime)
- Requires both base model and adapter

### With Merged Checkpoint (Deployment)

Load fully merged model (~9.6GB):

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ms swift infer \
  --model output/merged/checkpoint-XXX \
  --stream true \
  --max_new_tokens 2048
```

**Pros:**
- Faster inference (no adapter overhead)
- Self-contained (single checkpoint)

**Cons:**
- Large storage footprint
- Requires merging step (see [Merging Adapters](#merging-adapters))

## Dense Captioning Inference

Inference for grouped JSON with geometry (bbox/quad/line) + descriptions.

### Interactive CLI

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ms swift infer \
  --model /path/to/checkpoint \
  --stream true \
  --max_new_tokens 2048
```

**Input:** Upload images and provide prompt

**Output:** Grouped JSON with geometry and descriptions

**Example output:**
```json
{
  "图片_1": {
    "object_1": {
      "bbox_2d": [264, 144, 326, 201],
      "desc": "BBU设备/华为/显示完整/安装牢固"
    },
    "object_2": {
      "quad": [100, 50, 200, 55, 195, 150, 95, 145],
      "desc": "标签/5G-BBU光纤/可以识别"
    }
  },
  "图片_2": {
    "object_1": {
      "line": [458, 0, 476, 55, 494, 102, 510, 150],
      "line_points": 4,
      "desc": "光纤/有保护措施/蛇形管/弯曲半径合理"
    }
  }
}
```

### Programmatic Inference

```python
from swift.llm import get_model_tokenizer, get_template, inference
from transformers import AutoProcessor

# Load model and template
model_path = "/path/to/checkpoint"
model, tokenizer = get_model_tokenizer(
    model_path,
    torch_dtype="bfloat16",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)
template = get_template(tokenizer.model_dir, tokenizer, processor=processor)

# Prepare messages
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "/path/to/img1.jpg"},
        {"type": "image", "image": "/path/to/img2.jpg"},
        {"type": "text", "text": "请识别图中设备与几何位置，并用规范标记输出。"}
    ]
}]

# Run inference
response = inference(model, template, messages, max_new_tokens=2048)
print(response["response"])
```

## Stage-A: Image-Level Summarization

Generate one-line Chinese summaries per image, grouped by scene.

### Purpose

Stage-A provides lightweight per-image summaries without geometry, useful for:
- Group-level reasoning (Stage-B input)
- Quality control workflows
- Quick image screening

### CLI Usage

```bash
bash scripts/stage_a_infer.sh \
  --model /path/to/checkpoint \
  --data_root /path/to/data \
  --mission "挡风板安装检查" \
  --output_dir output_post/stage_a
```

**Input Structure:**
```
data_root/
├── 挡风板安装检查/
│   ├── 审核通过/
│   │   ├── QC-20230222-0000297/
│   │   │   ├── img-001.jpeg
│   │   │   └── img-002.jpeg
│   │   └── QC-20230311-0000741/
│   │       └── img-001.jpeg
│   └── 审核不通过/
│       └── QC-20230315-0000815/
│           ├── img-001.jpeg
│           └── img-002.jpeg
```

**Output JSONL** (one line per group):
```json
{
  "group_id": "QC-20230222-0000297",
  "mission": "挡风板安装检查",
  "label": "pass",
  "images": ["/abs/path/img-001.jpeg", "/abs/path/img-002.jpeg"],
  "per_image": {
    "图片_1": "BBU设备/华为/显示完整/安装牢固",
    "图片_2": "螺丝、光纤插头/BBU安装螺丝/符合要求"
  },
  "timestamp": "2025-10-24T12:34:56.000000Z"
}
```

### Supported Missions

- BBU安装方式检查（正装）
- BBU接地线检查
- BBU线缆布放要求
- 挡风板安装检查

Mission-specific focus is automatically appended to prompts (see `src/config/missions.py`).

### Batching

Stage-A processes one image at a time per forward pass, batched across images for efficiency:

```python
# In scripts/stage_a_infer.sh
--batch_size 4  # Process 4 images in parallel
```

## Stage-B: Group-Level Judgment

Given Stage-A summaries, infer a group-level Pass/Fail verdict with reasoning.

### Purpose

Stage-B provides final quality control judgment based on aggregated image summaries:
- Input: Text summaries from Stage-A (no images)
- Output: Two-line verdict (通过/不通过 + reasoning)

### Input Format

Stage-B reads Stage-A JSONL output and builds text-only prompts:

```python
# Example: scripts/run_grpo.py or custom inference
from src.stage_b import build_stage_b_messages

messages = build_stage_b_messages(
    stage_a_summaries={
        "图片_1": "BBU设备/华为/显示完整/安装牢固",
        "图片_2": "螺丝、光纤插头/BBU安装螺丝/符合要求"
    },
    task_type="挡风板安装检查"
)
```

### Output Format

**Strictly two lines:**
1. First line: `通过` or `不通过` (verdict only, no extra text)
2. Second line: `理由: <reasoning in natural Chinese>`

**Example:**
```
通过
理由: 图片_1和图片_2均显示BBU设备安装牢固，螺丝符合要求，无需安装挡风板。
```

### CLI Usage

Stage-B is typically used in GRPO training context. For inference:

```python
from swift.llm import inference

# Build messages from Stage-A output
messages = build_stage_b_messages(summaries, task_type)

# Run inference
response = inference(model, template, messages, max_new_tokens=128)
print(response["response"])
```

### Validation

Check output format:
- Exactly two lines (split by `\n`)
- First line is exactly `通过` or `不通过`
- Second line starts with `理由:` and has content

## Merging Adapters

Merge LoRA adapter into base model for deployment.

### Why Merge?

- **Faster inference**: No adapter overhead
- **Simpler deployment**: Single checkpoint
- **Better compatibility**: Works with any inference framework

### Merge Command

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ms swift export \
  --model /path/to/base/Qwen3-VL-4B-Instruct \
  --adapters output/stage_2/checkpoint-XXX \
  --merge_lora true \
  --output_dir output/merged/checkpoint-XXX \
  --safe_serialization true \
  --max_shard_size 5GB
```

**Parameters:**
- `--model`: Base model path (same as training)
- `--adapters`: LoRA adapter checkpoint
- `--merge_lora true`: Enable merging (not just export)
- `--output_dir`: Where to save merged checkpoint
- `--save_safetensors true`: Use SafeTensors format (recommended)

### Verify Merged Checkpoint

```bash
# Check size (should be ~9.6GB for Qwen3-VL-4B)
du -sh output/merged/checkpoint-XXX/

# Test inference
CUDA_VISIBLE_DEVICES=0 conda run -n ms swift infer \
  --model output/merged/checkpoint-XXX
```

### Merging Multiple Adapters

If you have multiple stage adapters (e.g., stage1 + stage2), merge sequentially:

```bash
# Step 1: Merge stage1 into base
swift export \
  --model /base/Qwen3-VL \
  --adapters output/stage_1/checkpoint-100 \
  --merge_lora true \
  --output_dir temp/stage_1_merged

# Step 2: Merge stage2 into stage_1_merged
swift export \
  --model temp/stage_1_merged \
  --adapters output/stage_2/checkpoint-200 \
  --merge_lora true \
  --output_dir output/final_merged
```

**Alternatively**, resume training in stage2 loads stage1 adapter automatically, so you only need to merge the final stage2 checkpoint.

## Troubleshooting

### Slow Inference

**Symptom:** Inference takes too long per sample.

**Solutions:**
1. Merge adapters (remove adapter overhead)
2. Use flash_attention_2:
   ```python
   model, tokenizer = get_model_tokenizer(
       model_path,
       attn_impl="flash_attention_2"
   )
   ```
3. Reduce max_new_tokens for shorter outputs
4. Use bfloat16 or int8 quantization

### Wrong Output Format

**Symptom:** Model doesn't follow grouped JSON or two-line format.

**Cause:** Training prompt mismatch or insufficient training.

**Solutions:**
1. Verify system prompt matches training (see `src/config/prompts.py`)
2. Check that model was trained on the correct format
3. Add few-shot examples in user prompt
4. Lower temperature for more deterministic output:
   ```python
   response = inference(..., temperature=0.1, top_p=0.9)
   ```

### Missing Geometry in Dense Output

**Symptom:** Dense captioning doesn't include bbox/quad/line.

**Cause:** Model trained with `summary_ratio=1.0` (summary-only).

**Solution:** Retrain with `summary_ratio=0.0` or use a dense-trained checkpoint.

### Image Loading Errors

**Symptom:** Can't load images during inference.

**Cause:** File path issues or format incompatibility.

**Solutions:**
1. Use absolute paths: `{"type": "image", "image": "/abs/path/img.jpg"}`
2. Verify image format (JPG/PNG supported)
3. Check file permissions
4. Convert HEIC/WebP to JPG if needed:
   ```bash
   convert input.heic output.jpg
   ```

## Additional Resources

- **Training workflows**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Data preparation**: See [DATA_FORMATS.md](DATA_FORMATS.md)
- **Advanced topics**: See [REFERENCE.md](REFERENCE.md)

---

**Last Updated**: October 25, 2025
