# Stage-A Inference Engine

Lightweight per-image inference for Qwen3-VL generating Chinese single-line summaries for GRPO group reasoning.

## Features

- **Mission-dependent prompts**: Tailored context for 4 supported BBU QC missions
- **Batch inference**: 4-5x speedup with configurable batch size (default 8)
- **Strict validation**: Fail-fast on empty summaries or 图片_{i} misalignment
- **Flat JSONL output**: One file per mission for easy downstream processing

## Quick Start

```bash
# Simple usage - only mission and cuda device
bash scripts/stage_a_infer.sh mission=挡风板安装检查 cuda=0

# All other parameters (batch_size, max_pixels, etc.) are pre-configured in the script
# Results are written incrementally - you can monitor progress with:
tail -f output_post/stage_a/挡风板安装检查_stage_a.jsonl
```

## Input Structure

```
<input_dir>/
  <mission>/
    审核通过/
      <group_id>/
        *.{jpg,jpeg,png}
    审核不通过/
      <group_id>/
        *.{jpg,jpeg,png}
```

## Output Format

One JSONL file per mission: `<output_dir>/<mission>_stage_a.jsonl`

Each line:
```json
{
  "group_id": "QC-20250118-0001",
  "mission": "挡风板安装检查",
  "label": "pass",
  "images": ["/abs/path/img1.jpeg", "/abs/path/img2.jpeg"],
  "per_image": {
    "图片_1": "BBU设备/华为/显示完整/...",
    "图片_2": "螺丝、光纤插头/BBU安装螺丝/..."
  },
  "raw_texts": ["...", "..."],
  "clean_texts": ["...", "..."],
  "timestamp": "2025-01-18T10:30:00.000000Z"
}
```

## Supported Missions

1. `BBU安装方式检查（正装）`
2. `BBU接地线检查`
3. `BBU线缆布放要求`
4. `挡风板安装检查`

## Configuration

### CLI Arguments (Simplified)

**Only 2 arguments needed:**
- `mission=<name>`: One of the 4 supported missions (required, default: 挡风板安装检查)
- `cuda=<id>`: GPU device ID or 'cpu' (optional, default: 0)

**All other parameters are pre-configured in `scripts/stage_a_infer.sh`:**
- Checkpoint: `output/summary_merged/10-24`
- Input: `group_data/bbu_scene_2.0_order`
- Output: `output_post/stage_a`
- Batch size: `4`
- Max pixels: `786432` (1024×768 equivalent)
- Max new tokens: `1024`
- Temperature: `0.0` (greedy)

**To change defaults, edit the script directly.**

## Features

### Batch Inference
The engine uses hybrid batching:
- Groups are processed with batch_size=4 (pre-configured)
- Large groups split into chunks automatically
- ~4-5x speedup over sequential for typical groups (3-10 images)

### Streaming Output
Results are written immediately after each group:
- **No waiting** until all groups complete
- **Monitor progress** in real-time with `tail -f`
- **Safe interruption**: Ctrl+C preserves already-processed groups
- **Resume-friendly**: Can restart from where it left off

## Validation

Strict fail-fast validation per OpenSpec requirements:
- Empty summary → `ValueError` (aborts group)
- 图片_{i} mismatch → `ValueError` (aborts group)
- Non-empty `clean_text` required for all images

## Examples

### Process one mission
```bash
# Simplest form - uses all defaults
bash scripts/stage_a_infer.sh mission=挡风板安装检查 cuda=0

# Monitor progress in real-time
tail -f output_post/stage_a/挡风板安装检查_stage_a.jsonl
```

### Process all missions concurrently (4 GPUs)
```bash
gpu=0
for mission in "BBU安装方式检查（正装）" "BBU接地线检查" "BBU线缆布放要求" "挡风板安装检查"; do
  bash scripts/stage_a_infer.sh mission="$mission" cuda=$gpu &
  gpu=$((gpu + 1))
done
wait

# Monitor all outputs in separate terminals
tail -f output_post/stage_a/BBU安装方式检查（正装）_stage_a.jsonl
tail -f output_post/stage_a/BBU接地线检查_stage_a.jsonl
tail -f output_post/stage_a/BBU线缆布放要求_stage_a.jsonl
tail -f output_post/stage_a/挡风板安装检查_stage_a.jsonl
```

## Architecture

```
src/stage_a/
  ├── __init__.py       # Module exports
  ├── prompts.py        # Mission-dependent prompt builder
  ├── inference.py      # Core engine (discover, infer, aggregate)
  └── cli.py            # CLI entry point

scripts/
  └── stage_a_infer.sh  # Convenience launcher
```

## Integration with GRPO Pipeline

Output JSONL is compatible with Stage-B dataset builder:
1. Stage-A generates per-image summaries with 图片_{i} keys
2. Stage-B builder reads these summaries and creates text-only GRPO dataset
3. GRPO training uses group-level labels (通过/不通过) for supervision

## Alignment with OpenSpec

Implements `openspec/changes/2025-10-24-add-grpo-group-reasoning/specs/stage-a-inference/spec.md`:
- ✅ Per-image inference with grouped aggregation
- ✅ Strict 图片_{i} alignment and coverage validation
- ✅ Native chat_template via HF processor
- ✅ Mission-dependent Chinese prompts
- ✅ Fail-fast on empty summaries

## Troubleshooting

**Out of memory**:
- Reduce `--batch_size` (try 4 or 2)
- Use CPU: `--device cpu`

**Empty summaries**:
- Check checkpoint is trained on summary variant
- Verify image quality and format
- Try higher temperature: `--temperature 0.5`

**Mission not recognized**:
- Ensure exact match (including parentheses and full-width chars)
- Use one of: `BBU安装方式检查（正装）`, `BBU接地线检查`, `BBU线缆布放要求`, `挡风板安装检查`

**No groups found**:
- Verify input directory structure matches expected format
- Check mission directory exists: `<input_dir>/<mission>/`
- Ensure label directories exist: `审核通过`, `审核不通过`

