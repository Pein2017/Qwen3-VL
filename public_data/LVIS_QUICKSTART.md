# LVIS Dataset Quick Start

**LVIS First** - Focused workflow for LVIS v1.0 (1203 categories, long-tail distribution)

## Important Note

**LVIS uses COCO 2017 images** - LVIS only provides new annotations, the images are from COCO.
- You can download both together OR
- If you already have COCO 2017 images, just download LVIS annotations

---

## Option 1: Download Everything (LVIS + Images)

```bash
cd /data/public_data

# Download LVIS annotations + COCO 2017 images (~25 GB total)
conda run -n ms python scripts/download_lvis.py
```

## Option 2: Already Have COCO Images? Download Only LVIS Annotations

```bash
cd /data/public_data

# Download only LVIS annotations (~60 MB)
conda run -n ms python scripts/download_lvis.py --skip_images

# Then create symlinks to your existing COCO images
mkdir -p lvis/raw/images
ln -s /path/to/your/coco/train2017 lvis/raw/images/train2017
ln -s /path/to/your/coco/val2017 lvis/raw/images/val2017
```

---

## Quick Pipeline (1K samples for testing)

```bash
cd /data/public_data

# 1. Download (if not done)
conda run -n ms python scripts/download_lvis.py

# 2. Convert train split (test with 1K samples)
conda run -n ms python scripts/convert_lvis.py \
  --split train \
  --max_samples 1000

# 3. Validate output
conda run -n ms python scripts/validate_jsonl.py \
  lvis/processed/train.jsonl

# 4. Create stratified sample (500 samples)
conda run -n ms python scripts/sample_dataset.py \
  --input lvis/processed/train.jsonl \
  --output lvis/processed/samples/train_500_stratified.jsonl \
  --num_samples 500 \
  --strategy stratified

# 5. Validate sample
conda run -n ms python scripts/validate_jsonl.py \
  lvis/processed/samples/train_500_stratified.jsonl
```

---

## Full Pipeline (All Data)

```bash
cd /data/public_data

# 1. Download all
conda run -n ms python scripts/download_lvis.py

# 2. Convert train split (all ~100K images, takes ~10 min)
conda run -n ms python scripts/convert_lvis.py --split train

# 3. Convert val split
conda run -n ms python scripts/convert_lvis.py --split val

# 4. Create training samples (5K, stratified - recommended)
conda run -n ms python scripts/sample_dataset.py \
  --input lvis/processed/train.jsonl \
  --output lvis/processed/samples/train_5k_stratified.jsonl \
  --num_samples 5000 \
  --strategy stratified \
  --stats

# 5. Validate training sample
conda run -n ms python scripts/validate_jsonl.py \
  lvis/processed/samples/train_5k_stratified.jsonl
```

---

## Expected Directory Structure

After setup:

```
./lvis/
├── raw/
│   ├── annotations/
│   │   ├── lvis_v1_train.json          # LVIS annotations
│   │   └── lvis_v1_val.json
│   └── images/                          # COCO 2017 images
│       ├── train2017/
│       │   ├── 000000000001.jpg
│       │   └── ...
│       └── val2017/
│           └── ...
├── processed/
│   ├── train.jsonl                      # Full converted dataset
│   ├── train_stats.json                 # Conversion statistics
│   ├── val.jsonl
│   └── samples/
│       ├── train_500_stratified.jsonl   # Quick test sample
│       ├── train_5k_stratified.jsonl    # Main training sample
│       └── train_10k_stratified.jsonl
└── metadata/
```

---

## Sampling Strategies

### Stratified (Recommended for LVIS)
Preserves long-tail distribution (Frequent/Common/Rare categories)
```bash
python scripts/sample_dataset.py \
  --input lvis/processed/train.jsonl \
  --output lvis/processed/samples/train_5k_stratified.jsonl \
  --num_samples 5000 \
  --strategy stratified
```

### Uniform (Balanced Evaluation)
Equal samples per category
```bash
python scripts/sample_dataset.py \
  --input lvis/processed/train.jsonl \
  --output lvis/processed/samples/train_3k_uniform.jsonl \
  --num_samples 3000 \
  --strategy uniform
```

### Top-K (Common Objects Only)
Focus on most frequent K categories
```bash
python scripts/sample_dataset.py \
  --input lvis/processed/train.jsonl \
  --output lvis/processed/samples/train_5k_top100.jsonl \
  --num_samples 5000 \
  --strategy top_k \
  --top_k 100
```

---

## Integration with Qwen3-VL Training

Create training config at `/data/Qwen3-VL/configs/lvis_stage1.yaml`:

```yaml
model:
  model: /path/to/Qwen3-VL-4B-Instruct

template:
  template: qwen3_vl
  max_length: 4096

tuner:
  train_type: lora
  target_modules: [all-linear]
  freeze_llm: true
  freeze_vit: true
  freeze_aligner: false

training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 1e-4
  packing: true

custom:
  train_jsonl: ./lvis/processed/samples/train_5k_stratified.jsonl
  val_jsonl: ./lvis/processed/val.jsonl
  emit_norm: norm1000
  images_per_user_turn: 1

global_max_length: 4096
```

Launch training:
```bash
cd /data/Qwen3-VL
bash scripts/train.sh config=/data/Qwen3-VL/configs/lvis_stage1.yaml gpus=0
```

---

## Dataset Statistics (LVIS v1.0)

- **Total categories**: 1203
  - Frequent (>100 instances): 337 categories
  - Common (10-100 instances): 461 categories
  - Rare (<10 instances): 405 categories

- **Train split**: 100,170 images, 1,270,141 annotations
- **Val split**: 19,809 images, 244,707 annotations
- **Avg annotations/image**: ~12.7

---

## Troubleshooting

### Images not found during conversion
```bash
# Check if images exist
ls ./lvis/raw/images/train2017/ | head

# If empty, download with:
python scripts/download_lvis.py
```

### Conversion too slow
```bash
# Test with small sample first
python scripts/convert_lvis.py --split train --max_samples 100
```

### Want to skip validation (faster)
```bash
python scripts/validate_jsonl.py \
  lvis/processed/train.jsonl \
  --skip-image-check
```

---

## Next: Objects365 & Open Images

After LVIS works, we can add:
- **Objects365** (365 categories, balanced) - Similar conversion process
- **Open Images V7** (6000+ categories) - More complex (CSV format)

But for now, **LVIS is the priority** ✅

