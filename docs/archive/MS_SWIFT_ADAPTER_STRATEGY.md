# 🚀 ms-swift Aligner Training Guide: Full Strategy

Status: Archived — Superseded by docs/TRAINING_GUIDE.md (LoRA/prepare_model)

## Overview

This guide consolidates strategies for training the MLP aligner in Qwen3-VL, from basic full-parameter tuning to advanced techniques. It covers:
- Why `modules_to_save` is the core mechanism
- Comparison of tuning methods (adapter vs LoRA)
- Advanced alternative techniques
- Anti-overfitting strategies
- Complete configuration reference

---

## Part 1: Core Concept — `modules_to_save`

### What is `modules_to_save`?

**The Problem You're Solving:**
- Full-parameter tuning of the aligner alone isn't giving good alignment
- You want to expand aligner capacity while keeping LLM/Vision frozen
- You need maximum expressiveness without LoRA bottleneck constraints

**The Solution:**
ms-swift's `modules_to_save` mechanism allows **full-parameter training on specific modules** while other modules can be frozen or use LoRA.

### How It Works

When using `train_type: lora` with `modules_to_save`:

1. **Modules in `target_regex`**: Get LoRA adapters (lightweight, rank-constrained)
2. **Modules in `modules_to_save`**: Bypass LoRA, all weights are fully trainable
3. **Other modules**: Remain frozen

**Code Flow** (`/data/ms-swift/swift/llm/train/tuner.py` lines 115-165):
```python
# Get modules for LoRA
target_modules = get_target_modules(args, model)

# Get modules for full-parameter tuning
modules_to_save = get_modules_to_save(args, model)

# Pass both to LoRA config
lora_kwargs = {
    'target_modules': target_modules,      # ← LoRA targets
    'modules_to_save': modules_to_save,    # ← Full-param targets
    ...
}
lora_config = LoraConfig(**lora_kwargs)
model = Swift.prepare_model(model, lora_config)
```

---

## Part 2: Tuning Methods Comparison

### Method 1: Full-Parameter (modules_to_save) — RECOMMENDED for Aligner

**Best for:** Aligner-only training when full-tuning underperforms

```yaml
tuner:
  train_type: lora
  freeze_llm: true
  freeze_vit: true
  freeze_aligner: true
  target_regex: '^$'  # No LoRA targets
  modules_to_save:
    - model.visual.merger
    - model.visual.deepstack_merger_list.0
    - model.visual.deepstack_merger_list.1
    - model.visual.deepstack_merger_list.2
```

**Pros:**
- ✅ Maximum capacity (no rank constraint)
- ✅ Best for precise tasks (grounding, detection)
- ✅ Direct weight updates

**Cons:**
- More parameters → more memory
- Risk of overfitting without proper regularization

---

### Method 2: DoRA (Weight-Decomposed LoRA)

**Best for:** When full-param overfits or needs better optimization

```yaml
tuner:
  train_type: lora
  use_dora: true  # ← Enable DoRA
  freeze_llm: true
  freeze_vit: true
  freeze_aligner: true
  target_regex: '^$'
  modules_to_save: [aligner modules]
  lora_rank: 16
  lora_alpha: 32
```

**Why it helps:**
- Separates magnitude and direction components
- More expressive than standard LoRA
- Better convergence dynamics

**Paper:** "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024)

---

### Method 3: AdaLoRA (Adaptive Rank Allocation)

**Best for:** Auto-discovering optimal capacity per layer

```yaml
tuner:
  train_type: adalora
  freeze_llm: true
  freeze_vit: true
  freeze_aligner: false  # AdaLoRA targets directly
  target_regex: '^model\.visual\.(merger|deepstack_merger_list\.[0-2]).*'
  
  adalora_target_r: 16       # Final rank
  adalora_init_r: 32         # Initial rank
  adalora_tinit: 2000        # Start pruning after 2k steps
  adalora_tfinal: 20000      # Finish pruning at 20k steps
  adalora_deltaT: 200        # Update frequency
  adalora_beta1: 0.85
  adalora_beta2: 0.85
  adalora_orth_reg_weight: 0.5
```

**Why it helps:**
- Automatically finds which layers need more capacity
- Reveals bottlenecks in your aligner
- Prevents redundant parameters

**Paper:** "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning" (ICLR 2023)

---

### Method 4: BOFT (Butterfly Orthogonal Fine-Tuning)

**Best for:** Preserving geometric structure (good for grounding)

```yaml
tuner:
  train_type: boft
  freeze_llm: true
  freeze_vit: true
  freeze_aligner: false
  target_regex: '^model\.visual\.(merger|deepstack_merger_list\.[0-2]).*'
  
  boft_block_size: 4
  boft_dropout: 0.1
```

**Why it helps:**
- Orthogonal matrices preserve coordinate space
- Better for bbox/quad/line grounding
- Prevents overfitting

**Paper:** "Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization" (ICLR 2024)

---

### Method 5: FourierFT (Frequency Domain Tuning)

**Best for:** Spatial patterns in vision

```yaml
tuner:
  train_type: fourierft
  freeze_llm: true
  freeze_vit: true
  freeze_aligner: false
  target_regex: '^model\.visual\.(merger|deepstack_merger_list\.[0-2]).*'
  
  fourier_n_frequency: 2000
  fourier_scaling: 300.0
```

**Paper:** "FourierFT: Rethinking Adaptation in the Frequency Domain" (2024)

---

### Why NOT Use `train_type: adapter`?

**The Problem:**
ms-swift's `adapter` is **hardcoded to target LLM MLPs**, NOT the aligner.

**Source:** `/data/ms-swift/swift/llm/train/tuner.py` lines 253-264:
```python
elif args.train_type == 'adapter':
    mlp_key = model_arch.mlp.split('.{}.')[1]  # ← Always LLM MLP!
    adapter_config = AdapterConfig(
        target_modules=[mlp_key],  # Won't target aligner
        ...
    )
```

**Result:**
- ❌ Bottleneck MLPs added to LLM layers (not aligner)
- ❌ Aligner remains completely untouched
- ❌ Won't help with alignment issues

---

## Part 3: Training Strategies to Prevent Overfitting

### Strategy 1: Conservative Geometric + Aggressive Appearance Augmentation

**Problem:** Aggressive geometric transforms (rotate, scale) cause coordinate drift in quad/line objects.

**Solution:** Minimize geometric variation, maximize appearance variation.

```yaml
custom:
  augmentation:
    enabled: true
    ops:
      # CONSERVATIVE geometric (preserve grounding)
      - name: hflip
        params: { prob: 0.5 }
      - name: rotate
        params: { max_deg: 8.0, prob: 0.25 }  # ← Reduced from 20°
      - name: scale
        params: { lo: 1.02, hi: 1.15, prob: 0.2 }  # ← Very conservative
      - name: resize_by_scale
        params: { lo: 0.9, hi: 1.1, align_multiple: 32, prob: 0.5 }  # ← Tight range
      
      # AGGRESSIVE appearance (no geometric impact)
      - name: color_jitter
        params: { brightness: [0.6, 1.4], contrast: [0.6, 1.4], saturation: [0.6, 1.4], prob: 0.8 }
      - name: gamma
        params: { gamma: [0.6, 1.5], prob: 0.5 }
      - name: hsv
        params: { hue_delta_deg: [-25, 25], sat: [0.6, 1.5], val: [0.6, 1.5], prob: 0.6 }
      - name: clahe
        params: { clip_limit: 4.0, tile_grid_size: [8, 8], prob: 0.3 }
      - name: sharpness
        params: { factor: [0.5, 1.8], prob: 0.4 }
      - name: solarize
        params: { threshold: 128, prob: 0.15 }
      - name: posterize
        params: { bits: 4, prob: 0.15 }
      - name: pad_to_multiple
        params: { multiple: 32 }
```

**Key insight:** Appearance transforms have ZERO impact on coordinates but force robustness.

---

### Strategy 2: Anti-Overfitting Regularization

```yaml
training:
  # Strong weight decay
  weight_decay: 0.2  # Increase from 0.1
  
  # Longer warmup
  warmup_ratio: 0.3  # 30% of training
  
  # Early stopping
  load_best_model_at_end: true
  save_total_limit: 5  # Keep multiple checkpoints
```

---

### Strategy 3: Curriculum Learning (If Severe Overfitting)

**Phase 1 (30 epochs):** Simple objects only
```yaml
custom:
  train_jsonl: data/simple_1_to_3_objects.jsonl
```

**Phase 2 (30 epochs):** Medium complexity
```yaml
custom:
  train_jsonl: data/medium_complexity.jsonl
```

**Phase 3 (40 epochs):** Full dataset

---

### Strategy 4: Progressive Layer Unfreezing

Start conservative, gradually increase variation:

**Phase 1:** Minimal geometric augmentation
```yaml
- rotate: { max_deg: 5.0, prob: 0.15 }
- scale: { lo: 1.01, hi: 1.08, prob: 0.1 }
```

**Phase 2:** Moderate augmentation
```yaml
- rotate: { max_deg: 8.0, prob: 0.25 }
- scale: { lo: 1.02, hi: 1.15, prob: 0.2 }
```

**Phase 3:** Final tuning
```yaml
- rotate: { max_deg: 12.0, prob: 0.3 }
- scale: { lo: 1.03, hi: 1.2, prob: 0.25 }
```

---

## Part 4: Configuration Reference

### Critical Settings

```yaml
tuner:
  # Freeze flags: Controls which modules get LoRA
  freeze_llm: true        # LLM frozen (no updates)
  freeze_vit: true        # Vision frozen (no updates)
  freeze_aligner: true    # Aligner frozen from LoRA, but full-param via modules_to_save
  
  # Target selection
  target_regex: '^$'      # Empty = no LoRA adapters
  
  # Full-parameter training on aligner ONLY
  modules_to_save:
    - model.visual.merger
    - model.visual.deepstack_merger_list.0
    - model.visual.deepstack_merger_list.1
    - model.visual.deepstack_merger_list.2
```

### Learning Rates

```yaml
training:
  optimizer: multimodal
  
  # Only aligner_lr matters (others are frozen)
  learning_rate: 5.0e-4      # Not used (LLM frozen)
  vit_lr: 5.0e-4             # Not used (Vision frozen)
  aligner_lr: 5.0e-4         # ← ONLY active learning rate
```

**For overfitting:**
```yaml
training:
  aligner_lr: 1.0e-4         # Conservative
  weight_decay: 0.2          # Strong regularization
  warmup_ratio: 0.3          # Longer warmup
```

---

## Part 5: Usage & Troubleshooting

### Launch Training

```bash
conda run -n ms bash /data/Qwen3-VL/scripts/train.sh \
  config=/data/Qwen3-VL/configs/stage_4_aligner_adapter.yaml \
  gpus=0
```

### Monitor Metrics

Track separately:
- **bbox accuracy** - Should improve
- **quad accuracy** - If drifting, reduce geometric augmentation
- **line accuracy** - If drifting, reduce geometric augmentation

### Verify Parameters

```python
import torch
ckpt = torch.load('checkpoint-XXX/pytorch_model.bin')
for key in sorted(ckpt.keys()):
    if 'merger' in key:
        print(f"{key}: {ckpt[key].shape}")
```

---

## Part 6: Method Selection Guide

| Problem | Recommended Method | Why |
|---------|-------------------|-----|
| Full-param overfitting | Reduce LR, add regularization | Conservative approach |
| Grounding drift (quad/line) | Conservative augmentation | Geometry matters |
| Need automatic capacity allocation | AdaLoRA | Reveals bottlenecks |
| Better optimization needed | DoRA | More expressive |
| Preserve coordinate space | BOFT | Orthogonal matrices |
| Spatial pattern learning | FourierFT | Frequency domain |

---

## References

- **ms-swift LoRA**: `/data/ms-swift/swift/llm/argument/tuner_args.py` (line 113)
- **modules_to_save handling**: `/data/ms-swift/swift/tuners/base.py` (line 125-212)
- **Tuner branching**: `/data/ms-swift/swift/llm/train/tuner.py` (line 166-319)
- **DoRA**: "DoRA: Weight-Decomposed Low-Rank Adaptation" (Liu et al., 2024)
- **AdaLoRA**: "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning" (ICLR 2023)
- **BOFT**: "Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization" (ICLR 2024)
- **FourierFT**: "FourierFT: Rethinking Adaptation in the Frequency Domain" (2024)
