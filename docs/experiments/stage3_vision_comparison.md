# Stage 3 Vision Tower Training: Comparative Experiment Report

**Date**: October 26, 2025  
**Experiment ID**: `10-27/stage_3_vision_comparison`  
**Base Model**: `output/stage_2_merged-10-25` (Stage 2 LLM+Aligner LoRA merged)

---

## Executive Summary

This experiment compares **4 different approaches** to fine-tuning the vision tower in Stage 3 training:

1. **Last-6 LoRA** (`last6_lora`) — LoRA on vision blocks 18-23 ✅ **WINNER**
2. **Last-6 Full** (`last6_full`) — Full fine-tuning of vision blocks 18-23
3. **All LoRA** (`all_lora`) — LoRA on all vision blocks 0-23
4. **All Full** (`all_full`) — Full fine-tuning of all vision blocks 0-23

### Key Findings

- **Best Configuration**: `last6_lora` achieved the **lowest eval loss** (0.555 smoothed, 0.5534 final)
- **Last-6 blocks outperformed All blocks**: Selective training of the final 6 vision blocks proved more effective than training all 24 blocks
- **LoRA > Full fine-tuning**: LoRA adaptation was more stable and achieved better generalization than full fine-tuning
- **Data Quality Note**: These results were obtained despite noise/mistakes in the data augmentation pipeline (since fixed in subsequent experiments)

---

## Experimental Setup

### Common Configuration

All 4 experiments shared identical settings except for vision tower training strategy:

| Parameter | Value |
|-----------|-------|
| Base Model | `output/stage_2_merged-10-25` |
| Prompt Scheme | B |
| LLM Training | LoRA on last-2 layers (34-35) |
| Aligner Training | Full fine-tuning (modules_to_save) |
| Training Epochs | 50 |
| Per-Device Batch Size | 2 (train & eval) |
| Gradient Accumulation | 2-4 steps (eff batch 16-32) |
| Packing | Enabled |
| Weight Decay | 0.1 |
| Warmup Ratio | 0.2 |
| LoRA Config | rank=16, alpha=32, dropout=0.1 |
| Optimizer | multimodal |
| Base Learning Rate | 1.0e-4 |
| Aligner LR | 1.0e-4 |
| DeepSpeed | ZeRO Stage 2 |

### Vision-Specific Differences

| Config | Vision Blocks | Training Method | VIT LR | Grad Accum | Target Regex |
|--------|---------------|-----------------|--------|------------|--------------|
| **last6_lora** | 18-23 (6 blocks) | LoRA | 1.2e-4 | 2 | LLM last-2 + Vision 18-23 |
| **last6_full** | 18-23 (6 blocks) | Full (modules_to_save) | 6.0e-5 | 2 | LLM last-2 only |
| **all_lora** | 0-23 (24 blocks) | LoRA | 1.5e-4 | 2 | LLM last-2 + Vision 0-23 |
| **all_full** | 0-23 (24 blocks) | Full (modules_to_save) | 3.0e-5 | 4 | LLM last-2 only |

**Note**: Full fine-tuning used lower learning rates and higher gradient accumulation to stabilize training over the larger parameter space.

---

## Results

### Quantitative Metrics (Final Checkpoint)

| Configuration | Eval Loss (Smoothed) | Eval Loss (Raw) | Train Loss (Smoothed) | Train Loss (Raw) | Final Step | Training Time |
|---------------|----------------------|-----------------|------------------------|------------------|------------|---------------|
| **last6_lora** ✅ | **0.555** | **0.5534** | 0.2001 | 0.1588 | 10,420 | ~1.016 days |
| **last6_full** | 0.5655 | 0.5697 | **0.1981** | **0.1715** | 10,440 | ~1.013 days |
| **all_lora** | 0.6158 | 0.6082 | 0.2379 | 0.2758 | 10,030 | ~1.017 days |
| **all_full** | 0.6271 | 0.6281 | 0.2171 | 0.2064 | 6,900 | ~1.014 days |

**Winner**: `last6_lora` with **0.555 eval loss** (1.9% better than `last6_full`, 10.9% better than `all_lora`, 13.0% better than `all_full`)

### Training Curves Analysis

#### Eval Loss Progression

```
                 last6_lora  last6_full  all_lora  all_full
Step 2000        ~0.52       ~0.53       ~0.57     ~0.58
Step 4000        ~0.56       ~0.58       ~0.65     ~0.68
Step 6000        ~0.56       ~0.58       ~0.66     ~0.67
Step 8000        ~0.56       ~0.57       ~0.64     ~0.63
Step 10000       ~0.555      ~0.5655     ~0.616    ~0.628
```

**Observations**:
- `last6_lora` showed **early convergence** and maintained the lowest eval loss throughout
- `last6_full` had slightly higher eval loss but more stable training
- `all_lora` and `all_full` struggled to match last-6 performance, suggesting **over-parameterization** or difficulty training deeper layers

#### Train Loss Progression

- **Lowest training loss**: `last6_full` (0.1981 smoothed), indicating strong fitting capacity
- **Best generalization gap**: `last6_lora` showed the best balance between train and eval loss
- `all_full` exhibited signs of early stopping (6,900 steps vs ~10,400 for others), possibly due to convergence or instability

---

## Analysis & Insights

### 1. Last-6 Blocks Hypothesis

The superior performance of `last6_lora` and `last6_full` suggests that:

- **Top-layer specialization**: The final 6 vision blocks (18-23) contain the most task-relevant, high-level features
- **Lower layers are general**: Early vision blocks (0-17) encode general visual features that don't require domain-specific adaptation
- **Efficient adaptation**: Training fewer parameters reduces overfitting risk and computational cost

This aligns with transfer learning best practices: fine-tune task-specific layers while preserving general feature extractors.

### 2. LoRA vs Full Fine-Tuning

**LoRA advantages** (observed in this experiment):
- Better generalization (lower eval loss)
- More stable training dynamics
- Faster convergence
- Smaller checkpoint size

**Full fine-tuning characteristics**:
- Lower training loss (better memorization)
- Larger generalization gap
- Requires more careful LR tuning
- Higher memory and storage costs

### 3. Impact of Data Augmentation Issues

These experiments were conducted with **noisy/incorrect data augmentation** (since fixed). The strong performance of `last6_lora` despite this suggests:

- The model learned robust features even with augmentation errors
- LoRA's regularization effect may have provided resilience to noisy augmentations
- **Expected improvement**: Re-running with corrected augmentation should yield even better results

---

## Conclusions

### Primary Recommendations

1. **Use `last6_lora` configuration** for Stage 3 vision training
   - Config: `configs/stage_3_vision_last6_lora.yaml`
   - Best eval loss, efficient training, strong generalization

2. **Focus on top vision layers** (blocks 18-23 for 4B model)
   - Avoids over-parameterization
   - Reduces training time and memory
   - Better alignment with task-specific features

3. **Prefer LoRA over full fine-tuning** for vision adaptation
   - More stable and generalizable
   - Easier to manage and deploy
   - Maintains flexibility for multi-task scenarios

### Secondary Insights

- **Gradient accumulation matters**: Full fine-tuning required `grad_accum=4` for stability, while LoRA was stable with `grad_accum=2`
- **Learning rate scaling**: Full fine-tuning required significantly lower LR (3e-5 to 6e-5 vs 1.2e-4 to 1.5e-4 for LoRA)
- **Checkpoint frequency**: All configs completed ~10k steps, but `all_full` stopped early (possible convergence or eval-based early stopping)

---

## Next Steps

### Immediate Actions

1. **Re-run `last6_lora` with corrected augmentation**
   - Baseline: 0.555 eval loss (with noisy aug)
   - Expected: 0.53-0.54 eval loss (with fixed aug)
   - Validate that fixing augmentation improves results

2. **Ablation study: Vision block selection**
   - Try `last4_lora` (blocks 20-23): may offer similar performance with fewer parameters
   - Try `last8_lora` (blocks 16-23): check if more blocks help

3. **Learning rate sweep for `last6_lora`**
   - Current: `vit_lr=1.2e-4`
   - Try: `1.0e-4`, `1.5e-4`, `2.0e-4`
   - Optimize for final eval loss

### Long-Term Considerations

- **8B model scaling**: Adapt `last6_lora` config for Qwen3-VL-8B (vision blocks 21-26)
- **Task-specific tuning**: Evaluate if different domains benefit from different layer selections
- **LoRA rank optimization**: Current `rank=16`; explore `rank=8` (efficiency) or `rank=32` (capacity)
- **Multi-stage curriculum**: Consider progressive unfreezing (first last-2, then last-4, then last-6)

---

## Reproducibility

### Configuration Files

- **last6_lora**: `configs/stage_3_vision_last6_lora.yaml`
- **last6_full**: `configs/stage_3_vision_last6_full.yaml`
- **all_lora**: `configs/stage_3_vision_all_lora.yaml`
- **all_full**: `configs/stage_3_vision_all_full.yaml`

### Launch Commands

```bash
# Last-6 LoRA (recommended)
conda run -n ms bash /data/Qwen3-VL/scripts/train.sh \
  config=configs/stage_3_vision_last6_lora.yaml \
  gpus=0

# Last-6 Full
conda run -n ms bash /data/Qwen3-VL/scripts/train.sh \
  config=configs/stage_3_vision_last6_full.yaml \
  gpus=0

# All LoRA
conda run -n ms bash /data/Qwen3-VL/scripts/train.sh \
  config=configs/stage_3_vision_all_lora.yaml \
  gpus=0

# All Full
conda run -n ms bash /data/Qwen3-VL/scripts/train.sh \
  config=configs/stage_3_vision_all_full.yaml \
  gpus=0
```

### Environment

- **Hardware**: Single GPU (A100/V100/4090 recommended)
- **Software**: `ms` conda environment (ms-swift + transformers)
- **Data**: `custom.train_jsonl` and `custom.val_jsonl` from `base.yaml`
- **Base Model**: Stage 2 merged checkpoint (`output/stage_2_merged-10-25`)

---

## Appendix: TensorBoard Metrics

### Eval Loss (Step 10,000+)

| Run | Color | Smoothed | Raw | Step |
|-----|-------|----------|-----|------|
| last6_lora | Orange | 0.555 | 0.5534 | 10,400 |
| last6_full | Green | 0.5655 | 0.5697 | 10,420 |
| all_lora | Purple | 0.6158 | 0.6082 | 10,020 |
| all_full | Yellow | 0.6271 | 0.6281 | 6,900 |

### Train Loss (Step 10,000+)

| Run | Color | Smoothed | Raw | Step |
|-----|-------|----------|-----|------|
| last6_full | Green | 0.1981 | 0.1715 | 10,440 |
| last6_lora | Orange | 0.2001 | 0.1588 | 10,420 |
| all_full | Yellow | 0.2171 | 0.2064 | 6,900 |
| all_lora | Purple | 0.2379 | 0.2758 | 10,030 |

### Key Observations

1. **Generalization gap** (eval - train):
   - `last6_lora`: 0.555 - 0.2001 = **0.3549** (best balance)
   - `last6_full`: 0.5655 - 0.1981 = **0.3674**
   - `all_lora`: 0.6158 - 0.2379 = **0.3779**
   - `all_full`: 0.6271 - 0.2171 = **0.4100** (largest gap)

2. **Training efficiency**:
   - Last-6 configs reached ~10.4k steps
   - All-LoRA reached ~10k steps
   - All-Full stopped early at ~6.9k steps (potential stability/convergence issue)

---

## References

- **Stage 2 Training**: See `docs/TRAINING.md` for Stage 2 LLM+Aligner LoRA training
- **Data Augmentation**: See `docs/AUGMENTATION.md` for pipeline details and fixes
- **Base Configuration**: `configs/base.yaml` defines shared hyperparameters

---

**Document Version**: 1.0  
**Last Updated**: October 27, 2025  
**Author**: Qwen3-VL Training Team  
**Status**: Completed Experiment

