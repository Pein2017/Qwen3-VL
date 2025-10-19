# MS-Swift Training Guide for Qwen3-VL

## Complete Reference of Training Techniques & Hyperparameters

Based on comprehensive analysis of ms-swift library (`/data/Pein/ms-swift/`).

---

## 1. Training Types (`--train_type`)

ms-swift supports multiple training approaches:

### Available Training Types:
```python
train_type choices: [
    'lora',        # LoRA (default, recommended for efficiency)
    'full',        # Full fine-tuning (all parameters)
    'longlora',    # Long-context LoRA
    'adalora',     # Adaptive LoRA (dynamic rank)
    'llamapro',    # LLaMAPro (block expansion)
    'adapter',     # Adapter modules
    'vera',        # VeRA (Vector-based Random Matrix Adaptation)
    'boft',        # BOFT (Butterfly Orthogonal FT)
    'fourierft',   # FourierFT
    'reft',        # ReFT (Representation Fine-Tuning)
    'bone',        # Block-Affine Adaptation
]
```

**Recommendation for Qwen3-VL**: Use `lora` (default) for best efficiency/performance trade-off.

---

## 2. LoRA Configuration (`--train_type lora`)

### Core LoRA Parameters

```bash
# Basic LoRA settings
--lora_rank 8              # LoRA rank (default: 8)
                           # Higher = more capacity, slower
                           # Typical range: 4-128

--lora_alpha 32            # LoRA alpha (default: 32)
                           # Scaling factor = alpha/rank
                           # Common: 2x rank

--lora_dropout 0.05        # LoRA dropout (default: 0.05)
                           # Regularization, typical: 0.0-0.1

--target_modules           # Which modules to apply LoRA
  ['all-linear']           # Default: all linear layers
  # Examples:
  # ['q_proj', 'v_proj']          - Only attention Q/V
  # ['q_proj', 'k_proj', 'v_proj', 'o_proj']  - All attention
  # ['mlp.up_proj', 'mlp.down_proj']  - Only MLP
```

### Advanced LoRA Variants

```bash
# LoRA+ (different LR for A and B matrices)
--lorap_lr_ratio 16        # LR multiplier for B matrix (default: 16)
                           # B gets lr * lorap_lr_ratio

# RS-LoRA (Rank-Stabilized LoRA)
--use_rslora true          # Enable RS-LoRA (default: false)
                           # Better stability for higher ranks

# DoRA (Weight-Decomposed LoRA)
--use_dora true            # Enable DoRA (default: false)
                           # Decomposes into magnitude + direction

# LoRA initialization methods
--init_weights             # How to initialize LoRA weights
  'true'                   # Default initialization
  'gaussian'               # Gaussian init
  'pissa'                  # PiSSA init
  'olora'                  # OLoRA init  
  'loftq'                  # LoftQ (for quantized models)
  'lora-ga'                # Gradient-aware init
```

### LoRA-GA (Gradient-Aware Initialization)

```bash
--init_weights lora-ga
--lora_ga_batch_size 2           # Batch size for gradient estimation
--lora_ga_iters 2                # Number of gradient estimation iterations
--lora_ga_max_length 1024        # Max length for estimation
--lora_ga_direction 'ArB2r'      # Init direction: ArBr, A2rBr, ArB2r, random
--lora_ga_scale 'stable'         # Scaling: gd, unit, stable, weightS
--lora_ga_stable_gamma 16        # Gamma for stable scaling
```

---

## 3. Other Tuner-Specific Parameters

### AdaLoRA (Adaptive LoRA)
```bash
--train_type adalora
--adalora_target_r 8           # Target rank
--adalora_init_r 12            # Initial rank
--adalora_tinit 0              # Initial T
--adalora_tfinal 0             # Final T
--adalora_deltaT 1             # Delta T
--adalora_beta1 0.85           # Beta1
--adalora_beta2 0.85           # Beta2
--adalora_orth_reg_weight 0.5  # Orthogonal regularization weight
```

### Adapter
```bash
--train_type adapter
--adapter_length 128           # Adapter hidden size
--adapter_act 'gelu'           # Activation function
```

### BOFT
```bash
--train_type boft
--boft_block_size 4            # Block size
--boft_block_num 0             # Number of blocks
--boft_n_butterfly_factor 1    # Butterfly factor
--boft_dropout 0.0             # Dropout
```

### VeRA
```bash
--train_type vera
--vera_rank 256                # VeRA rank
--vera_dropout 0.0             # Dropout
--vera_d_initial 0.1           # Initial D value
```

### ReFT (Representation Fine-Tuning)
```bash
--train_type reft
--reft_layers [8, 16, 24]      # Which layers to apply
--reft_rank 4                  # Rank
--reft_intervention_type       # Intervention type
  'LoreftIntervention'         # (default)
  'NoreftIntervention'
  'ConsreftIntervention'
```

### LLaMAPro
```bash
--train_type llamapro
--llamapro_num_new_blocks 4    # Number of new blocks to add
```

---

## 4. Full Fine-Tuning (`--train_type full`)

### For Multimodal Models (Qwen3-VL)

```bash
--train_type full

# Control what to freeze/train
--freeze_llm false             # Freeze language model (default: false)
--freeze_vit true              # Freeze vision tower (default: true)
--freeze_aligner true          # Freeze vision-language aligner (default: true)

# Fine-grained control
--freeze_parameters            # List of modules to freeze
  ['model.layers.0', 'model.embed_tokens']

--freeze_parameters_regex      # Regex pattern for freezing
  'model.layers.[0-5].*'       # Freeze first 6 layers

--freeze_parameters_ratio 0.5  # Freeze ratio of parameters (0-1)

--trainable_parameters         # Force these to be trainable
  ['lm_head', 'model.norm']
```

---

## 5. GaLore (Gradient Low-Rank Projection)

Memory-efficient full fine-tuning alternative:

```bash
--use_galore true
--galore_target_modules        # Modules to apply GaLore
  ['mlp', 'self_attn']
--galore_rank 128              # Projection rank
--galore_update_proj_gap 50    # Update projection every N steps
--galore_scale 1.0             # Scaling factor
--galore_proj_type 'std'       # Projection type
```

---

## 6. Model Configuration

### Basic Model Settings

```bash
--model MODEL_PATH             # Path or ID (required)
--model_type qwen_3-vl         # Model type (auto-detected usually)
--model_revision main          # Model revision/branch

# Precision
--torch_dtype                  # Model dtype
  bfloat16                     # Recommended for A100/H100
  float16                      # For V100/T4
  float32                      # Full precision

# Attention Implementation
--attn_impl                    # Attention implementation
  flash_attn                   # Flash Attention (recommended, fastest)
  flash_attention_2            # FlashAttention-2
  flash_attention_3            # FlashAttention-3
  sdpa                         # PyTorch SDPA
  eager                        # Standard attention
```

### Device & Memory Management

```bash
--device_map auto              # Automatic device mapping
--device_map '{"": 0}'         # All on GPU 0
--device_map                   # Custom mapping
  '{"model.layers.0-15": 0, "model.layers.16-31": 1}'

--max_memory                   # Max memory per device
  '{"0": "40GB", "1": "40GB", "cpu": "100GB"}'
```

### Context Length & RoPE

```bash
--max_model_len 4096           # Maximum context length
--max_length 2048              # Max sequence length for training

# RoPE Scaling (for longer contexts)
--rope_scaling linear          # linear, dynamic, yarn
--rope_scaling                 # Custom config
  '{"type": "yarn", "factor": 2.0}'
```

---

## 7. Quantization

### AWQ (Activation-aware Weight Quantization)

```bash
--quant_method awq
--quant_bits 4                 # 4-bit quantization
--quant_n_samples 128          # Calibration samples
```

### GPTQ

```bash
--quant_method gptq
--quant_bits 4
--gptq_group_size 128
```

### BitsAndBytes (bnb)

```bash
--quantization_bit 4           # 4-bit quantization
--bnb_4bit_comp_dtype bfloat16 # Computation dtype
--bnb_4bit_quant_type nf4      # nf4 or fp4
--bnb_4bit_use_double_quant true
```

---

## 8. Training Hyperparameters

### Learning Rate & Optimization

```bash
--learning_rate 1e-4           # Learning rate
                               # LoRA default: 1e-4
                               # Full default: 1e-5

--optimizer                    # Optimizer choice
  adamw_torch                  # (default) AdamW
  adamw_8bit                   # 8-bit AdamW (memory efficient)
  adamw_bnb_8bit              # BitsAndBytes 8-bit
  adafactor                    # Adafactor (memory efficient)
  sgd                          # SGD
  lorap                        # Auto-set with lorap_lr_ratio
  galore                       # Auto-set with use_galore

--weight_decay 0.01            # Weight decay (default: 0.01)
--adam_beta1 0.9               # Adam beta1
--adam_beta2 0.999             # Adam beta2
--adam_epsilon 1e-8            # Adam epsilon
--max_grad_norm 1.0            # Gradient clipping
```

### Learning Rate Schedule

```bash
--lr_scheduler_type            # LR scheduler
  cosine                       # (default) Cosine decay
  linear                       # Linear decay
  constant                     # Constant LR
  constant_with_warmup         # Constant with warmup
  cosine_with_restarts         # Cosine with restarts
  polynomial                   # Polynomial decay

--warmup_ratio 0.05            # Warmup ratio (default: 0.05)
--warmup_steps 100             # Or specify warmup steps directly
```

### Batch Size & Gradient Accumulation

```bash
--per_device_train_batch_size 1      # Batch size per GPU
--per_device_eval_batch_size 1       # Eval batch size
--gradient_accumulation_steps 16     # Gradient accumulation
                                     # Effective batch = batch_size * accumulation * num_gpus

--auto_find_batch_size true          # Auto-find largest batch size
```

### Epochs & Steps

```bash
--num_train_epochs 3           # Number of epochs
--max_steps 10000              # Or max steps (overrides epochs)

# Evaluation & Saving
--eval_strategy steps          # steps, epoch, or no
--eval_steps 200               # Eval every N steps
--save_strategy steps          # steps, epoch
--save_steps 500               # Save every N steps
--save_total_limit 2           # Keep only N checkpoints
--load_best_model_at_end true  # Load best checkpoint at end
```

---

## 9. Data & Dataset Configuration

### Dataset Arguments

```bash
--dataset DATASET_NAME         # Dataset name (or placeholder for custom)
--dataset_num_proc 4           # Number of processes for dataset loading
--dataloader_num_workers 2     # Number of dataloader workers

# Custom dataset (what you're using)
--jsonl PATH                   # Your custom argument
--val_jsonl PATH               # Your custom argument

# Data splitting
--split_dataset_ratio 0.01     # Split ratio for validation (if no val set)
--train_dataset_mix_ratio      # Mix ratio for multiple datasets
--val_dataset_sample 1000      # Sample N examples for validation
```

### Sequence Processing

```bash
--max_length 2048              # Max sequence length
--truncation_strategy right    # left or right truncation
--check_dataset_strategy none  # none, warning, error (data validation)

# Packing (efficient sequence packing)
--packing true                 # Enable packing (requires flash_attn)
--padding_free true            # Padding-free training (requires flash_attn)
```

---

## 10. Memory Optimization Techniques

### Gradient Checkpointing

```bash
--gradient_checkpointing true  # Enable gradient checkpointing
                               # Trades compute for memory (slower but less VRAM)
```

### Mixed Precision Training

```bash
--fp16 true                    # FP16 training (auto-set from torch_dtype)
--bf16 true                    # BF16 training (recommended for A100/H100)
--fp16_full_eval false         # Use FP32 for evaluation
```

### DeepSpeed

```bash
--deepspeed zero2              # DeepSpeed ZeRO Stage 2
--deepspeed zero3              # DeepSpeed ZeRO Stage 3 (most memory efficient)
--deepspeed zero2_offload      # ZeRO-2 with CPU offload
--deepspeed zero3_offload      # ZeRO-3 with CPU offload
--deepspeed path/to/config.json # Custom DeepSpeed config

# DeepSpeed ZeRO++ settings
--zero_hpz_partition_size 8    # Partition size for ZeRO++
```

### Distributed Training

```bash
# Multi-GPU (automatic with torchrun or accelerate)
torchrun --nproc_per_node 4 -m src.sft ...

# FSDP (Fully Sharded Data Parallel)
--fsdp full_shard              # Enable FSDP
--fsdp_config path/config.json # FSDP config
```

---

## 11. Advanced Techniques

### NEFTune (Noisy Embeddings)

```bash
--neftune_noise_alpha 5        # Add noise to embeddings (regularization)
```

### LISA (Layerwise Importance Sampled AdamW)

```bash
--lisa_activated_layers 2      # Number of layers to activate
--lisa_step_interval 20        # Interval to switch layers
```

### Sequence Parallelism

```bash
--sequence_parallel_size 2     # Sequence parallelism degree
```

---

## 12. Logging & Monitoring

### Basic Logging

```bash
--logging_steps 10             # Log every N steps
--logging_first_step true      # Log first step
--logging_dir ./logs           # TensorBoard log directory

# Reporting
--report_to                    # Where to report
  tensorboard                  # TensorBoard (default)
  wandb                        # Weights & Biases
  swanlab                      # SwanLab

# WandB
--wandb_project PROJECT_NAME
--wandb_entity ENTITY_NAME

# SwanLab
--swanlab_project PROJECT_NAME
--swanlab_token TOKEN
```

### Checkpointing

```bash
--output_dir ./output          # Output directory
--save_only_model true         # Save only model (not optimizer states)
--push_to_hub false            # Push to HuggingFace Hub
```

---

## 13. Template & Prompt Configuration

### Template Settings

```bash
--template_type qwen           # Template type for conversation
--system_prompt "You are..."   # System prompt
--tools_prompt "..."           # Tools prompt (for function calling)
```

---

## 14. Qwen3-VL Specific Recommendations

### Recommended Configuration for Dense Caption Training

```bash
# Basic setup
python -m src.sft \
  --model /path/to/Qwen3-VL-4B-Instruct \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  \
  # LoRA settings
  --train_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  \
  # Training hyperparameters
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  \
  # Memory optimization
  --gradient_checkpointing true \
  --max_length 2048 \
  --truncation_strategy right \
  --padding_free true \
  \
  # Evaluation & saving
  --eval_strategy steps \
  --eval_steps 200 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 2 \
  \
  # Dataset (your custom)
  --jsonl data/train.jsonl \
  --val_jsonl data/val.jsonl \
  --dataset_num_proc 4 \
  --dataloader_num_workers 2 \
  \
  # Output
  --output_dir ./output/qwen3vl_lora \
  --logging_steps 10
```

### For Better Quality (Higher Rank LoRA)

```bash
--lora_rank 32 \
--lora_alpha 64 \
--target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]'
```

### For Memory Constrained (Lower Resources)

```bash
--lora_rank 4 \
--lora_alpha 8 \
--gradient_accumulation_steps 32 \
--max_length 1024 \
--gradient_checkpointing true \
--deepspeed zero2_offload
```

### For Maximum Performance (Multiple GPUs)

```bash
torchrun --nproc_per_node 4 -m src.sft \
  --lora_rank 16 \
  --lora_alpha 32 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --deepspeed zero2
```

---

## 15. Complete Command-Line Reference

### All Major Flags Summary

```
Model & Hardware:
  --model, --model_type, --torch_dtype, --attn_impl
  --device_map, --max_memory, --quantization_bit

Training Type & Tuner:
  --train_type {lora|full|adalora|...}
  --lora_rank, --lora_alpha, --lora_dropout
  --target_modules, --modules_to_save

Hyperparameters:
  --learning_rate, --optimizer, --weight_decay
  --lr_scheduler_type, --warmup_ratio
  --num_train_epochs, --max_steps
  --per_device_train_batch_size, --gradient_accumulation_steps

Data:
  --dataset, --max_length, --truncation_strategy
  --packing, --padding_free
  --dataset_num_proc, --dataloader_num_workers

Memory & Optimization:
  --gradient_checkpointing, --fp16, --bf16
  --deepspeed, --fsdp
  --use_galore, --neftune_noise_alpha

Evaluation & Logging:
  --eval_strategy, --eval_steps
  --save_strategy, --save_steps, --save_total_limit
  --logging_steps, --report_to
  --output_dir
```

---

## 16. Tuning Tips & Best Practices

### LoRA Rank Selection
- **Small models (<7B)**: rank 4-8
- **Medium models (7B-13B)**: rank 8-16
- **Large models (>13B)**: rank 16-32
- **Vision-Language (Qwen3-VL)**: rank 8-16 (balance vision/language)

### Learning Rate Guidelines
- **LoRA**: 1e-4 to 5e-4
- **Full fine-tuning**: 1e-5 to 5e-5
- **With warmup**: Use 5-10% warmup_ratio

### Batch Size Strategy
- Start with `batch_size=1`, increase `gradient_accumulation_steps`
- Effective batch size = `batch * accumulation * num_gpus`
- Target effective batch: 32-128 for most tasks

### Memory Optimization Priority
1. Enable `gradient_checkpointing`
2. Use `flash_attn` + `padding_free`
3. Lower `max_length` if possible
4. Use `deepspeed zero2` or `zero3`
5. Reduce `lora_rank` if still OOM

### For Vision-Language Models (Qwen3-VL)
- Keep `--freeze_vit true` (vision encoder frozen)
- Keep `--freeze_aligner true` initially
- Focus LoRA on language model layers
- Use `--padding_free true` with flash attention for efficiency

---

## Quick Reference Card

```bash
# Minimal LoRA training
python -m src.sft \
  --model MODEL_PATH \
  --jsonl data/train.jsonl \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --train_type lora \
  --output_dir output

# Production LoRA training
python -m src.sft \
  --model MODEL_PATH \
  --jsonl data/train.jsonl \
  --val_jsonl data/val.jsonl \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --train_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing true \
  --padding_free true \
  --max_length 2048 \
  --warmup_ratio 0.05 \
  --eval_steps 200 \
  --save_steps 500 \
  --logging_steps 10 \
  --output_dir output/my_model
```

---

**Generated**: 2025-10-17  
**Source**: ms-swift library analysis at `/data/Pein/ms-swift/`

