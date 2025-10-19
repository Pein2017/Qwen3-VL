# Qwen3VL Deep Dive: Complete Debugging Guide

A comprehensive guide for depth-first exploration of Qwen3VL model internals.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Execution Flow Overview](#execution-flow-overview)
3. [Depth-First Trace](#depth-first-trace)
4. [Key Breakpoints](#key-breakpoints)
5. [Data Flow & Architecture](#data-flow--architecture)
6. [Tensor Shapes Reference](#tensor-shapes-reference)
7. [Common Debugging Tasks](#common-debugging-tasks)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Run Enhanced Demo
```bash
python demo/demo_with_debugging.py
```

### Using Debug Helpers
```python
from debug_helpers import quick_inspect, ForwardHookManager, monitor_memory

# Inspect any tensor
quick_inspect("my_tensor", tensor)

# Hook into model layers
hook_manager = ForwardHookManager()
hook_manager.register_module_hooks(model, ['visual.blocks.0', 'lm_head'])
# ... run model ...
hook_manager.print_summary()
```

---

## Execution Flow Overview

Your `demo.py` follows this pipeline:

```
1. Load Processor (AutoProcessor)
   ├─> Qwen3VLProcessor (with image/video processors + tokenizer)
   └─> Special tokens: <|image_pad|>, <|vision_start|>, <|vision_end|>

2. Load Model (Qwen3VLForConditionalGeneration)
   ├─> Visual encoder (Qwen3VLVisionModel)
   ├─> Language model (Qwen3Model)
   └─> LM head (Linear projection to vocab)

3. Process Inputs (processor.__call__)
   ├─> Images → pixel_values + image_grid_thw
   ├─> Text → input_ids + attention_mask
   └─> Calculate image tokens, replace placeholders

4. Generate (model.generate)
   └─> Calls model.forward() in autoregressive loop

5. Decode (processor.batch_decode)
   └─> Token IDs → Text
```

---

## Depth-First Trace

### Phase 1: Processor Initialization

**Entry:** `AutoProcessor.from_pretrained(checkpoint_path)`

```python
Qwen3VLProcessor.__init__()  # processing_qwen3_vl.py:83
├─> image_processor (Qwen2VLImageProcessor)
├─> video_processor (Qwen3VLVideoProcessor)
├─> tokenizer (Qwen2TokenizerFast)
└─> Special token IDs:
    ├─> image_token_id
    ├─> video_token_id
    ├─> vision_start_token_id
    └─> vision_end_token_id
```

**🎯 Breakpoint #1:** `processing_qwen3_vl.py:83`  
**Inspect:** Special token IDs, processor configuration

---

### Phase 2: Model Architecture

**Entry:** `Qwen3VLForConditionalGeneration.from_pretrained()`

```python
Qwen3VLForConditionalGeneration.__init__()  # modeling_qwen3_vl.py:1278
├─> self.model: Qwen3VLModel
│   ├─> self.visual: Qwen3VLVisionModel (~line 703)
│   │   ├─> patch_embed: Qwen3VLVisionPatchEmbed (Conv3d)
│   │   ├─> rotary_emb: Qwen3VLVisionRotaryEmbedding
│   │   ├─> blocks: ModuleList[Qwen3VLVisionBlock]
│   │   │   └─> Each block:
│   │   │       ├─> norm1 + self_attn (Qwen3VLVisionAttention)
│   │   │       └─> norm2 + mlp (Qwen3VLVisionMLP)
│   │   └─> merger: Qwen3VLVisionPatchMerger
│   │
│   └─> self.language_model: Qwen3Model
│       ├─> embed_tokens: Embedding
│       ├─> layers: ModuleList[Qwen3DecoderLayer]
│       │   └─> Each layer:
│       │       ├─> input_layernorm + self_attn
│       │       └─> post_attention_layernorm + mlp
│       └─> norm: RMSNorm
│
└─> self.lm_head: Linear(hidden_size, vocab_size)
```

**🎯 Breakpoint #2:** `modeling_qwen3_vl.py:1278`  
**Inspect:** Model structure, config parameters

---

### Phase 3: Input Processing

**Entry:** `processor(images=images, text=text)`

```python
Qwen3VLProcessor.__call__()  # processing_qwen3_vl.py:114

1. Process Images
   └─> image_processor(images) → {pixel_values, image_grid_thw}
       ├─> Smart resize (based on min/max pixels)
       ├─> Patchify into 3D grids
       └─> Normalize

2. Calculate Image Tokens  # ~line 186-194
   ├─> For each image: num_tokens = grid_t * grid_h * grid_w / merge_size²
   └─> Replace "<|image_pad|>" with calculated tokens

3. Tokenize Text
   └─> tokenizer(modified_text) → {input_ids, attention_mask}
```

**🎯 Breakpoint #3:** `processing_qwen3_vl.py:186` (token replacement)  
**Inspect:** `image_grid_thw`, `merge_length`, modified text

---

### Phase 4: Model Forward Pass

**Entry:** `model.generate()` calls `model.forward()` repeatedly

```python
Qwen3VLForConditionalGeneration.forward()  # modeling_qwen3_vl.py:1315
└─> Qwen3VLModel.forward()  # ~line 1108

    ┌─────────────────────────────────────┐
    │  A. VISION ENCODING                 │
    └─────────────────────────────────────┘
    
    if pixel_values is not None:
        vision_features = self.visual(pixel_values, image_grid_thw)
        
        Qwen3VLVisionModel.forward()  # ~line 703
        ├─> 1. Patch Embedding
        │   └─> patch_embed(pixel_values)  # line 70
        │       ├─> Reshape: [B,C,T,H,W] → [B,C,temp_patch,patch,patch]
        │       └─> Conv3d → [total_patches, embed_dim]
        │
        ├─> 2. Rotary Embeddings
        │   └─> rotary_emb(seqlen) → cos, sin  # line 87
        │
        ├─> 3. Vision Transformer Blocks
        │   └─> For each block in self.blocks:  # ~line 416
        │       ├─> x = norm1(x)
        │       ├─> attn_out = self_attn(x, rotary_emb)  # line 182
        │       │   ├─> qkv = self.qkv(x) → Q, K, V
        │       │   ├─> apply_rotary_pos_emb(Q, K, cos, sin)  # line 116
        │       │   ├─> attention = softmax(Q @ K.T / √d) @ V
        │       │   └─> proj(attention)
        │       ├─> x = x + attn_out  # residual
        │       ├─> x = norm2(x)
        │       ├─> mlp_out = mlp(x)  # line 55
        │       │   └─> fc2(act_fn(fc1(x)))
        │       └─> x = x + mlp_out  # residual
        │
        └─> 4. Patch Merger
            └─> merger(x)  # line 103
                ├─> norm(x)
                ├─> Reshape: merge spatial patches
                └─> fc2(act_fn(fc1(x))) → [merged_patches, text_hidden_size]

    ┌─────────────────────────────────────┐
    │  B. VISION-TEXT FUSION ⭐ KEY STEP  │
    └─────────────────────────────────────┘
    
    ├─> text_embeddings = self.language_model.embed_tokens(input_ids)
    │   └─> Shape: [B, seq_len, hidden_size]
    │
    ├─> Find vision token positions
    │   ├─> image_mask = (input_ids == image_token_id)
    │   └─> video_mask = (input_ids == video_token_id)
    │
    └─> Replace vision tokens with vision features
        ├─> inputs_embeds = text_embeddings.clone()
        ├─> inputs_embeds[vision_positions] = vision_features
        └─> Result: [B, seq_len, hidden_size] with vision features injected

    ┌─────────────────────────────────────┐
    │  C. LANGUAGE MODEL                  │
    └─────────────────────────────────────┘
    
    hidden_states = self.language_model(inputs_embeds, attention_mask, ...)
    
    For each decoder layer:
    ├─> x = input_layernorm(x)
    ├─> attn_out = self_attn(x)  # Causal attention with RoPE
    │   ├─> Q, K, V projections
    │   ├─> Apply RoPE to Q, K
    │   ├─> Causal masked attention
    │   └─> Output projection
    ├─> x = x + attn_out
    ├─> x = post_attention_layernorm(x)
    ├─> mlp_out = mlp(x)  # SwiGLU: down(act(gate) * up)
    └─> x = x + mlp_out
    
    └─> final_hidden = norm(hidden_states)

    ┌─────────────────────────────────────┐
    │  D. OUTPUT PROJECTION               │
    └─────────────────────────────────────┘
    
    logits = self.lm_head(final_hidden)
    └─> Linear: [B, seq_len, hidden_size] → [B, seq_len, vocab_size]
```

---

## Key Breakpoints

### Top 13 Strategic Breakpoints

| Priority | File | Line/Location | Purpose |
|----------|------|---------------|---------|
| ⭐⭐⭐ | `modeling_qwen3_vl.py` | ~1315 | Main forward entry |
| ⭐⭐⭐ | `modeling_qwen3_vl.py` | ~1108 | Vision+Text fusion |
| ⭐⭐⭐ | `modeling_qwen3_vl.py` | ~703 | Vision encoder |
| ⭐⭐ | `processing_qwen3_vl.py` | ~114 | Input processing |
| ⭐⭐ | `modeling_qwen3_vl.py` | ~70 | Patch embedding |
| ⭐⭐ | `modeling_qwen3_vl.py` | ~182 | Vision attention |
| ⭐⭐ | `modeling_qwen3_vl.py` | ~103 | Patch merger |
| ⭐⭐⭐ | `modeling_qwen3_vl.py` | Fusion section | Embedding replacement |
| ⭐ | `modeling_qwen3_vl.py` | ~116 | RoPE application |
| ⭐ | `processing_qwen3_vl.py` | ~186 | Token calculation |
| ⭐ | Language decoder | First layer | Text generation |
| ⭐ | `modeling_qwen3_vl.py` | Before lm_head | Pre-vocabulary |
| ⭐ | `processing_qwen3_vl.py` | batch_decode | Output decoding |

### Setting Breakpoints

**In VS Code/PyCharm:**
1. Open the transformers file
2. Click left gutter to set breakpoint
3. Configure launch.json with `"justMyCode": false`

**Or add inline:**
```python
# In modeling_qwen3_vl.py at key locations:
import pdb; pdb.set_trace()
```

---

## Data Flow & Architecture

### Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: Images (PIL) + Text (str)                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
  image_processor                    tokenizer
        ↓                                   ↓
  pixel_values                      input_ids
  [B,C,T,H,W]                      [B,seq_len]
  image_grid_thw
  [num_imgs,3]
        ↓
┌──────────────────────────────────────────────────────────────┐
│ VISION ENCODER                                               │
│  pixel_values → Conv3d → [patches, embed_dim]               │
│               → Vision Transformer Blocks                    │
│               → Patch Merger                                 │
│               → vision_features [merged_patches, hidden_dim] │
└──────────────────────────────────────────────────────────────┘
        ↓
        │            ┌─────────────────────────────────────┐
        │            │ TEXT EMBEDDINGS                      │
        │            │ embed_tokens(input_ids)              │
        │            │ → [B, seq_len, hidden_size]          │
        │            └─────────────────────────────────────┘
        │                           ↓
        └──────────────► FUSION ◄──┘
                          ↓
        inputs_embeds [B, seq_len, hidden_size]
        (vision tokens replaced with vision features)
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ LANGUAGE MODEL (Transformer Decoder)                         │
│  → Decoder Layers (causal attention + MLP)                   │
│  → hidden_states [B, seq_len, hidden_size]                   │
└──────────────────────────────────────────────────────────────┘
                          ↓
                       lm_head
                          ↓
                  logits [B, seq_len, vocab_size]
                          ↓
                   Sample next token
                          ↓
                    Generated text
```

### Module Hierarchy

```
Qwen3VLForConditionalGeneration
├── model: Qwen3VLModel
│   ├── visual: Qwen3VLVisionModel
│   │   ├── patch_embed: Conv3d(3, embed_dim, kernel=(2,14,14))
│   │   ├── rotary_emb: VisionRotaryEmbedding
│   │   ├── blocks[N]: VisionBlock
│   │   │   ├── norm1: LayerNorm
│   │   │   ├── self_attn: VisionAttention (qkv, proj)
│   │   │   ├── norm2: LayerNorm
│   │   │   └── mlp: VisionMLP (fc1, fc2, act)
│   │   └── merger: PatchMerger (norm, fc1, fc2)
│   │
│   └── language_model: Qwen3Model
│       ├── embed_tokens: Embedding(vocab_size, hidden_size)
│       ├── layers[N]: Qwen3DecoderLayer
│       │   ├── input_layernorm: RMSNorm
│       │   ├── self_attn: (q_proj, k_proj, v_proj, o_proj)
│       │   ├── post_attention_layernorm: RMSNorm
│       │   └── mlp: (gate_proj, up_proj, down_proj)
│       └── norm: RMSNorm
│
└── lm_head: Linear(hidden_size, vocab_size)
```

---

## Tensor Shapes Reference

### Expected Shapes (Qwen3-VL-4B, 2 images)

| Stage | Tensor | Shape | Notes |
|-------|--------|-------|-------|
| **Input** | `input_ids` | `[1, ~600]` | Text + image placeholders |
| | `pixel_values` | `[2, 3, T, H, W]` | T,H,W vary by image size |
| | `image_grid_thw` | `[2, 3]` | Grid dimensions per image |
| **Vision** | patch_embeddings | `[total_patches, 2048]` | After Conv3d |
| | after blocks | `[total_patches, 2048]` | After transformers |
| | vision_features | `[merged_patches, 4096]` | After merger |
| **Fusion** | text_embeddings | `[1, seq_len, 4096]` | From embed_tokens |
| | inputs_embeds | `[1, seq_len, 4096]` | After fusion |
| **Language** | hidden_states | `[1, seq_len, 4096]` | Per layer |
| **Output** | logits | `[1, seq_len, 151936]` | Vocab projection |
| | generated_ids | `[1, total_len]` | Final sequence |

**Key Numbers for Qwen3-VL-4B:**
- Vision hidden_size: 2048
- Text hidden_size: 4096
- Vocab size: 151936
- Default patch_size: 14
- Default temporal_patch_size: 2
- Default merge_size: 2

---

## Common Debugging Tasks

### Task 1: Trace Image to Patches

**Goal:** See how images become patch embeddings

**Key Locations:**
1. `processing_qwen3_vl.py:162` - Image processor call
2. `modeling_qwen3_vl.py:70` - Patch embed forward

**What to Inspect:**
```python
# Before patch_embed
print(f"Pixel values: {pixel_values.shape}")  # [B,3,T,H,W]

# After patch_embed
print(f"Patches: {patches.shape}")  # [total_patches, embed_dim]
print(f"Total patches: {total_patches}")
```

### Task 2: Find Vision-Text Fusion

**Goal:** See where vision features replace text tokens

**Key Location:** `modeling_qwen3_vl.py` in `Qwen3VLModel.forward()`

**What to Look For:**
```python
# 1. Get embeddings
text_embeddings = self.language_model.embed_tokens(input_ids)

# 2. Find vision token positions
image_mask = (input_ids == image_token_id)

# 3. Replace
inputs_embeds[image_mask] = vision_features

# Verify:
print(f"Num vision tokens: {image_mask.sum()}")
print(f"Num vision features: {vision_features.shape[0]}")
# These should match!
```

### Task 3: Inspect Attention Mechanism

**Goal:** Understand how attention works

**Key Location:** `modeling_qwen3_vl.py:182` (vision) or decoder layer (text)

**What to Inspect:**
```python
# In attention forward:
print(f"Q: {query.shape}")  # [seq_len, num_heads, head_dim]
print(f"K: {key.shape}")
print(f"V: {value.shape}")

# After RoPE:
query, key = apply_rotary_pos_emb(query, key, cos, sin)

# Attention weights:
attn_weights = (query @ key.T) * scaling
print(f"Attention weights: {attn_weights.shape}")
```

### Task 4: Track Token Generation

**Goal:** See autoregressive generation

**Key:** Generation loop in `model.generate()`

**What to Track:**
```python
# Each iteration:
logits = model.forward(...)[:, -1, :]  # Last position
next_token = torch.argmax(logits, dim=-1)
print(f"Next token: {next_token.item()} -> '{tokenizer.decode([next_token])}'")
```

### Task 5: Check for NaN/Inf

**Goal:** Detect numerical issues

**Use Debug Helpers:**
```python
from debug_helpers import TensorInspector

stats = TensorInspector.inspect("suspicious", tensor, detailed=True)
if stats.get('has_nan') or stats.get('has_inf'):
    print("⚠️ Numerical issue detected!")
```

---

## Troubleshooting

### Shape Mismatch in Fusion

**Symptom:** RuntimeError during forward pass

**Check:**
```python
# Vision features must match text hidden size
assert vision_features.shape[-1] == text_hidden_size
assert vision_features.shape[0] == num_vision_tokens
```

**Debug:**
```python
print(f"Vision feat shape: {vision_features.shape}")
print(f"Text embed shape: {text_embeddings.shape}")
print(f"Expected vision tokens: {image_grid_thw.prod() // (merge_size**2)}")
```

### Out of Memory

**Solutions:**
- Reduce `max_new_tokens`
- Use smaller images
- Process one image at a time
- `torch.cuda.empty_cache()`

**Monitor:**
```python
from debug_helpers import monitor_memory
monitor_memory()
```

### Can't Set Breakpoints in Library

**Problem:** Debugger skips transformers code

**Solution:** In `.vscode/launch.json`:
```json
{
    "justMyCode": false  // Critical!
}
```

### Lost in Deep Call Stack

**Solution:**
- Use Call Stack panel in debugger
- Set breakpoint at known location
- Use "Step Out" (Shift+F11)
- Refer to call graphs above

---

## Key Concepts

### 1. Vision Encoding
Images → Patches → Transformer → Features
- Conv3d splits into 3D patches
- Vision transformer processes with self-attention
- Patch merger combines spatial info

### 2. Multimodal Fusion ⭐ MOST IMPORTANT
- Text tokens → embeddings via `embed_tokens`
- Vision placeholders → replaced with vision features
- Result: unified embedding sequence
- This is the key innovation!

### 3. Rotary Position Embeddings (RoPE)
- Encodes relative positions via rotation
- Applied to Q and K in attention
- Better than absolute position encoding

### 4. Autoregressive Generation
- Generate one token at a time
- Each token conditions on all previous
- Uses KV cache for efficiency

### 5. Causal Attention
- Tokens can only attend to past (left)
- Implemented via attention mask
- Crucial for autoregressive generation

---

## Special Tokens

| Token | Purpose | Default |
|-------|---------|---------|
| `image_token` | Image placeholder | `<|image_pad|>` |
| `video_token` | Video placeholder | `<|video_pad|>` |
| `vision_start_token` | Vision start | `<|vision_start|>` |
| `vision_end_token` | Vision end | `<|vision_end|>` |

Access via: `processor.image_token_id`, etc.

---

## Learning Path

### Beginner (Week 1-4)
1. Run `demo_with_debugging.py`
2. Set breakpoint at main forward (line 1315)
3. Step through vision encoder
4. **Focus:** Understand fusion mechanism
5. Trace one generation step

### Intermediate (Week 5-8)
1. Deep dive into attention
2. Study RoPE implementation
3. Trace complete generation loop
4. Experiment with hooks
5. Profile performance

### Advanced (Ongoing)
1. Modify architecture
2. Implement custom attention
3. Optimize inference
4. Fine-tune model
5. Export to production formats

---

## Quick Reference Commands

### Find Functions
```bash
grep -n "class Qwen3VL" modeling_qwen3_vl.py
grep -n "def forward" modeling_qwen3_vl.py
```

### Debug Helpers Usage
```python
from debug_helpers import (
    quick_inspect,
    print_shapes,
    monitor_memory,
    ForwardHookManager,
    TensorInspector,
    ModelArchitectureInspector
)

# Inspect tensor
quick_inspect("name", tensor)

# Print all shapes
print_shapes(inputs, "Input Shapes")

# Monitor GPU
monitor_memory()

# Hook into layers
hooks = ForwardHookManager()
hooks.register_module_hooks(model, ['visual', 'lm_head'])
# ... run model ...
hooks.print_summary()
hooks.remove_hooks()

# Model structure
ModelArchitectureInspector.print_architecture(model, max_depth=3)
ModelArchitectureInspector.count_parameters(model)
```

---

## Resources

### Papers
- **Qwen-VL**: Official model paper
- **ViT**: "An Image is Worth 16x16 Words"
- **RoFormer**: "Enhanced Transformer with Rotary Position Embedding"
- **Attention**: "Attention Is All You Need"

### Documentation
- [Qwen-VL GitHub](https://github.com/QwenLM/Qwen-VL)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Docs](https://pytorch.org/docs/)

---

## Summary

**Most Important Points:**

1. **Vision-Text Fusion** happens at embedding level
   - Vision features replace placeholder tokens
   - Creates unified multimodal sequence

2. **Vision Encoder** is separate pipeline
   - Patch embedding → Transformer → Merger
   - Output matches text hidden size

3. **Three key breakpoints:**
   - Main forward: `modeling_qwen3_vl.py:1315`
   - Vision encoder: `modeling_qwen3_vl.py:703`
   - Fusion section in `Qwen3VLModel.forward()`

4. **Generation is autoregressive**
   - One token at a time
   - Each conditions on previous tokens

5. **Debug helpers are your friend**
   - Use them to inspect tensors
   - Hook into layers for activation tracking

**Start Here:**
1. Run `demo_with_debugging.py`
2. Set breakpoint at line 1315
3. Step through and observe tensor shapes
4. Focus on fusion mechanism first

Good luck with your deep dive! 🚀
