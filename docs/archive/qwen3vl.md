### Qwen3-VL (dense) and Qwen3-VL-MoE: From inputs to the first generated token

Status: Archived — Superseded by docs/README.md

This guide has two parts:
- Chat-style walkthrough: a readable narrative from inputs to the first token.
- Stack-tracing path: a precise, line-referenced reading order through the source.

The dense and MoE models share the same multimodal pipeline. The only material difference is in the text decoder MLP: dense uses a single MLP, MoE routes each token to top-k experts via a router.


### Part I — Chat-style walkthrough (image + text)

1) Prepare inputs with the Processor
   - You build a chat prompt (optionally via `apply_chat_template`) and pass an image and text to the `Qwen3VLProcessor`. It tokenizes text, preprocesses the image into `pixel_values`, and computes an image grid `image_grid_thw` describing how many temporal/spatial patches reach the LLM side. The output is a single `BatchFeature` mapping for the model call.

2) Model entry point (CausalLM wrapper)
   - You call `Qwen3VLForConditionalGeneration.forward(**inputs)`. Internally it forwards to `Qwen3VLModel.forward`, which performs the multimodal “glue”: visual encoding, placement into the text stream, and position index builds.

3) Visual encoding
   - The vision tower patchifies the image (3D conv for temporal × H × W), interpolates 2D positional embeddings for the grid you asked, runs several Transformer blocks, and merges features to the LLM embedding size. It also returns “DeepStack” features from early/mid layers (used for early fusion in the text model).

4) Replace text placeholders with image embeddings
   - The text contains special placeholders at `<|vision_start|> <|image_pad|>...<|vision_end|>`. The model finds them and “masked_scatter” swaps the placeholder token embeddings with the actual image features computed above. Now the text sequence contains real visual embeddings at the correct positions.

5) Build multimodal RoPE position indices (3D: temporal, height, width)
   - The model computes 3D RoPE indices aligned with your text-and-vision interleaving. For videos, it uses timestamps; for images, grids are derived from your processor’s `image_grid_thw`. It also caches rope “deltas” to speed up subsequent generation steps.

6) Decode one step in the text model (with optional DeepStack fusion)
   - The text model embeds tokens, constructs a causal mask, builds shared rotary embeddings, and runs the Transformer layers. In early layers, DeepStack adds the visual features at the masked positions back into the hidden states. The final norm produces the decoder’s last hidden state for this step.

7) Project to logits and pick the first new token
   - `lm_head` maps the last hidden state to vocabulary logits. `GenerationMixin` then samples/greedy-picks the first output token. From here on, the model reuses KV cache (and reuses RoPE deltas) for efficient subsequent tokens.


### Part II — Stack-tracing path (dense), with exact lines

- Processor: build inputs and placeholder counts

```114:121:transformers/models/qwen3_vl/processing_qwen3_vl.py
def __call__(
    self,
    images: ImageInput = None,
    text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
    videos: VideoInput = None,
    **kwargs: Unpack[Qwen3VLProcessorKwargs],
) -> BatchFeature:
```

```162:171:transformers/models/qwen3_vl/processing_qwen3_vl.py
if images is not None:
    image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"]) 
    image_grid_thw = image_inputs["image_grid_thw"]
```

```185:195:transformers/models/qwen3_vl/processing_qwen3_vl.py
# Replace <|image_pad|> with the right number of placeholders based on grid and merge_size
text = text.copy()
if image_grid_thw is not None:
    merge_length = self.image_processor.merge_size**2
    index = 0
    for i in range(len(text)):
        while self.image_token in text[i]:
            num_image_tokens = image_grid_thw[index].prod() // merge_length
            text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
            index += 1
        text[i] = text[i].replace("<|placeholder|>", self.image_token)
```

Result: `BatchFeature` contains `input_ids`, `attention_mask`, `pixel_values`, and `image_grid_thw`.

- CausalLM wrapper → base model

```1344:1356:transformers/models/qwen3_vl/modeling_qwen3_vl.py
outputs = self.model(
    input_ids=input_ids,
    pixel_values=pixel_values,
    pixel_values_videos=pixel_values_videos,
    image_grid_thw=image_grid_thw,
    video_grid_thw=video_grid_thw,
    position_ids=position_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    inputs_embeds=inputs_embeds,
    cache_position=cache_position,
    **kwargs,
)
```

- Visual encoding (vision tower)

```714:726:transformers/models/qwen3_vl/modeling_qwen3_vl.py
hidden_states = self.patch_embed(hidden_states)

pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
hidden_states = hidden_states + pos_embeds

rotary_pos_emb = self.rot_pos_emb(grid_thw)
...
emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
position_embeddings = (emb.cos(), emb.sin())
```

```737:753:transformers/models/qwen3_vl/modeling_qwen3_vl.py
for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(
        hidden_states,
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    if layer_num in self.deepstack_visual_indexes:
        deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
            hidden_states
        )
        deepstack_feature_lists.append(deepstack_feature)

hidden_states = self.merger(hidden_states)
```

Back in `Qwen3VLModel.get_image_features`, the merged features are split per image according to the grid:

```1050:1064:transformers/models/qwen3_vl/modeling_qwen3_vl.py
pixel_values = pixel_values.type(self.visual.dtype)
image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
image_embeds = torch.split(image_embeds, split_sizes)
return image_embeds, deepstack_image_embeds
```

- Replace placeholders with real image features

```1066:1104:transformers/models/qwen3_vl/modeling_qwen3_vl.py
special_image_mask, special_video_mask = ...
...
if pixel_values is not None:
    image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
    image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask, _ = self.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

- Build RoPE indices (3D) and cache deltas

```916:934:transformers/models/qwen3_vl/modeling_qwen3_vl.py
def get_rope_index(...):
    """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    vision_start_token_id = self.config.vision_start_token_id
```

```1188:1222:transformers/models/qwen3_vl/modeling_qwen3_vl.py
if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
    position_ids, rope_deltas = self.get_rope_index(
        input_ids,
        image_grid_thw,
        video_grid_thw,
        attention_mask=attention_mask_tensor,
    )
    self.rope_deltas = rope_deltas
else:
    # reuse cached deltas for subsequent decoding steps
    ...
```

- Text model forward + DeepStack fusion in early layers

```782:796:transformers/models/qwen3_vl/modeling_qwen3_vl.py
def forward(..., visual_pos_masks=None, deepstack_visual_embeds=None, ...):
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    ...
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
```

```849:870:transformers/models/qwen3_vl/modeling_qwen3_vl.py
for layer_idx, decoder_layer in enumerate(self.layers):
    layer_outputs = decoder_layer(...)
    hidden_states = layer_outputs

    if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
        hidden_states = self._deepstack_process(
            hidden_states,
            visual_pos_masks,
            deepstack_visual_embeds[layer_idx],
        )
hidden_states = self.norm(hidden_states)
```

- Project to logits (first-token step)

```1358:1373:transformers/models/qwen3_vl/modeling_qwen3_vl.py
hidden_states = outputs[0]
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits = self.lm_head(hidden_states[:, slice_indices, :])
return Qwen3VLCausalLMOutputWithPast(..., logits=logits, ...)
```

Generation uses `prepare_inputs_for_generation` to drop images after prefill and reuse caches:

```1390:1414:transformers/models/qwen3_vl/modeling_qwen3_vl.py
model_inputs = super().prepare_inputs_for_generation(...)
model_inputs["position_ids"] = None
if cache_position[0] != 0:
    model_inputs["pixel_values"] = None
    model_inputs["pixel_values_videos"] = None
```


### Part III — MoE deltas: what changes in the stack (and where to read)

The multimodal flow is identical. The difference is only inside the text decoder layer’s MLP:
- Some layers use a Mixture-of-Experts block: a router picks top-k experts per token, combines outputs.
- Others keep a standard dense MLP (according to `decoder_sparse_step` or explicit `mlp_only_layers`).

- Where MoE selects MLP vs. MoE per layer:

```320:333:transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py
class Qwen3VLMoeTextDecoderLayer(...):
    ...
    if (layer_idx not in config.mlp_only_layers) and (
        config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
    ):
        self.mlp = Qwen3VLMoeTextSparseMoeBlock(config)
    else:
        self.mlp = Qwen3VLMoeTextMLP(config, intermediate_size=config.intermediate_size)
```

- Router and experts (token-level top-k routing):

```141:152:transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py
router_logits = self.gate(hidden_states)
routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
routed_out = self.experts(hidden_states, router_weights, router_indices)
```

```67:76:transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py
class Qwen3VLMoeTextExperts(nn.Module):
    ...
    self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
    self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
```

- MoE compile note and optional aux loss (training):

```400:416:transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py
class Qwen3VLMoePreTrainedModel(...):
    _can_compile_fullgraph = False  # torch.compile full-graph disabled for MoE
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen3VLMoeTextSparseMoeBlock, index=1),
        "hidden_states": Qwen3VLMoeTextDecoderLayer,
        "attentions": Qwen3VLMoeTextAttention,
    }
```


### Part IV — Minimal call tree (dense)

- `Qwen3VLProcessor.__call__` → returns `BatchFeature`
- `Qwen3VLForConditionalGeneration.forward` → `Qwen3VLModel.forward`
- Visual path: `Qwen3VLVisionModel.forward` → returns `image_embeds`, `deepstack_image_embeds`
- Glue: replace placeholders, compute `position_ids` via `get_rope_index`
- Text path: `Qwen3VLTextModel.forward` → layers (+ optional DeepStack fusion) → norm
- Head: `lm_head` → logits → first token via `generate`

MoE is the same except: inside each `Qwen3VLMoeTextDecoderLayer`, MLP is either dense or routed MoE.


### Part V — Practical reading order (dense first, then MoE)

1) Processor (inputs and placeholder math)
   - `transformers/models/qwen3_vl/processing_qwen3_vl.py`: `__call__` body and the image placeholder expansion
2) CausalLM wrapper
   - `transformers/models/qwen3_vl/modeling_qwen3_vl.py`: `Qwen3VLForConditionalGeneration.forward`
3) Visual tower
   - Same file: `Qwen3VLVisionModel.forward` (patchify, pos embeds, blocks, merger)
4) Multimodal glue
   - Same file: `Qwen3VLModel.get_image_features`, `get_placeholder_mask`, and `Qwen3VLModel.forward` for position IDs
5) Text decoder
   - Same file: `Qwen3VLTextModel.forward`, `Qwen3VLTextDecoderLayer.forward` (attention, MLP), DeepStack injection
6) Head + generate
   - Same file: `Qwen3VLForConditionalGeneration.forward`, `prepare_inputs_for_generation`
7) Now skim MoE differences
   - `transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py`:
     - `Qwen3VLMoeTextDecoderLayer.__init__` (MLP vs MoE selection)
     - `Qwen3VLMoeTextSparseMoeBlock.forward` (router+experts)
     - `Qwen3VLMoePreTrainedModel` notes


### Part VI — Quick mental model: MoE vs dense

- Dense: one MLP per layer; compute is predictable and simple.
- MoE: many MLP “experts” exist; a router selects top-k for each token; capacity grows without running every expert per token. Trade-offs: more parameters, routing complexity, optional load-balancing loss, limited full-graph compile.


### Part VII — First-token timeline (dense)

- Processor builds `input_ids`, `pixel_values`, `image_grid_thw`.
- Visual tower encodes image → `image_embeds`, `deepstack_image_embeds`.
- Model swaps placeholders with `image_embeds`.
- Model computes 3D RoPE `position_ids` (and caches deltas).
- Text decoder runs first step (+ DeepStack early fusion).
- `lm_head` produces logits → first token is selected.
- Subsequent steps reuse KV cache and RoPE deltas; images are dropped from the forwarded kwargs.

### Part VIII — Input formats and concrete examples

- input_ids (LongTensor [batch, seq_len]): tokenized text including multimodal tokens:
  - `<|vision_start|> ... <|vision_end|>` wraps each image/video segment.
  - `<|image_pad|>` / `<|video_pad|>` are expanded to the exact number of visual placeholders.
- attention_mask (LongTensor [batch, seq_len]): 1 for valid; 0 for padding.
- pixel_values (FloatTensor [num_images, image_tokens, channels*temporal_patch_size*16*16]):
  - Pre-patchified image tokens from the processor (normalized float tensors).
- pixel_values_videos (FloatTensor [num_videos, video_tokens, channels*temporal_patch_size*16*16]).
- image_grid_thw (LongTensor [num_images, 3]): per image [t, h, w] grids (t=1 for images).
- video_grid_thw (LongTensor [num_videos, 3]): per video [t, h, w] (t depends on temporal sampling/patching).

Notes
- Placeholders inserted per image/video = (t*h*w) // (merge_size^2). Defaults: patch_size=16, merge_size=2.
- Exact h, w, t depend on dynamic resize/sampling inside the processors.

Examples (runnable skeleton)

```python
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor

model_id = "Qwen/Qwen3-VL-4B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)

def show_shapes(inputs):
    keys = ["input_ids", "attention_mask", "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]
    for k in keys:
        if k in inputs and inputs[k] is not None:
            t = inputs[k]
            if isinstance(t, torch.Tensor):
                print(k, tuple(t.shape), t.dtype)
            else:
                print(k, type(t))

# 1) One image
img = Image.new("RGB", (768, 768), color="white")
text = "Describe the image: <|image_pad|>"
inputs = processor(images=img, text=text, return_tensors="pt")
show_shapes(inputs)

# 2) Two images
img1 = Image.new("RGB", (512, 512), color="gray")
img2 = Image.new("RGB", (1024, 768), color="gray")
text = "Compare these: <|image_pad|> and <|image_pad|>"
inputs = processor(images=[img1, img2], text=text, return_tensors="pt")
show_shapes(inputs)

# 3) One video clip with 3 frames
frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(3)]
text = "What happens in the clip? <|video_pad|>"
inputs = processor(videos=[frames], text=text, return_tensors="pt")
show_shapes(inputs)
```

Sanity checks
- tokens_for_image_i == (image_grid_thw[i].prod() // (merge_size**2))
- pixel_values tokens per image == image_grid_thw[i].prod() (before spatial merge inside the model)
- analogous checks for videos with video_grid_thw

### Part IX — Training with SWIFT and Megatron (SFT and RL)

HF/Transformers SFT (swift sft)
- Entry: `swift/cli/sft.py` → `swift.llm.train.sft.SwiftSft`
- Flow: load model+processor via args; build dataset using template; HF Trainer loop.
- Example script:
```1:35:/data/Pein/ms-swift/examples/models/qwen3_vl/transformers.sh
swift sft \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct \
  --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#20000' \
  --train_type lora \
  --freeze_vit true --freeze_aligner true \
  --padding_free true --attn_impl flash_attn \
  --max_length 2048 ...
```

RL (DPO/KTO/PPO/GRPO) with HF backend (swift rlhf)
- Entry: `swift/llm/train/rlhf.py:SwiftRLHF`
- Extends SFT pipeline, prepares ref/reward/value models and RL trainers; switches template modes.

Megatron SFT (megatron sft)
- Entry: `swift/megatron/train/sft.py:MegatronSft`
- Wraps Qwen3‑VL with Megatron mcore for large-scale parallelism; trainer: `MegatronTrainer`.
- Example script (VL baseline shown):
```1:36:/data/Pein/ms-swift/examples/megatron/multimodal/dense/full.sh
megatron sft \
  --load Qwen2.5-VL-7B-Instruct-mcore \
  --freeze_vit true --freeze_aligner true \
  --tensor_model_parallel_size 2 --sequence_parallel true ...
```

Qwen3‑VL Megatron wrapper details
- Registration and HF↔mcore conversion, plus inputs_embeds glue (masked scatter) and DeepStack:
```517:539:/data/Pein/ms-swift/swift/megatron/model/mm_gpt/qwen3_vl.py
class Qwen3VL_Vit(HuggingFaceModule): ...
register_megatron_model(... ModelType.qwen3_vl, ModelType.qwen3_moe_vl ...)
```
- Embedding replacement and DeepStack masks:
```131:144:/data/Pein/ms-swift/swift/megatron/model/mm_gpt/qwen3_vl.py
image_mask = (input_ids == config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
video_mask = (input_ids == config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
visual_pos_masks = image_mask[..., 0] | video_mask[..., 0]
```

Differences vs pure HF
- swift sft: standard HF model; easier multi‑GPU via device_map and tuners (LoRA/QLoRA).
- megatron sft: mcore model for tensor/sequence/context parallel, fp8, activation checkpoint tricks; better for big runs.
- swift rlhf: attaches ref/reward/value models and RL trainers around the same model/processor.

### Part XI — Choosing between ms‑swift and VERL (customization and control)

Below is a pragmatic comparison focused on your needs: dataset flexibility, parameter freezing, LoRA, and selectively training `embedding.weight` and `lm_head.weight`.

- Dataset customization
  - ms‑swift: supports custom datasets and preprocessors via registration or plain `torch.utils.data.Dataset` in scripts. Example:
```375:389:ms-swift/examples/custom/dataset.py
class CustomPreprocessor(ResponsePreprocessor):
    def preprocess(self, row):
        return super().preprocess({ 'query': ..., 'response': ... })

register_dataset(DatasetMeta(ms_dataset_id='swift/stsb', hf_dataset_id='SetFit/stsb', preprocess_func=CustomPreprocessor()))
```
  - VERL: config-first; set `data.custom_cls.path/name` to plug your own class, or use built-in `MultiTurnSFTDataset`/`SFTDataset`. Entry selection:
```375:389:verl-main/verl/trainer/sft_trainer.py
if data_config.custom_cls.get("path", None):
    dataset_cls = load_extern_type(...)
else:
    dataset_cls = MultiTurnSFTDataset
dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
```
  - Verdict: both are flexible. VERL’s Hydra config makes swapping datasets easy at runtime; ms‑swift integrates well with its template/collator stack and examples.

- LoRA/adapters
  - ms‑swift: many PEFT variants via `TunerArguments` and tuners; pick targets by names/regex and `modules_to_save` to keep base heads trainable.
```99:124:ms-swift/swift/llm/argument/tuner_args.py
freeze_parameters, trainable_parameters, target_modules=['all-linear'], target_regex=None, modules_to_save=[]
```
  - VERL: enable LoRA by setting `model.lora_rank>0`, `model.target_modules`, handled in engine builders:
```241:253:verl-main/verl/workers/engine/fsdp/transformer_impl.py
if self.model_config.lora_rank > 0:
    module = get_peft_model(module, LoraConfig(..., target_modules=self.model_config.target_modules))
```
  - Verdict: ms‑swift exposes more tuner families out-of-the-box; VERL covers LoRA cleanly and integrates with FSDP/sequence parallel and vLLM rollout.

- Freezing knobs (including vision/aligner/LLM)
  - ms‑swift: `TunerArguments` supports granular freeze/train lists and regex; multimodal presets for `freeze_llm/freeze_vit/freeze_aligner` are auto-expanded:
```105:119:ms-swift/swift/llm/argument/tuner_args.py
freeze_llm: bool = False
freeze_vit: bool = True
freeze_aligner: bool = True
```
  - VERL: for RL flows, freeze vision via actor config; for SFT you control trainable modules through LoRA target modules or full fine‑tune.
```104:116:verl-main/verl/workers/config/actor.py
freeze_vision_tower: bool = False
```
  - Verdict: ms‑swift offers more fine‑grained user‑facing freeze/train include‑lists; VERL provides the common switches and expects you to steer via LoRA/full‑FT.

- Train `embedding.weight` and `lm_head.weight`
  - ms‑swift: add to `modules_to_save` or `trainable_parameters` to include embeddings/head while using LoRA elsewhere.
```360:423:ms-swift/swift/tuners/utils.py
class ModulesToSaveWrapper(...):  # keeps specified base modules trainable alongside adapters
```
  - VERL: simplest route is full‑finetune (no LoRA) for those layers, or include them in LoRA targets if using adapterized heads; direct selective base‑param training is doable by editing param groups (custom engine) but not exposed as a simple list.
  - Verdict: ms‑swift is friendlier for “LoRA + also train embeddings/lm_head” via `modules_to_save`.

- Loss customization
  - ms‑swift: plugin loss registry; override/choose losses without forking trainers.
```15:22:ms-swift/swift/plugin/loss.py
def cross_entropy_loss_func(outputs, labels, ...): ...  # plus contrastive/reranker variants
```
  - VERL: SFT uses `sft_loss` wired in trainer; PPO/GRPO have policy loss configs (`loss_mode`, KL, entropy) and fused backends.
```79:86:verl-main/verl/trainer/sft_trainer.py
self.loss_fn = partial(sft_loss, config=None)
```
  - Verdict: for standard CE SFT both are fine; ms‑swift offers a clean hook surface for swapping in custom objectives. VERL shines for RL with fused PPO/GRPO paths and rollout backends.

- Multimodal specifics (Qwen3‑VL)
  - Both: patch Qwen3‑VL to compute 3D `position_ids` and scatter visual embeddings; both support multiple images/videos per turn.
```30:133:verl-main/verl/models/transformers/qwen3_vl.py
def get_rope_index(...); def _get_input_embeds(...)
```

Recommendations
- Prefer ms‑swift if you want: rapid dataset tweaks, precise freeze lists, LoRA variants, and “LoRA + train embeddings/lm_head” without custom code.
- Prefer VERL if you want: RL at scale (GRPO/PPO) with vLLM/SG-Lang rollouts, FSDP2+Ulysses SP, and fused PPO heads; dataset swapping via Hydra; SFT or RL in the same infra.

Minimal configs
- ms‑swift (LoRA + train embeddings/head): set `--modules_to_save "wte,lm_head"` or via args API; optionally `--freeze_vit/--freeze_llm`.
- VERL (LoRA + RL): set `actor_rollout_ref.model.lora_rank>0`, `...target_modules=all-linear`, choose rollout `name=vllm`, optionally `freeze_vision_tower=True`.

### Part X — VERL integration for Qwen3‑VL

- Where Qwen3‑VL support lives
  - `verl/models/transformers/qwen3_vl.py`:
    - `get_rope_index(...)` builds 3D position_ids pre‑sharding, matching HF logic.
    - `_get_input_embeds(...)` computes visual features and performs masked_scatter into `inputs_embeds`, preparing `visual_pos_masks` and `deepstack_visual_embeds`.
    - Fused forward backends for PPO/GRPO: `forward_with_normal_backend`, `forward_with_torch_backend`, `forward_with_triton_backend`.
  - `verl/models/transformers/monkey_patch.py`:
    - For `model_type in ["qwen3_vl", "qwen3_vl_moe"]`, patches:
      - `Qwen3VLModel.forward` / `Qwen3VLMoeModel.forward` → `qwen3_vl_base_forward`.
      - `Qwen3VLForConditionalGeneration.forward` / `Qwen3VLMoeForConditionalGeneration.forward` → `forward_with_normal_backend` (or fused kernels via `patch_forward_with_backends`).
    - Adds Ulysses SP input slicing for VLMs (slices `inputs_embeds`, `position_ids`, `visual_pos_masks`, `deepstack_visual_embeds`) and wraps FlashAttention for all‑to‑all.

- How to run RL (GRPO) with Qwen3‑VL + Megatron (example)
```1:79:/data/Pein/verl-main/examples/grpo_trainer/run_qwen3_vl-30b-megatron.sh
python3 -m verl.trainer.main_ppo --config-path=config --config-name='ppo_megatron_trainer.yaml' \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=$HF_MODEL_PATH \
  ... (megatron parallelism/offload/fusion flags) ...
```

- SFT in VERL
  - Entry: `verl/trainer/sft_trainer.py` (Hydra). The engine loads the HF model and applies the Qwen3‑VL monkey patches automatically based on `model.config.model_type`.
  - Configs: `verl/trainer/config/sft_trainer.yaml`, `sft_trainer_engine.yaml`.

- Why VERL patches matter for Qwen3‑VL
  - Ensures multimodal inputs are correctly assembled into `inputs_embeds` even under Ulysses sequence parallel.
  - Provides fused‑kernel forwards (Torch/Triton) for PPO/GRPO: directly computes log_probs/entropy from hidden states.
  - Prepares 3D RoPE indices before sharding for correctness.

### What both Qwen3-VL and Qwen3-VL-MoE do (shared flow)

- 1) Preprocess inputs
  - Text, images, videos go through a Processor to produce `input_ids`, `attention_mask`, `pixel_values`/`pixel_values_videos`, and per-item grids `image_grid_thw`/`video_grid_thw`.

- 2) Encode vision
  - A vision stack turns patches into embeddings and provides “DeepStack” features for early fusion.

```714:753:transformers/models/qwen3_vl/modeling_qwen3_vl.py
hidden_states = self.patch_embed(hidden_states)

pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
hidden_states = hidden_states + pos_embeds
...
for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(
        hidden_states,
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    if layer_num in self.deepstack_visual_indexes:
        deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
            hidden_states
        )
        deepstack_feature_lists.append(deepstack_feature)

hidden_states = self.merger(hidden_states)
```

- 3) Place vision embeddings into the text sequence
  - Special image/video placeholder tokens in the text are replaced with the computed embeddings.

```1128:1144:transformers/models/qwen3_vl/modeling_qwen3_vl.py
if (input_ids is None) ^ (inputs_embeds is not None):
    raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

if inputs_embeds is None:
    inputs_embeds = self.get_input_embeddings()(input_ids)

if pixel_values is not None:
    image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
    image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask, _ = self.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

- 4) Build 3D RoPE indices (temporal, height, width)
  - Uses timestamps for videos; caches deltas for fast generation.

```923:931:transformers/models/qwen3_vl/modeling_qwen3_vl.py
"""Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

# Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
if video_grid_thw is not None:
    video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
    video_grid_thw[:, 0] = 1
```

- 5) Decode with the text model and early vision fusion
  - DeepStack injects vision features into early layers via a mask.

```849:870:transformers/models/qwen3_vl/modeling_qwen3_vl.py
for layer_idx, decoder_layer in enumerate(self.layers):
    layer_outputs = decoder_layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=text_position_ids,
        past_key_values=past_key_values,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = layer_outputs

    # add visual features to the hidden states of first several layers
    if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
        hidden_states = self._deepstack_process(
            hidden_states,
            visual_pos_masks,
            deepstack_visual_embeds[layer_idx],
        )
```

- 6) Language head produces logits; generation reuses KV cache and RoPE deltas.


### How Qwen3-VL-MoE differs

- Text backbone uses Mixture-of-Experts MLPs instead of a single dense MLP
  - A small “router” picks top-k experts per token; their outputs are combined.
  - This increases capacity without running all experts for every token.

- MoE-specific outputs and constraints
  - The causal LM output includes an optional routing auxiliary loss for load balancing.
  - MoE disables full-graph torch.compile optimizations.

```400:411:transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py
class Qwen3VLMoePreTrainedModel(PreTrainedModel):
    ...
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
```

- MoE-specific configuration knobs (router/expert settings)
  - Number of experts, experts per token (top-k), routing loss factor, sparse placement, etc.

```73:90:transformers/models/qwen3_vl_moe/configuration_qwen3_vl_moe.py
attention_dropout (`float`, *optional*, defaults to 0.0):
    The dropout ratio for the attention probabilities.
decoder_sparse_step (`int`, *optional*, defaults to 1):
    The frequency of the MoE layer.
moe_intermediate_size (`int`, *optional*, defaults to 1408):
    Intermediate size of the routed expert.
num_experts_per_tok (`int`, *optional*, defaults to 4):
    Number of selected experts.
num_experts (`int`, *optional*, defaults to 60):
    Number of routed experts.
norm_topk_prob (`bool`, *optional*, defaults to `True`):
    Whether to normalize the topk probabilities.
router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
    The aux loss factor for the total loss.
```

- Defaults reflect different scale
  - Base Qwen3-VL TextConfig defaults: hidden_size=4096, 32 layers/heads.
  - MoE TextConfig defaults: hidden_size=2048, 24 layers/heads, plus MoE fields.

- Vision stack and multimodal glue are otherwise the same
  - Same patchification, positional handling, DeepStack injection, placeholder replacement, RoPE indexing, generation flow.


### “What is MoE?” (quick mental model)

- Think of many parallel MLP “experts.”
- A learned router scores each token, selects top-k experts, normalizes their scores, and mixes their outputs.
- Benefits: larger total capacity; only a few experts run per token (compute-efficient).
- Trade-offs: more parameters in memory (all experts’ weights); routing adds complexity; compile optimizations are limited; training uses an extra load-balancing loss.


### When to choose which

- Qwen3-VL (dense): simpler, stable, great default for most inference and fine-tuning.
- Qwen3-VL-MoE: higher capacity per compute step; consider for larger-scale tasks or when MoE checkpoints are provided; expect higher VRAM for weights and more nuanced performance characteristics.


- Both models share the same multimodal prompt format, special tokens, processor, and vision-text fusion path. The key difference is purely the text MLP: dense vs. routed experts.

- If you’re new to MoE, you can use MoE exactly like the dense model: instantiate `Qwen3VLMoeForConditionalGeneration` with a MoE checkpoint, feed the same processor outputs, and (optionally) watch `aux_loss` during training for router balance.

- Critical note for videos: position indices are timestamp-based, and the model automatically splits video grids per frame during rope index construction, as shown above.


- In short: the pipeline is the same; MoE changes how the text MLP computes per token via a router + experts.