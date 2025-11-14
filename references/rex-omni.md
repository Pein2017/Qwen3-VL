# Rex-Omni Project — Comprehensive Technical Brief

## 1. Project Landscape
Rex-Omni is a 3B-parameter multimodal large language model (MLLM) derived from Qwen2.5-VL that unifies detection, pointing, keypoint extraction, OCR (box/polygon), and GUI grounding by treating every perception task as next-token generation. Image evidence is encoded as a sequence of visual tokens; spatial predictions are emitted as discrete coordinate bins (`<0>`–`<999>`), letting standard language modeling infrastructure cover dense vision problems. This repository mirrors the upstream IDEA-Research drop under `references/Rex-Omni/` and layers Codex-facing summaries, evaluation recipes, and application diagrams on top of it.

---

## 2. Architecture & Representation
### 2.1 Base Model
- Backbone: `Qwen2.5-VL-3B-Instruct`, a transformer with joint image-text token streams.
- Vision interface: images are resized to respect `min_pixels` / `max_pixels` (defaults `16*28*28` ≤ pixels ≤ `2560*28*28`) before patch embedding; both backends rely on 28×28 granularity.
- Language head: standard causal LM head fine-tuned to emit Rex-Omni spatial vocabulary, including guard tokens (`<|object_ref_start|>`, `<|box_start|>`, `<|im_end|>`, etc.).

### 2.2 Wrapper Abstraction (`rex_omni/wrapper.py`)
- `RexOmniWrapper` materializes the chosen backend (`transformers` or `vllm`), loads the paired `AutoProcessor`, and stores generation controls (`max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `stop`).
- `_normalize_batch_inputs` harmonizes scalar versus per-image lists for tasks, categories, keypoint types, and visual prompt boxes, ensuring mixed workloads stay aligned.
- `_generate_prompt` renders task templates from `TaskConfig`, optionally injecting keypoint schemas (`get_keypoint_config`) or serialized visual prompt boxes.
- Backends diverge at generation time but share downstream parsing and return structure.

### 2.3 Output Grammar
- Categories appear inside `<|object_ref_start|>category<|object_ref_end|>` spans.
- Geometry payloads live inside `<|box_start|> ... <|box_end|>`:
  - Bounding boxes: `<x0><y0><x1><y1>`
  - Points: `<x><y>`
  - Polygons: alternating `<x_i><y_i>` pairs
- Coordinates are 0–999 bins. Conversion utilities in `parser.py` map between absolute pixels and bins.
- Keypoints are emitted as JSON with `bbox` and per-joint entries, still using the bin vocabulary.

---

## 3. Dataset & Annotation Pipeline
### 3.1 Ground-Truth Encoding
1. **Resample Images**: Align each image to the `smart_resize` budget so the processor and tokenizer agree on patch counts.
2. **Normalize Coordinates**: For every annotation, divide by original width/height, clamp to `[0,1]`, then multiply by 999 and floor to integers (`convert_boxes_to_normalized_bins`).
3. **Assemble Prompt-Response Pairs**: Build task-specific prompts (`TaskConfig.prompt_template`). The target text is a concatenation of category spans plus geometry tokens (or JSON for keypoints) finishing with `<|box_end|>`.
4. **Multi-Task Mixing**: Blend detection, pointing, visual prompting, OCR, GUI grounding, and keypoint samples in the training schedule so the model learns the shared grammar.

### 3.2 Data Sources & Preparation
- **Curated Benchmarks**: Official evaluation uses `Mountchicken/Rex-Omni-EvalData` (COCO, LVIS, Dense200, DocLayNet, HierText, RefCOCOg, VisDrone, etc.). Each dataset contributes JSONL annotations aligned with Rex-Omni prompts.
- **Phrase Grounding Generation** (`applications/_2_automatic_grounding_data_engine`):
  - Extract noun phrases via spaCy, detect them with Rex-Omni, and save JSONL containing image metadata, text spans, and bboxes for downstream fine-tuning.
  - Useful for bootstrapping grounding corpora or augmenting underrepresented categories.
- **Visual Prompt Pairs**: When curating visual prompting data, store reference boxes alongside the target instances. During training, serialize those reference boxes into `<bin>` tokens within the prompt.

### 3.3 Data Management Considerations
- Maintain consistent category phrasing; the model reproduces category text verbatim in outputs.
- Ensure polygons maintain clockwise or counter-clockwise ordering to simplify post-processing.
- For OCR, include transcription text immediately after geometry tokens to keep recognition supervision tied to spatial spans.
- Capture instance-level metadata (difficulty flags, occlusion) externally if needed—Rex-Omni tokens focus solely on geometry and phrase labels.

---

## 4. Training Strategy
> Note: This repository ships inference utilities and evaluation harnesses; fine-tuning scripts are not released. The following guidance derives from the model’s token grammar and dataset format.

1. **Objective**: Standard causal language modeling on mixed vision-text sequences. Loss is applied over the generated coordinate tokens, textual category spans, and OCR strings.
2. **Curriculum**: Begin with detection/pointing (simpler token patterns), then introduce polygons, keypoints, and OCR where sequences grow longer. Maintain a held-out GUI grounding split to track overfitting on UI-specific phrasing.
3. **Prompt Diversity**: Shuffle category ordering, include plural/singular variants, and vary instruction phrasing to prevent prompt overfitting.
4. **Augmentation Hooks**:
   - Apply geometry-preserving image transforms (scaling, flipping) while adjusting annotations before bin conversion.
   - For OCR, consider synthetic overlays to expand typography coverage.
5. **Optimization Parameters**:
   - Mixed precision (`bf16`/`fp16`) with gradient checkpointing to handle high token counts.
   - Cap `max_tokens` to the longest expected sequence (keypoint JSON + multiple instances) to avoid wasted padding.
   - Use low learning-rate adapters (LoRA) if full fine-tuning is infeasible.
6. **Validation**: Decode predictions back into pixel coordinates and compute task-specific metrics (AP for boxes, AR for points, OCR accuracy, PCK for keypoints). Reuse the evaluation scripts supplied in `evaluation/scrpts` to stay consistent with public reporting.

---

## 5. Inference Pipeline Deep Dive
1. **Initialization**
   - Choose backend.
     - *Transformers*: loads `Qwen2_5_VLForConditionalGeneration`, enforces left-padding tokenizer for Flash Attention, and executes `model.generate` directly on GPU. Device placement defaults to `auto`.
     - *vLLM*: instantiates `vllm.LLM` with shared tokenizer/processor, enabling high-throughput scheduled decoding.
   - Sampling parameters mirror training defaults (greedy or low-temperature with optional nucleus/top-k controls).

2. **Request Assembly**
   - Accept a single `PIL.Image` or list. `_normalize_batch_inputs` broadcasts scalar arguments across the batch and validates per-image lists.
   - Compute resized dimensions via `smart_resize`; embed the image and prompt within a chat template (system prompt + user turn).

3. **Generation**
   - Transformers backend: tokenizes multimodal inputs (`processor(...)`), runs `model.generate`, trims prompt tokens, and decodes.
   - vLLM backend: builds `{prompt, multi_modal_data}` payloads, calls `LLM.generate`, and captures sampling metadata from `SamplingParams`.

4. **Post-Processing**
   - `parser.parse_prediction` extracts category spans and converts bins → absolute pixels (`(bin / 999) * width/height`).
   - For keypoints, `parse_keypoint_prediction` reads JSON blocks, reporting visibility flags (`"unvisible"`) as necessary.
   - Each result dictionary bundles success flag, parsed predictions, raw text, timing, token counts, prompt used, and original/resized image sizes.

5. **Visualization**
   - `RexOmniVisualize` draws boxes, points, polygons, and skeletons using deterministic colors (`ColorGenerator`).
   - Supports optional custom color palettes and toggling label rendering.

6. **Batching & Throughput Tips**
   - Keep tasks homogeneous within a batch where possible to reduce variance in output length.
   - For vLLM, tune `gpu_memory_utilization`, `tensor_parallel_size`, and `limit_mm_per_prompt` to match hardware constraints.
   - Monitor `tokens_per_second` in returned metadata to spot regression across releases.

---

## 6. Evaluation & Quality Tracking
- **Datasets**: Extract archives under `Rex-Omni-Eval/`, keeping images and `_annotations/` JSONL in sync.
- **Metrics**:
  - Box tasks → COCO/LVIS AP via FastEvaluate (`evaluation/fastevaluate`).
  - Point tasks → average localization accuracy (distance thresholded).
  - Visual prompting → detection AP restricted to prompted categories.
  - Keypoints → Percentage of Correct Keypoints (PCK) computed from parsed JSON outputs.
- **Scripts** (`evaluation/scrpts/*.sh`): Parameterize dataset name, eval type, model path, image root, and output directory. Results aggregate under `_rex_omni_eval_results/` for reproducibility.
- **Error Analysis**: Compare raw text outputs alongside parsed structures to catch prompt drift or parsing mismatches. Visualization utilities accelerate qualitative review.

---

## 7. Applications & Extensibility
- **Segmentation Augmentation** (`applications/_1_rexomni_sam`): Use Rex-Omni detections as SAM prompts to achieve box-to-mask conversion, enriching segmentation datasets without manual labeling.
- **Automatic Grounding Data Engine**: Converts caption corpora into grounding datasets, suitable for further Rex-style training or evaluation.
- **GUI Interactions**: GUI grounding/pointing tasks demonstrate cross-domain coverage (desktop/mobile UI). Extend prompts with UI metadata (hierarchy tags, screen states) for richer supervision.
- **Custom Tasks**: To add a new capability, define a `TaskType`, register a `TaskConfig`, adjust `_generate_prompt`, and extend the parser to understand the new output grammar.

---

## 8. Limitations & Operational Notes
- Training code is absent; reproducing fine-tuning requires building an external trainer around the described token grammar and datasets.
- Token-bin discretization caps localization fidelity at ~0.1% of image dimension; extremely small objects may need post-processing or higher-resolution normalization.
- OCR sequences increase token length quickly; watch for `max_tokens` exhaustion when combining long transcription strings with dense layouts.
- Multi-task prompted batches can surface unexpected formatting if category phrasing diverges from training distribution—maintain prompt hygiene.

---

## 9. Key Takeaways for Codex Integrations
- Rex-Omni operates as a vision-to-text translator; once prompts and bins are respected, standard LLM infrastructure handles both training (if available) and inference.
- Dataset preparation hinges on accurate bin conversions and prompt construction. Automate these steps to avoid subtle off-by-one errors that degrade AP metrics.
- The provided evaluation stack doubles as a validation harness for future fine-tunes; keep outputs in the same directory layout to compare against upstream baselines.
- When extending Rex-Omni, focus on maintaining consistent grammar between training targets and inference parsing—this contract is the core of the project’s flexibility.

