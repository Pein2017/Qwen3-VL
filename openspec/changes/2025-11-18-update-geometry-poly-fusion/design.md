# Design: update-geometry-poly-fusion

## Geometry Model

### Canonical Types

- `bbox_2d`: `[x1, y1, x2, y2]` axis-aligned bounding boxes in pixel space.
- `poly`: flat list `[x1, y1, ..., xn, yn]` (even length, `n >= 4` for rectangular quads, `n >= 3` for general polygons). An optional companion field `poly_points: n` MAY be present when datasets naturally expose point counts; when present it MUST equal `len(poly) / 2`.
- `line`: flat list `[x1, y1, ..., xn, yn]` for open polylines. An optional companion field `line_points: n` MAY be present; when present it MUST equal `len(line) / 2`.

The training contract and augmentation pipeline operate purely on these three geometry types, with the following invariants:
- Exactly one geometry field per object (`bbox_2d` or `poly` or `line`).
- Coordinates are always integer pixels in the image frame before template encoding (normalization to `norm1000` is delegated to the template).
- `poly_points` and `line_points` are optional metadata fields: when present they MUST equal the number of vertices (`len(poly) / 2` or `len(line) / 2`), but they MAY be omitted in favor of inferring counts from the coordinate list length (useful for ablation runs).

### Offline image resize & JSONL conversion

Before any dataset enters the multi-dataset fusion pipeline, it is expected to have been processed by the existing offline conversion stack driven by `data_conversion/convert_dataset.sh`. That script calls `data_conversion/pipeline/unified_processor.py` and `data_conversion/pipeline/vision_process.py` to (1) apply EXIF orientation and any raw-dimension rescaling, (2) run the `smart_resize` routine with a global pixel budget (`MAX_PIXELS`, default `768 * 28 * 28`) and patch factor (`IMAGE_FACTOR`, default `28`) so that final `width` / `height` are aligned to the vision patch grid (often illustrated as "multiple-of-32" tiles), and (3) write resized images plus flat canonical JSONL (`images` / `objects` / `width` / `height`). During this offline step all geometries (`bbox_2d`, `poly`, `line`) are transformed into the resized image frame and clamped to bounds, so by the time JSONL is consumed by `DenseCaptionDataset` coordinates already match the target resolution and no per-step dynamic resizing is required. The same strategy applies to both target-domain datasets (BBU, future RRU, other QC datasets) and source-domain detection datasets (LVIS, COCO, etc.): converters for public data are expected to emit JSONL that has gone through an equivalent smart-resize stage rather than relying on on-the-fly resizing at training time. Additionally, image references stored inside JSONL are resolved relative to the JSONL directory and promoted to absolute paths during dataset loading so downstream code never depends on working-directory hacks or symlinks to locate images.

### Augmentation Expectations

`src/datasets/geometry.py` and `src/datasets/augmentation/ops.py` already implement:
- Affine transforms that operate on arbitrary point lists and can convert bbox inputs into polygon approximations when rotation/shear is present.
- Exact polygon clipping via Sutherland–Hodgman and minimum-area-rectangle approximations for degenerate post-crop shapes.

The spec delta switches semantics from “quad” to “poly”:
- Bboxes under general affines are promoted to 4-point `poly` (8 coordinates) rather than a dedicated `quad` field.
- All polygonal shapes (whether originally 4 or >4 points) are represented as `poly` and must preserve vertex ordering and non-degeneracy guarantees.

This keeps the implementation simple (one polygon field) while still allowing special handling of 4-point rectangles via helper functions (e.g., `min_area_rect`, `choose_four_corners`) where needed.

## Domain Model & Dataset Wrappers

- **Domains**
  - **Target domain**: downstream QC datasets such as BBU (current primary) and future programs like RRU. These datasets define the main loss objective, always participate in evaluation, and are eligible for all augmentation/curriculum features.
  - **Source domain**: public/general detection datasets (COCO, Objects365, Flickr3k, etc.) that regularize the vision encoder. They default to *no augmentation* and never contribute to evaluation metrics unless explicitly opted in.
- **Wrapper interface**
  - Each dataset is represented by a wrapper class that exposes `domain`, `name`, `template_id`, `train_jsonl`/`val_jsonl` handles, and boolean flags `supports_augmentation` / `supports_curriculum`. Polygon simplification (if any) happens in the offline converters, not in the loader.
  - Wrappers own any dataset-specific preprocessing (image roots, taxonomy quirks, attribute normalization) and emit canonical `ConversationRecord` entries; no downstream code needs per-dataset conditionals.
  - Prompt binding happens per wrapper: `template_id` selects the `(system_prompt, user_prompt)` pair at dataset-construction time, so individual records carry no template metadata.
- **Factory**
  - The fusion factory maps config keys (`bbu`, `rru`, `coco`, `objects365`, `flickr3k`, etc.) to wrapper classes, instantiates them with their `params`, and produces `DatasetSpec` objects consumed by the fusion dataset/builder.
  - Because wrappers report `supports_augmentation`, augmentation/curriculum can be attached (or skipped) per dataset instance without relying on `metadata.dataset` at the record level.

### Wrapper Onboarding Checklist

Whenever a new source or target dataset is introduced, the wrapper must supply:

1. **Canonical JSONL contract** — `images`/`objects`/`width`/`height` plus single-geometry objects, with relative image paths that resolve against the JSONL directory.
2. **Geometry policy controls** — document any offline polygon simplification (e.g., `poly_max_points`) applied during conversion.
3. **Domain flags** — `domain`, `supports_augmentation`, and `supports_curriculum` so the factory can wire augmentation/curriculum only when appropriate.
4. **Template binding** — a `template_id` mapping to prompts in `src/config/prompts.py`; add a new entry there if the dataset needs a specialized prompt.
5. **Image root validation** — explicit checks that referenced directories exist; wrappers must not rely on caller-provided symlinks or ambient CWD to reach the images.

## Multi-Dataset Fusion

### Epoch-Level Fusion Semantics

The fusion strategy treats BBU as the primary training domain and uses general-domain datasets only as a regularizer for the vision encoder.

Let `N_BBU` be the number of training records in the target BBU dataset.

Per logical training epoch:

- The loader SHALL consume **all BBU examples exactly once** (`N_BBU` samples; no subsampling of the target domain).
- For each auxiliary dataset `d`, it SHALL draw `quota[d] = round(ratio_d * N_BBU)` examples based on the ratio declared in the fusion config (e.g., `ratio_coco = 0.1`, `ratio_objects365 = 0.05`).
- Auxiliary examples SHALL be sampled **with replacement** from their respective datasets and **re-sampled independently each epoch** to maximize diversity.

Total samples per epoch:

- `N_total = N_BBU + Σ_d quota[d]`. The effective auxiliary fraction is `Σ_d quota[d] / N_BBU`, which for the example above is `0.15`.

Auxiliary data serves purely as a **regularization signal**; there is no requirement to exhaust or balance the auxiliary pool across training runs.
### Stability, Ratios, and Scheduling

To keep BBU performance stable while leveraging auxiliary regularization, the fusion scheme exposes **per-source ratios** instead of a single global knob:

- Every auxiliary dataset entry in the fusion config declares `ratio` (float, default `0.0`, recommended range `[0.0, 0.5]`). Setting `ratio = 0.0` removes that dataset from the epoch schedule.
- For dataset `d`, the per-epoch quota is `N_aux[d] = round(ratio_d * N_BBU)`. All quotas are computed relative to the same `N_BBU`, so a config like `coco: 0.1`, `objects365: 0.05` yields `10` COCO samples and `5` Objects365 samples when `N_BBU = 100`.

Training loss remains the **same teacher-forcing cross entropy** for every record. Mixed batches simply compute CE over the concatenated tokens; there is **no auxiliary-specific scaling factor**, and ratios stay fixed throughout training (no dynamic ramp-up/down).

**Distribution shift note**

- Auxiliary samples are drawn from general-domain datasets without augmentation, while BBU samples may be heavily augmented.
- The small default ratios (e.g., `0.05–0.1` per dataset) limit the impact of any appearance mismatch.
- Because both domains share the same geometry encoding (`bbox_2d` / `poly` / `line` in pixel space) and norm1000 templates, the main benefit is expected on geometry/representation robustness rather than BBU-specific semantics.



### Implementation: Fusion Builder vs Online Sampler

Two implementation modes are supported; both MUST satisfy the epoch-level semantics above.

1. **Offline fused JSONL (simple / reproducible)**

   To keep `DenseCaptionDataset` and augmentation preprocessors single-purpose, a fusion builder can materialize a fused train file:

   - Input: a small fusion config (YAML/JSON) declaring:
     - A **target dataset** (e.g., BBU) with `train_jsonl` and `val_jsonl`.
     - Zero or more **auxiliary datasets** (e.g., LVIS/COCO) with `train_jsonl`, optional `val_jsonl`, and a `ratio` field that controls each source's per-epoch quota.
   - The builder:
     - Loads or streams the target JSONL to compute `N_BBU`.
     - Computes `quota[d] = round(ratio_d * N_BBU)` for every auxiliary dataset.
     - Draws `quota[d]` auxiliary records **with replacement** from each dataset independently.
     - Writes a fused train JSONL containing `N_BBU + Σ_d quota[d]` records in canonical format (no extra metadata); the downstream config remembers which slices came from which wrapper.

   For strict per-epoch semantics, the builder MAY be invoked once per epoch (e.g., `train_fused.epoch_000.jsonl`, `train_fused.epoch_001.jsonl`). Training configs then point `custom.train_jsonl` at the epoch-specific file, while `custom.val_jsonl` always references the target dataset's validation file (BBU-only evaluation).

2. **Online mixture in the DataLoader (preferred for efficiency)**

   Alternatively, the fusion config can be consumed directly by the data pipeline:

   - The DataLoader iterates deterministically over all `N_BBU` BBU samples once per epoch.
   - In parallel, it draws `quota[d]` samples with replacement from each auxiliary dataset based on its `ratio_d` and interleaves them into the epoch (e.g., by cycling through per-source buffers).
   - No fused JSONL is written; target and auxiliary JSONLs remain separate.

Both modes MUST obey the same invariants:

- **BBU coverage**: each epoch consumes all BBU samples exactly once.
- **Aux budget**: each epoch draws exactly `quota[d]` samples per auxiliary dataset (up to rounding).
- **Aux resampling**: auxiliary draws are independent across epochs.
- **Validation**: validation and test splits remain BBU-only.

## Template Strategy

BBU and auxiliary data use distinct prompt / template schemes so that BBU quality-control semantics are not diluted by generic COCO/LVIS supervision.

- **BBU (target) data**
  - Uses the existing dense JSON and summary templates defined in `src/config/prompts.py` (e.g., `SYSTEM_PROMPT_JSON`, `USER_PROMPT_JSON`).
  - Output objects include `desc` fields with `"类型/属性/[条件属性]"` layering and geometry (`bbox_2d` / `poly` / `line`), plus completeness / 质量 attributes as in current BBU specs.

- **Auxiliary (general-domain) data**
  - Uses a separate "aux-dense" template whose purpose is to keep the vision encoder's object + geometry representations calibrated, **not** to teach BBU-specific quality rules.
  - The aux template **reuses the JSON geometry format** (per-object `bbox_2d` / `poly` / `line` with `norm1000` coordinates) but constrains `desc` to concise English class names (one or two words), without completeness fields or BBU-specific attributes.
  - No Stage-B tokens or fields (e.g., `"显示完整"` / `"只显示部分"`) appear in aux outputs.

Architecturally, template selection is driven by the wrapper-provided `template_id`:

- A prompt registry in `src/config/prompts.py` (or a thin wrapper module) maps template names (e.g., `"bbu_dense"`, `"aux_dense"`) to `(system_prompt, user_prompt)` pairs.
- Each wrapper declares its `template_id`, and the fusion factory passes the resolved prompts directly into the `DenseCaptionDataset` it builds for that wrapper.
- Mixed batches remain safe because the prompt binding happens when the sub-dataset is constructed; individual records require no extra metadata.


## Source-Aware Augmentation and Curriculum

Augmentation remains a record-level preprocessor but the on/off decision now happens per dataset wrapper rather than per-record metadata:

- **Target domain datasets (default: BBU, RRU)**
  - `supports_augmentation=True` and `supports_curriculum=True`.
  - The fusion factory attaches the configured augmentation pipeline (and shared curriculum state) when constructing their `DenseCaptionDataset` instances.
- **Source domain datasets (default: COCO, Objects365, Flickr3k)**
  - `supports_augmentation=False` by default, so the factory simply omits the augmentation preprocessor; records remain untouched apart from canonical geometry handling.
  - Wrappers can opt in by flipping the flag if a particular source should share augmentations with the target domain.

Because the curriculum scheduler already drives a multiprocessing-safe state dict, no additional metadata is necessary—only the datasets that opted in receive the state reference.

## `public_data/` Integration

The LVIS/COCO converters and validators under `public_data/` at the repo root (`/data/Qwen3-VL/public_data`) become first-class producers of canonical training JSONL that can participate in the auxiliary pool:

- Converters emit `poly` + `poly_points` for segmentation polygons, with a configurable `max_poly_points` parameter that switches large polygons to simplified representations (e.g., `bbox_2d` envelopes or reduced-vertex polygons) on a per-object basis.
- Validators check `bbox_2d` / `poly` / `line` plus geometry-specific constraints but do not hard-code any knowledge of the BBU taxonomy.

These public datasets can then be referenced in fusion configs as auxiliary sources and mixed into BBU training via the regularization scheme above, without changing the core training code.
`public_data/scripts/convert_rescale_source.sh` bundles `convert_lvis.py` + smart-resize with a hard-coded `--poly-max-points 12`, writing outputs to `public_data/lvis/rescale_32_768_poly_max_12/train.jsonl`. Fusion configs such as `configs/fusion/bbu_with_lvis.yaml` now point to that capped JSONL so the loader reads prefiltered geometry without any runtime fallback.
## Evaluation

- The default trainer wiring keeps `custom.val_jsonl` pointed at the target dataset’s validation split; auxiliary datasets never participate in evaluation unless a config explicitly adds a diagnostic loader.
- Because auxiliary ratios are fixed and low, regressions are caught by comparing against historical BBU validation runs (no mandatory re-train baseline baked into the spec).
