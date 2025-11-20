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

Before any dataset enters the multi-dataset fusion pipeline, it is expected to have been processed by the existing offline conversion stack driven by `data_conversion/convert_dataset.sh`. That script calls `data_conversion/pipeline/unified_processor.py` and `data_conversion/pipeline/vision_process.py` to (1) apply EXIF orientation and any raw-dimension rescaling, (2) run the `smart_resize` routine with a global pixel budget (`MAX_PIXELS`, default `768 * 28 * 28`) and patch factor (`IMAGE_FACTOR`, default `28`) so that final `width` / `height` are aligned to the vision patch grid (often illustrated as "multiple-of-32" tiles), and (3) write resized images plus flat canonical JSONL (`images` / `objects` / `width` / `height`). During this offline step all geometries (`bbox_2d`, `poly`, `line`) are transformed into the resized image frame and clamped to bounds, so by the time JSONL is consumed by `DenseCaptionDataset` coordinates already match the target resolution and no per-step dynamic resizing is required. The same strategy applies to both target-domain datasets (BBU, future RRU, other QC datasets) and source-domain detection datasets (LVIS, COCO, etc.): converters for public data are expected to emit JSONL that has gone through an equivalent smart-resize stage rather than relying on on-the-fly resizing at training time.

### Augmentation Expectations

`src/datasets/geometry.py` and `src/datasets/augmentation/ops.py` already implement:
- Affine transforms that operate on arbitrary point lists and can convert bbox inputs into polygon approximations when rotation/shear is present.
- Exact polygon clipping via Sutherland–Hodgman and minimum-area-rectangle approximations for degenerate post-crop shapes.

The spec delta switches semantics from “quad” to “poly”:
- Bboxes under general affines are promoted to 4-point `poly` (8 coordinates) rather than a dedicated `quad` field.
- All polygonal shapes (whether originally 4 or >4 points) are represented as `poly` and must preserve vertex ordering and non-degeneracy guarantees.

This keeps the implementation simple (one polygon field) while still allowing special handling of 4-point rectangles via helper functions (e.g., `min_area_rect`, `choose_four_corners`) where needed.

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

Training loss remains the **same teacher-forcing cross entropy** for every record. Mixed batches simply compute CE over the concatenated tokens; there is **no auxiliary-specific scaling factor**. Telemetry may still log `loss_bbu` / `loss_aux` (or per-source variants) for observability, but those breakdowns are post-hoc measurements rather than inputs to the optimizer.

To mitigate long-horizon drift, configs MAY also support an optional **auxiliary shutoff schedule**, e.g.:

- `aux_enabled_epochs = [0, E_aux)` where `E_aux < E_total`.
- After `E_aux`, the DataLoader behaves as if every auxiliary `ratio = 0.0` (pure BBU fine-tuning).

These knobs are required to be exposed in the training config (no hard-coded defaults in dataset code) and used consistently across both fusion modes.

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
     - Annotates each record with `metadata.dataset = "<name>"`.
     - Shuffles and writes a fused train JSONL containing `N_BBU + Σ_d quota[d]` records.

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
  - The aux template **reuses the JSON geometry format** (per-object `bbox_2d` / `poly` / `line` with `norm1000` coordinates) but constrains `desc` to simple category-level Chinese labels (e.g., `"物体类型/类别名"`), without completeness fields or BBU-specific attributes.
  - No Stage-B tokens or fields (e.g., `"显示完整"` / `"只显示部分"`) appear in aux outputs.

Architecturally, template selection SHALL be driven by `metadata.dataset` (or an equivalent flag) via a small registry:

- A prompt registry in `src/config/prompts.py` (or a thin wrapper module) maps template names (e.g., `"bbu_dense"`, `"aux_dense"`) to `(system_prompt, user_prompt)` pairs.
- The dataset builder or fusion step attaches `metadata.template = "<template_name>"` to each record (defaulting to `"bbu_dense"` for the target dataset and `"aux_dense"` for auxiliary datasets).
- The training loop selects prompts at batch construction time based on `metadata.template`, so mixed batches can safely contain both BBU and auxiliary samples.


## Validation & Telemetry

To ensure that auxiliary fusion helps (or at least does not harm) the BBU quality-control objective, the implementation MUST provide the following instrumentation and baselines.

### Validation & Telemetry

BBU validation metrics remain the single criterion for accepting auxiliary fusion runs. Any fusion-enabled config MUST be compared against a BBU-only baseline that shares the same random seed (when feasible), optimizer, LR schedule, augmentation curriculum, and training duration. The baseline is obtained by setting every auxiliary `ratio = 0.0`.

Optional diagnostics MAY include:

- Tracking per-dataset loss curves when multiple auxiliary datasets are used (e.g., `loss_aux_coco`, `loss_aux_lvis`).
- Running auxiliary-only evaluations (COCO-style detection/segmentation metrics) to ensure the encoder retains general-domain coverage.

These diagnostics inform production decisions, but the hard requirement is **non-degraded BBU validations** compared to the baseline (small noise-level fluctuations are tolerable).

## Source-Aware Augmentation and Curriculum

Augmentation remains a record-level preprocessor but follows a unified, domain-aware policy:

- **BBU (target) data**
  - Augmentation can be enabled/disabled via config and may use the full geometry-aware pipeline and curriculum described in `docs/DATA_AUGMENTATION.md`.
  - Curriculum scheduling (`bypass_prob`, rotation strength, crop configuration, etc.) operates **only on BBU samples**: its progress is defined over global training steps, but only records with `metadata.dataset == "bbu"` are eligible for augmentation.

- **Auxiliary (source) data**
  - Always uses **clean images**: no geometric or color operators are applied.
  - The augmentation preprocessor inspects `metadata.dataset` (or `metadata.is_target_domain`) and passes auxiliary samples through unchanged; coverage-based filtering, completeness updates, and `allows_geometry_drops` logic are skipped.

This implies a small API refinement for the augmentation layer:

- `AugmentationPreprocessor` MUST receive a per-record domain indicator (`metadata.dataset` or equivalent) and gate application of ops accordingly.
- The existing curriculum implementation remains unchanged, but its effective scope is limited to BBU samples by this gating.

## `public_data/` Integration

The LVIS/COCO converters and validators under `public_data/` at the repo root (`/data/Qwen3-VL/public_data`) become first-class producers of canonical training JSONL that can participate in the auxiliary pool:

- Converters emit `poly` + `poly_points` for segmentation polygons, with a configurable `max_poly_points` parameter that switches large polygons to simplified representations (e.g., `bbox_2d` envelopes or reduced-vertex polygons) on a per-object basis.
- Validators check `bbox_2d` / `poly` / `line` plus geometry-specific constraints but do not hard-code any knowledge of the BBU taxonomy.

These public datasets can then be referenced in fusion configs as auxiliary sources and mixed into BBU training via the regularization scheme above, without changing the core training code.
