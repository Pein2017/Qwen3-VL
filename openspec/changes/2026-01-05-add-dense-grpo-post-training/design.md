# Design: Mixed-mode dense GRPO post-training (BBU/RRU detection) with reward shaping

## 1) Primary objective and ordering
This GRPO stage optimizes dense detection quality while preserving summary behavior:
- **Primary**: localization quality (geometry overlap with recall+precision)
- **Secondary**: category correctness (`类别`) on matched objects
- **Auxiliary**: attribute correctness (key=value) with business weighting
- **Regularization**: summary samples (BBU/RRU summary + irrelevant) are mixed into the same GRPO run

## 2) Data and mode routing (mixed-mode)

### 2.1 Fusion dataset composition
Training uses fusion datasets (single template, per-dataset prompt resolution) and relies on metadata tags:
- `metadata._fusion_mode`: `"dense"` or `"summary"`
- `metadata._fusion_source`: dataset name (`bbu_dense`, `bbu_summary`, `irrelevant_summary`, etc.)
- `metadata._fusion_template`: template id (`target_dense_bbu`, `summary_bbu`, etc.)

The GRPO fusion mix is:
- **Targets** (dense): `bbu_dense`, `rru_dense`
- **Sources** (summary regularization):
  - `bbu_summary` ratio `0.5`
  - `rru_summary` ratio `0.5`
  - `irrelevant_summary` ratio `0.2`
- **Explicit exclusions**: `lvis`, `lang_chat` are not included in this GRPO stage.

### 2.2 Output contracts
Dense and summary outputs both follow a two-line contract except irrelevant summary:
- Dense samples: line 1 header, line 2 dense JSON object dict
  - Header: `<DOMAIN=BBU|RRU>, <TASK=DETECTION>`
- Summary samples: line 1 header, line 2 summary JSON string
  - Header: `<DOMAIN=BBU|RRU>, <TASK=SUMMARY>`
- Irrelevant summary: exact single-line `无关图片` (no header line)

These contracts align with the assistant-prefix mechanism that maps `dense -> DETECTION`, `summary -> SUMMARY`.

## 3) Reward function calling convention (ms-swift GRPO)
ms-swift GRPO calls reward functions as:
`reward_func(completions, **kwargs)`
where:
- `completions` is a list of decoded assistant strings
- `kwargs` contains batched per-row fields (e.g., `metadata`, `assistant_payload`, `messages`, etc.)

Custom ORMs must therefore accept the positional `completions` argument and use batched kwargs for context.

## 4) Dense reward architecture (`dense.*`)

### 4.1 Parsing and schema validation
Dense rewards parse the completion as:
1. Split into lines. Require exactly 2 lines.
2. Validate header tokens:
   - `<DOMAIN=...>` must match expected domain token (BBU/RRU).
   - `<TASK=DETECTION>` must match dense task token.
3. Parse line 2 as JSON with duplicate-key rejection.
4. Validate dense object schema:
   - keys are `object_1..object_N` (contiguous numbering preferred)
   - each object has:
     - `desc` (non-empty string)
     - exactly one geometry key: `bbox_2d` or `poly` or `line`
     - optional `line_points` (for `line` objects) is allowed and ignored for scoring
5. Strict geometry validity rules (no fallback):
   - `bbox_2d`: flat array length == 4
   - `poly`: either:
     - list of point pairs `[[x, y], ...]` with at least 3 points, OR
     - flat even-length array `[x1, y1, ...]` with at least 3 points
   - `line`: either:
     - list of point pairs `[[x, y], ...]` with at least 2 points, OR
     - flat even-length array `[x1, y1, ...]` with at least 2 points
   - `line_points` (when present): equals the number of point pairs (`len(line)` when paired, `len(line_flat)//2` when flat)

Invalid geometry is treated as a schema failure and MUST NOT be evaluated via bbox/AABB fallback.

### 4.2 Matching and overlap rulers (exact)
Dense rewards compute overlap in norm1000 space:
- Region family: `bbox_2d` and `poly`
  - Use pixel-level filled-shape IoU computed on the norm1000 grid (size 1000×1000, with coordinates clamped to `[0, 999]`) by rasterizing each region into a binary mask and then taking `|A∩B|/|A∪B|`.
  - This ruler supports non-convex polygons and avoids fragile analytic clipping assumptions.
  - Cross-type `bbox_2d`↔`poly` matching is supported by rasterizing `bbox_2d` as a filled rectangle and `poly` as a filled polygon under the same ruler; the geometry representation type does not matter.
  - The polygon fill rule is the even-odd (parity) rule for determinism and MUST be shared between rewards and offline evaluation.
  - Self-intersecting polygons are scored under the same even-odd fill rule (not rejected), to keep scoring robust to minor vertex-ordering mistakes in model outputs.
  - Data audit note: `analysis/poly_convexity_audit_report.json` shows non-convex `poly` shapes exist in the training targets (BBU ≈0.2% non-convex; RRU ≈10% non-convex), so convex-only polygon clipping would be incorrect for a non-trivial slice of RRU data.
  - Data audit note: `data_new_schema/bbu_full_2048_poly/all_samples.jsonl` and `data_new_schema/rru_full_2048_poly/all_samples.jsonl` contain 0 self-intersecting GT polygons under a segment-intersection check (raw and angle-sorted orders), but the scoring path still accepts self-intersecting predictions for robustness.
- Line family: `line`
  - Use the project’s polyline overlap metric (distance-tolerant coverage-F1 style).
  - Use a stability-first tolerance: default `tol=8.0` in norm1000 space (aligned with offline evaluator defaults).
  - The tolerance is configurable for ablations and MUST be reported in evaluation artifacts.

Matching is greedy 1-to-1 assignment by descending overlap score.

### 4.3 Dense reward set (few composite rewards)
To keep compute practical (exact poly/line overlap is expensive), dense rewards are expressed as a small number of reward functions that share cached parsing/matching within each call:
- `dense.format`: strict 2-line contract + header token check
- `dense.parse_schema_strict`: strict JSON parse + object schema + strict geometry validity
- `dense.loc_mean_fbeta`: mean Fβ over IoU thresholds 0.50..0.95 (localization-only), **default β=2.0**
- `dense.loc_soft_recall`: smooth recall shaping using per-GT best overlap (missing-object reduction)
- `dense.cat_mean_f1`: category-aware mean F1 (pairs must match `类别`)
- `dense.attr_weighted_recall`: attribute scoring on matched pairs (exact strings)

## 5) Dense `desc` scoring rules (exact strings; business weighting)

### 5.1 Parsing
`desc` is a comma-separated list of `key=value` terms. All parsed keys/values are treated as prediction targets.

### 5.2 Category
`类别=<value>` is the category label used for category-aware matching and category reward.

### 5.3 Attribute weighting
For matched objects, attributes are scored via exact match:
- `可见性`: down-weighted (noisy annotation)
- Other non-OCR keys: equal weight
- OCR/notes keys (`文本`, `备注`):
  - bonus-only: high positive reward when exact match
  - no penalty when mismatch or missing

### 5.4 RRU site distance strictness
For RRU objects with `类别=站点距离`:
- Require key `站点距离=<int>` and reward only when the integer value matches exactly.
- This term is weighted highly relative to other attributes to reflect business criticality.

## 6) Summary rewards in mixed-mode
Summary rewards remain responsible for summary contract stability and are included as regularization signal in the mixed-mode run.

Key rule: summary rewards MUST no-op on dense samples to avoid template mapping assumptions (e.g., `target_dense_bbu` is not a summary template).

## 7) Config surface (new presets)

### 7.1 Fusion configs
- `configs/fusion/base/dense_grpo_mixed.yaml`
- `configs/fusion/variants/bbu_rru_dense_grpo_mixed_2048.yaml`

These configs enforce:
- dense targets (bbu/rru)
- summary sources + irrelevant
- exclusion of LVIS/chat

### 7.2 Training presets
- `configs/components/rlhf/dense_summary_grpo_mixed.yaml`
- `configs/train/grpo/dense_summary_mixed_base.yaml`
- `configs/train/grpo/dense_summary_mixed_2048.yaml`

Constraints:
- `rlhf.max_completion_length = 2048`
- `rlhf.reward_funcs` includes both `dense.*` and selected `summary.*` rewards
- `rlhf.num_generations` divides `rlhf.generation_batch_size`

### 7.3 Recommended default reward weights (non-normative starting point)
These defaults are designed to be recall-biased (missing-object reduction) while keeping hallucination penalties small (GT may be incomplete).

Dense-mode reward weights (applied only when `metadata._fusion_mode == "dense"`):
- `dense.format`: `0.1` (format stability; low but non-zero)
- `dense.parse_schema_strict`: `0.2` (hard gate against invalid JSON/geometry; negative scoring inside the reward)
- `dense.loc_mean_fbeta` (F2): `1.0` (primary objective)
- `dense.loc_soft_recall`: `0.5` (missing-object shaping; still bounded)
- `dense.cat_mean_f1`: `0.3` (secondary)
- `dense.attr_weighted_recall`: `0.2` (tertiary; matched-pair only)

Summary-mode reward weights reuse the existing summary GRPO preset (unchanged), and summary rewards no-op on dense samples.

## 8) Offline evaluation alignment
`vis_tools/geometry_eval_metrics.py` is extended to report:
- localization + category mean-F1 (existing)
- attribute weighted match on matched pairs (new)
- OCR/text match rate for `文本` (bonus-only semantics)
- notes match rate for `备注` (bonus-only semantics)
- RRU `站点距离` exact match accuracy (new)

Evaluation MUST use the same geometry rulers as rewards (poly/line exact; invalid geometry is treated as invalid, not approximated).

## 9) Recall bias and weak hallucination penalties (annotation incompleteness)
Dense GT JSONL is known to be incomplete in some slices (missed annotations), so rewards bias toward reducing missing objects and avoid strong penalties on unmatched predictions.

Concretely:
- Localization aggregation uses recall-biased Fβ with moderate bias: **default F2 (β=2.0)**.
  - This yields a simple, explicit weighting: false negatives are weighted `β² = 4×` relative to false positives in the Fβ denominator.
- An additional `dense.loc_soft_recall` term accelerates missing-object reduction early in GRPO.
- Explicit precision/false-positive penalties remain small and bounded (including possibly zero); the default policy relies on the FP term already present in F2 and does not introduce large extra penalties.
- Category/attribute rewards are computed only on matched pairs, so unmatched predictions do not induce large negative signals.

### 9.1 Attribute key weights (non-normative defaults)
For `desc` key=value scoring on matched pairs:
- Default per-key weight for non-OCR keys (excluding `可见性`) is `1.0`.
- `可见性` down-weight factor: `0.1` relative to a normal key (least important).
- OCR/notes keys (`文本`, `备注`) are bonus-only:
  - default bonus weight: `6.0` when exact match (harder; high-signal when matched)
  - no penalty when missing/mismatch
- RRU `站点距离`:
  - default key weight: `4.0` (important but simpler than OCR/notes)
  - exact integer match only (`站点距离=<int>`)

### 9.2 Attribute normalization rules
`desc` attribute matching uses exact string equality after normalization:
- Remove all whitespace characters from both keys and values before comparison (e.g., `品牌=华为` equals `品牌 = 华为`).
- Preserve punctuation (`-`, `/`, `,`, `|`, `=`) exactly as emitted (no fuzzy or semantic matching).
