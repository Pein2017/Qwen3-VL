# fusion-dataset Specification

## Purpose
Define unified fusion-dataset behavior: source/target quotas, curriculum-friendly scheduling, and dataset-specific wrapper expectations used by fusion loaders.
## Requirements
### Requirement: Single-template unified fusion with prompt priority
The fusion loader SHALL reuse a single template instance and select user/system prompts per sample based on that sample’s dataset **and** mode using the priority `default < domain < dataset`, with system prompt injection restored after encoding.

#### Scenario: Mode-aware prompt selection
- **WHEN** encoding a fused sample
- **THEN** dense samples follow `default < domain < dataset` prompt priority, while summary samples use the configured summary prompt for their dataset/template
- **AND** the template system prompt is restored after encoding to avoid cross-sample contamination.

### Requirement: Per-source augmentation and curriculum policy
The fusion loader SHALL honor per-source policies for augmentation and curriculum: targets inherit the configured augmentation/curriculum; sources remain clean with augmentation/curriculum disabled regardless of configuration.

#### Scenario: Target-only augmentation
- **WHEN** a target-domain sample is fetched during training and global augmentation is enabled
- **THEN** augmentation and optional curriculum preprocessors run on the target sample
- **AND** source-domain samples in the same epoch bypass augmentation and curriculum even if their entries request it.

### Requirement: Per-source object cap
The fusion loader SHALL support per-source object caps to control sequence length, applying caps only to source-domain samples during training; targets remain uncapped in both training and evaluation.

#### Scenario: Source train cap applied
- **WHEN** the split is `train` and a source-domain policy sets `max_objects_per_image=K`
- **THEN** the loader enforces the cap deterministically before encoding that source sample
- **AND** target-domain samples ignore any cap and keep all objects
- **AND** evaluation mode ignores caps for all domains.

### Requirement: Deterministic per-epoch resampling and shuffling
The fusion loader SHALL rebuild its schedule each epoch: compute each target quota as `round(len(target) * ratio_i)` with `ratio_i` defaulting to `1.0` (ratio < 1 downsamples, ratio > 1 upsamples with replacement), then sample each source by `round(source_ratio * N_target_total)` using either without-replacement draws when enabled or with-replacement draws otherwise, and finally shuffle with a deterministic seed; when without-replacement is enabled and the quota exceeds the source pool, the loader SHALL fall back to deterministic with-replacement sampling and surface this in epoch telemetry.

#### Scenario: Source without-replacement within pool size
- **WHEN** a source sets `sample_without_replacement: true` and its quota for the epoch is less than or equal to its pool size
- **THEN** the fusion loader shuffles the source indices deterministically for that epoch and takes the first `quota` indices (no duplicates within the epoch)
- **AND** target sampling and other sources remain unchanged.

#### Scenario: Source without-replacement quota exceeds pool
- **WHEN** a source sets `sample_without_replacement: true` but its computed quota exceeds its available pool size
- **THEN** the fusion loader falls back to deterministic with-replacement sampling for that source for the epoch
- **AND** the fallback is reflected in epoch-level telemetry/plan for observability.

#### Scenario: Default replacement retained
- **WHEN** a source does not set `sample_without_replacement`
- **THEN** the fusion loader samples that source with replacement exactly as before (deterministic, seeded) while still honoring target quotas and per-epoch shuffling.

### Requirement: No online smart-resize guard
The fusion loader SHALL NOT perform any online smart-resize; inputs are assumed pre-scaled, and resizing occurs only via augmentation ops when configured.

#### Scenario: Oversized input
- **WHEN** a sample exceeds expected dimensions
- **THEN** the fusion loader raises/alerts rather than silently resizing; any resizing must be via explicit augmentation ops.

### Requirement: Telemetry and validation hooks
The fusion loader SHALL emit per-sample debug info and per-epoch aggregates covering dataset name, prompt selection, augmentation enabled/disabled, object-cap hits, smart-resize hits, and input length to support OOD/OOM diagnostics.

#### Scenario: Telemetry capture
- **WHEN** a batch is drawn from mixed sources
- **THEN** the loader records the source name, whether augmentation ran, whether object cap or smart-resize was applied, and the input length for that sample, and exposes aggregate counters per epoch for monitoring.

### Requirement: Optional per-source evaluation splits
The fusion loader SHALL support optional source `val_jsonl` splits; evaluation defaults to target-only and uses fixed (non-shuffled) splits.

#### Scenario: Source eval provided
- **WHEN** a source supplies `val_jsonl`
- **THEN** the eval dataloader can include that split without shuffling, while target evaluation remains the default when source eval is absent.

### Requirement: Unified-only fusion loader
The fusion stack SHALL run exclusively through the unified fusion loader (single shared template); the legacy MultiSourceFusionDataset path is removed.

#### Scenario: No legacy fallback
- **WHEN** a training config declares `custom.fusion_config`
- **THEN** the trainer instantiates the unified fusion loader without a legacy flag or alternate code path
- **AND** attempts to reference the legacy loader are rejected during config parsing.

### Requirement: Per-dataset mode selection with strict validation
Fusion datasets SHALL honor per-dataset mode declarations and validate records accordingly, while falling back to the run-level default only when a dataset omits `mode`.

- **WHEN** a fusion config entry declares `mode ∈ {dense, summary}` (or `use_summary` alias)
- **THEN** the loader SHALL resolve the sample’s mode from the dataset’s declaration, falling back to the global `custom.use_summary` only when the dataset omits `mode`
- **AND** summary mode SHALL require each record to carry a non-empty `summary` string, while dense mode SHALL require at least one object with valid geometry
- **AND** mixing summary and dense datasets in the same epoch SHALL be supported without schema errors.

#### Scenario: Mixed-mode fusion run
- **GIVEN** a fusion config whose target dataset sets `mode: summary` and whose source dataset sets `mode: dense`
- **WHEN** the fusion loader builds a batch
- **THEN** target samples use summary prompts and summary validation, source samples use dense prompts and object/geometry validation
- **AND** per-sample telemetry/debug output records the dataset name, resolved mode, and prompt source.

### Requirement: Multi-target fusion config acceptance
The fusion loader SHALL accept a `targets` list (one or more entries) and treat legacy `target` as a one-element `targets`, enforcing unique dataset names across all targets and sources.

#### Scenario: Parse multi-target config
- **WHEN** a fusion config declares `targets: [bbu, rru]` plus sources `lvis` and `lang_chat`
- **THEN** the loader parses both targets as target-domain datasets, sources as auxiliary, and rejects any name collisions.

#### Scenario: Legacy single target
- **WHEN** a fusion config provides only `target: { ... }`
- **THEN** it is normalized to a single-element targets list without changing existing behavior.

### Requirement: Self-scaled target quotas per epoch
The fusion scheduler SHALL support per-target `ratio` values that scale each target by its own pool size: `quota_i = round(len_i * ratio_i)` with `ratio_i` defaulting to `1.0`; ratios below 1 downsample, ratios above 1 upsample with replacement, and indices are shuffled deterministically each epoch.

#### Scenario: Three targets scaled independently
- **GIVEN** target pool sizes 100, 200, 300 with ratios 0.5, 1.0, 1.5
- **WHEN** an epoch schedule is built
- **THEN** quotas are 50, 200, and 450 respectively (upsample with replacement for the third target)
- **AND** each target’s indices are shuffled deterministically using mixed seeds (global, epoch, per-target seed).

### Requirement: Source quotas keyed to total target quota
The fusion scheduler SHALL compute each source quota as `round(source.ratio * total_target_quota)` where `total_target_quota` is the sum of all target quotas for that epoch; sources are sampled with replacement using deterministic seeds.

#### Scenario: Source quota from combined targets
- **GIVEN** total target quota of 303 and a source ratio 0.1
- **WHEN** building the epoch schedule
- **THEN** 30 source samples are drawn (with replacement) using deterministic seeding, and sources remain augmentation- and curriculum-off with object caps applied when configured.

### Requirement: Multi-target evaluation coverage
The fusion eval loader SHALL iterate all target validation splits (concatenated, deterministic order), SHALL NOT include sources in eval, and SHALL NOT downsample targets unless an explicit eval limit is provided.

#### Scenario: Eval over two targets
- **WHEN** targets `bbu` and `rru` each provide `val_jsonl`
- **THEN** the eval dataloader length equals the sum of their val records, with no source samples included.

### Requirement: Compatibility and defaults
When no target ratios are provided, the scheduler SHALL default each target to `ratio=1.0` (full coverage), and legacy single-target configs SHALL remain valid without code changes to callers.

#### Scenario: No ratios given
- **WHEN** a multi-target config omits `ratio`
- **THEN** every target sample is included once per epoch (full coverage), and sources still use the total target count for their quotas.

### Requirement: Target-only evaluation
The fusion loader SHALL build evaluation datasets from the target domain only; source splits are ignored even when provided, and no augmentation or object caps run during evaluation.

#### Scenario: Source eval ignored
- **WHEN** a fusion config includes a target `val_jsonl` and one or more source `val_jsonl`
- **THEN** the eval dataset is constructed solely from the target split (or errors if missing)
- **AND** source-domain samples are excluded from evaluation metrics and epoch counts
- **AND** no augmentation or object caps are applied in evaluation mode.

### Requirement: Domain provenance for fused records
The fusion loader SHALL tag every sample with `_fusion_domain`, `_fusion_source`, and `_fusion_template` so downstream telemetry and auditing can attribute padded batches to the correct domain.

#### Scenario: Provenance metadata available
- **WHEN** a sample is emitted by the fusion loader (online or offline fused JSONL)
- **THEN** its metadata contains `_fusion_domain`, `_fusion_source`, and `_fusion_template`
- **AND** these fields are preserved through preprocessing and encoding for downstream debugging, metric attribution, and padded batching

### Requirement: Irrelevant summary stream uses single-line output via BBU prompt
Irrelevant-summary samples SHALL be identified by `metadata._fusion_source` (e.g., `irrelevant_summary`). For those samples, the summary prompt SHALL reuse the BBU summary template and SHALL instruct a single-line output exactly `无关图片` (no `<DOMAIN=...>, <TASK=...>` header). All non-irrelevant summary samples SHALL retain the two-line summary contract with the `<DOMAIN=...>, <TASK=...>` header line followed by a JSON summary line.

#### Scenario: Irrelevant summary source uses single-line output
- **GIVEN** a summary-mode sample whose `metadata._fusion_source` equals `irrelevant_summary`
- **WHEN** prompt resolution runs for that sample
- **THEN** the applied prompt is the BBU summary template
- **AND** the prompt specifies a single-line output exactly `无关图片` without a header line.

#### Scenario: Non-irrelevant summary keeps two-line contract
- **GIVEN** a summary-mode sample whose `metadata._fusion_source` is not `irrelevant_summary`
- **WHEN** prompt resolution runs for that sample
- **THEN** the prompt keeps the two-line summary output contract (`<DOMAIN=...>, <TASK=...>` + JSON line).

#### Scenario: Irrelevant pool listed under targets for full coverage
- **GIVEN** the fusion config lists `irrelevant_summary` under targets with `ratio=1.0`
- **WHEN** target quotas are computed
- **THEN** the loader schedules the full irrelevant pool (subject to rounding) like other targets
- **AND** each record still carries `metadata._fusion_source=irrelevant_summary` so downstream handling can treat it as source-like.

### Requirement: Irrelevant summary prompts alternate between BBU and RRU
For samples where `metadata._fusion_source` equals `irrelevant_summary`, the prompt resolver SHALL randomize between `summary_bbu` and `summary_rru` prompts (~50/50 over the pool) while keeping `_fusion_source` unchanged. During training, the assignment SHALL be randomized per sample; evaluation MAY use a deterministic mapping.

#### Scenario: Per-sample randomized prompt alternation for irrelevant samples
- **GIVEN** two irrelevant records with distinct stable identifiers (e.g., image paths)
- **WHEN** prompts are resolved during training
- **THEN** each record selects either `summary_bbu` or `summary_rru` using randomized assignment
- **AND** repeated epochs may assign a different prompt for the same record identifier.

### Requirement: Summary header derivation uses existing fusion metadata
For non-irrelevant summary samples, the assistant header SHALL be generated by deriving a domain token from existing metadata:
- If `_fusion_source == "irrelevant_summary"` → no header (even if `_fusion_template` is `summary_bbu`).
- Else if `_fusion_template == "summary_bbu"` → `<DOMAIN=BBU>`.
- Else if `_fusion_template == "summary_rru"` → `<DOMAIN=RRU>`.
Any summary sample that cannot resolve a domain token by these rules SHALL fail fast before prompt construction.

#### Scenario: Header derived from fusion template
- **GIVEN** a summary-mode sample with `_fusion_template = "summary_rru"`
- **WHEN** prompts are resolved
- **THEN** the assistant header uses `<DOMAIN=RRU>, <TASK=SUMMARY>`

#### Scenario: Irrelevant summary suppresses header
- **GIVEN** a summary-mode sample with `_fusion_source = "irrelevant_summary"`
- **WHEN** prompts are resolved
- **THEN** no header is emitted and the assistant output is single-line `无关图片`

#### Scenario: Unknown summary template fails fast
- **GIVEN** a summary-mode sample whose `_fusion_template` is not `summary_bbu` or `summary_rru`
- **WHEN** prompts are resolved
- **THEN** prompt construction fails with a clear error naming the template value

