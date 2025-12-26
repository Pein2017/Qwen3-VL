# detection-preprocessor Spec Delta (update-desc-summary-contract)

## MODIFIED Requirements

### Requirement: Canonical converter contract (all datasets)
All target and source detection converters SHALL emit the canonical BBU-style JSONL and remain resize-free. BBU/RRU converters SHALL follow the key=value description contract and JSON-string summary contract.

#### Scenario: Schema and single-geometry invariant
- **WHEN** any dataset converter (BBU, RRU, LVIS/COCO/Objects365, etc.) writes JSONL
- **THEN** each record provides `images`, `objects`, `width`, `height` where every object has exactly one geometry (`bbox_2d` or `poly`+`poly_points` or `line`) plus a non-empty `desc`
- **AND** image paths are relative to the output JSONL by default (no symlink/CWD assumptions)
- **AND** converters do not apply resizing or augmentation; they only “cook” annotations to canonical JSONL
- **AND** for BBU/RRU domains, desc uses comma-separated key=value pairs with no spaces and no slash-delimited levels

#### Scenario: Converter extensibility
- **WHEN** a new source dataset is added
- **THEN** only the converter implementation is customized; it reuses the shared smart-resize offline path (online guard is legacy-only)

#### Scenario: BBU vs. RRU desc contract
- **WHEN** a BBU converter emits object descriptions
- **THEN** desc includes `备注` when present and MUST NOT include `组`
- **AND** `需复核` is never emitted
- **WHEN** an RRU converter emits object descriptions
- **THEN** desc MAY include `组=<id>` and MUST NOT include `备注`

#### Scenario: Value normalization and OCR handling
- **WHEN** a converter writes desc values
- **THEN** values MUST remove all whitespace (including fullwidth spaces)
- **AND** OCR/备注 free text preserves punctuation (including `,|=`); no symbol replacement is applied
- **AND** OCR content preserves `-` and `/` characters (no replacement)
- **AND** stray comma tokens without `key=` are folded into `备注`
- **AND** `这里已经帮助修改,请注意参考学习` is stripped from `备注` when present
- **AND** station distance values MUST normalize to an integer token (strip non-digits) and be emitted as `站点距离=<int>`

#### Scenario: Deterministic key ordering
- **WHEN** a converter emits desc for any object
- **THEN** key=value pairs follow a deterministic per-category order
- **AND** `类别` is always first
- **AND** `备注` (BBU) or `组` (RRU) is always last when present

#### Scenario: Group encoding (RRU)
- **WHEN** an RRU object has group membership
- **THEN** the desc encodes groups as a single `组=` key
- **AND** multiple groups are joined with `|` in ascending numeric order

#### Scenario: Drop occlusion judgments
- **WHEN** a converter ingests occlusion/遮挡 answers (e.g., `有遮挡`, `无遮挡`, `挡风板有遮挡`)
- **THEN** those values are not emitted in desc key=value pairs
- **AND** occlusion values are not counted in summary JSON

#### Scenario: Conflict resolution (negative precedence)
- **WHEN** a single object includes both a positive and a negative compliance choice (e.g., `符合` and `不符合/露铜`)
- **THEN** the converter selects the negative branch and records the specific issue
- **AND** multiple negative issues are joined with `|`

#### Scenario: JSON-string summaries
- **WHEN** a BBU or RRU converter writes summaries
- **THEN** the summary field is a JSON string describing per-category counts
- **AND** the summary does not use ×N aggregation
- **AND** the summary only reports observed values (no missing counts)
- **AND** BBU summaries include a global `备注` list while RRU summaries omit `备注` and MAY include group statistics
- **AND** irrelevant-image summary streams MAY use the literal string `无关图片` instead of JSON

#### Scenario: Summary JSON schema keys
- **WHEN** a BBU or RRU converter writes summaries
- **THEN** the JSON string includes `dataset`, `objects_total`, and `统计`
- **AND** `统计` is a list of per-category objects each containing `类别` plus any observed attribute counts
- **AND** BBU summaries include a `备注` list of strings (may be empty)
- **AND** RRU summaries MAY include `分组统计` (group id → count) and per-category `组` counts
- **AND** summaries include an `异常` object with `无法解析`, `未知类别`, `冲突值`, and `示例` fields
- **AND** the JSON string is single-line and uses standard separators (`, ` and `: `)
