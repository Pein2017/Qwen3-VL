# summary-grpo-post-training Delta

## MODIFIED Requirements

### Requirement: Header matching uses fusion metadata and TASK=SUMMARY
For non-irrelevant samples, header rewards SHALL validate `<DOMAIN>` against a domain token derived from `_fusion_template` (`summary_bbu` → `BBU`, `summary_rru` → `RRU`) and SHALL validate `<TASK>` as the fixed token `SUMMARY` for summary-mode outputs. `_fusion_source == "irrelevant_summary"` SHALL suppress header checks entirely (even if `_fusion_template` is `summary_bbu`).

#### Scenario: Header token match via fusion template
- **GIVEN** a summary-mode sample with `_fusion_template = "summary_bbu"`
- **WHEN** the model emits `<DOMAIN=BBU>, <TASK=SUMMARY>`
- **THEN** header reward is positive

### Requirement: JSON validity and order-invariant equivalence
Content rewards SHALL parse the JSON summary and compare against the ground-truth summary from `metadata.summary_ref` using order-invariant equivalence. The comparison SHALL treat `统计` and `备注` as order-insensitive multisets and SHALL ignore key ordering at the top level. For RRU summaries, `分组统计` SHALL be included in content scoring (group_id → count, order-insensitive by key). The JSON MUST NOT contain a `dataset` key; its presence SHALL invalidate the sample for content rewards.

#### Scenario: Dataset key invalidates summary JSON
- **GIVEN** a summary-mode sample whose JSON contains `"dataset"`
- **WHEN** computing content rewards
- **THEN** the summary is treated as invalid and content rewards are zeroed (or the sample fails fast per implementation)

#### Scenario: Group statistics are scored for RRU
- **GIVEN** an RRU summary reference that includes `分组统计`
- **WHEN** a prediction omits or mismatches the `分组统计` counts
- **THEN** the content reward reflects the mismatch (lower than a matching prediction)
