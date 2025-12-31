# Tasks

- [x] Add optional `summary_label_grouping` field to fusion dataset specs and wrappers.
- [x] Propagate per-dataset override through fusion config parsing and `FusionCaptionDataset` preprocessing.
- [x] Keep global `custom.summary_label_grouping` as default when no override is provided.
- [x] Update `configs/fusion/summary_lang_chat_0p2.yaml` to disable grouping for `bbu_summary` and `rru_summary` entries.
- [x] Update docs to describe the per-dataset override and precedence.
- [ ] (Optional) Add a small unit test to validate per-dataset override behavior in fusion summary mode.
