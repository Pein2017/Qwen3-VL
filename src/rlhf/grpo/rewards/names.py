"""Canonical reward identifiers for summary GRPO."""

LEGACY_SUMMARY_REWARD_NAMES: dict[str, str] = {
    "summary_format": "summary.format",
    "summary_header": "summary.header",
    "summary_strict": "summary.strict",
    "summary_parse": "summary.parse",
    "summary_content": "summary.content_eq",
    "summary_dataset": "summary.dataset",
    "summary_objects_total": "summary.objects_total",
    "summary_content_f1": "summary.content_f1",
    "summary_category_f1": "summary.category_f1",
    "summary_notes_presence": "summary.notes_presence",
    "summary_group_stats_presence": "summary.group_stats_presence",
    "summary_no_dup_keys": "summary.no_dup_keys",
    "summary_objects_total_lb": "summary.objects_total_lb",
    "summary_category_recall": "summary.category_recall",
    "summary_content_structured_tversky": "summary.content_structured_tversky",
    "summary_text_bbu": "summary.text_bbu",
    "summary_notes_bbu": "summary.notes_bbu",
}

SUMMARY_REWARD_NAMES: tuple[str, ...] = tuple(LEGACY_SUMMARY_REWARD_NAMES.values())
