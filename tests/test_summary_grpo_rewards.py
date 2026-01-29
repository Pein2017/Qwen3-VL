import math

from src.rlhf.grpo.rewards.summary.rewards import (
    SummaryAttrPathRecallReward,
    SummaryCategoryRecallReward,
    SummaryDatasetReward,
    SummaryGroupStatsPresenceReward,
    SummaryNoDupKeysPenalty,
    SummaryNotesBBUReward,
    SummaryStrictPenaltyReward,
    SummaryStructuredContentTverskyReward,
    SummaryTextBBUReward,
)


def _completion_with_json(payload: str, *, domain: str = "BBU") -> str:
    return f"<DOMAIN={domain}>, <TASK=SUMMARY>\n{payload}"


def test_summary_no_dup_keys_penalty_detects_duplicate_keys_nested_and_top_level():
    reward = SummaryNoDupKeysPenalty()
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_template": "summary_bbu",
        "summary_ref": '{"统计":[]}',
    }

    dup_top = _completion_with_json('{"统计":[],"a":1,"a":2}')
    dup_nested = _completion_with_json(
        '{"统计":[{"类别":"标签","文本":{"A":1,"A":2}}]}'
    )
    clean = _completion_with_json('{"统计":[],"a":1}')

    assert reward([dup_top], metadata=[meta])[0] == -1.0
    assert reward([dup_nested], metadata=[meta])[0] == -1.0
    assert reward([clean], metadata=[meta])[0] == 0.0


def test_summary_dataset_reward_matches_domain_token():
    reward = SummaryDatasetReward()
    completion = _completion_with_json('{"统计":[]}')
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_template": "summary_bbu",
        "summary_ref": '{"统计":[]}',
    }
    assert reward([completion], metadata=[meta])[0] == 1.0


def test_summary_dataset_reward_mismatch_is_zero():
    reward = SummaryDatasetReward()
    completion = _completion_with_json('{"统计":[]}', domain="RRU")
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_template": "summary_bbu",
        "summary_ref": '{"统计":[]}',
    }
    assert reward([completion], metadata=[meta])[0] == 0.0


def test_summary_dataset_key_penalizes_structured_reward():
    reward = SummaryStructuredContentTverskyReward()
    completion = _completion_with_json(
        '{"dataset":"BBU","统计":[{"类别":"BBU设备","品牌":{"华为":1}}]}'
    )
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_template": "summary_bbu",
        "summary_ref": '{"统计":[{"类别":"BBU设备","品牌":{"华为":1}}]}',
    }
    score = reward([completion], metadata=[meta])[0]
    assert math.isclose(score, 1.0 / 1.3, rel_tol=1e-9, abs_tol=1e-12)


def test_summary_category_recall_reward_partial_overlap():
    reward = SummaryCategoryRecallReward()
    ref = '{"统计":[{"类别":"RRU设备"},{"类别":"站点距离","站点距离":{"26":1}}]}'  # noqa: E501
    pred = '{"统计":[{"类别":"站点距离","站点距离":{"26":1}}]}'
    meta = {
        "_fusion_source": "rru_summary",
        "_fusion_template": "summary_rru",
        "summary_ref": ref,
    }
    completion = _completion_with_json(pred, domain="RRU")
    assert reward([completion], metadata=[meta])[0] == 0.5


def test_summary_attr_path_recall_reward_rewards_nested_path_completeness():
    reward = SummaryAttrPathRecallReward()
    ref = '{"统计":[{"类别":"电线","捆扎":{"整齐":1}}]}'
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_template": "summary_bbu",
        "summary_ref": ref,
    }

    pred_missing = '{"统计":[{"类别":"电线"}]}'
    pred_ok = '{"统计":[{"类别":"电线","捆扎":{"整齐":1}}]}'

    score_missing = reward([_completion_with_json(pred_missing)], metadata=[meta])[0]
    score_ok = reward([_completion_with_json(pred_ok)], metadata=[meta])[0]

    assert score_missing == 0.0
    assert score_ok == 1.0


def test_summary_attr_path_recall_reward_does_not_penalize_omitting_category():
    reward = SummaryAttrPathRecallReward()
    ref = '{"统计":[{"类别":"电线","捆扎":{"整齐":1}}]}'
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_template": "summary_bbu",
        "summary_ref": ref,
    }

    pred_omit_category = '{"统计":[]}'
    score = reward([_completion_with_json(pred_omit_category)], metadata=[meta])[0]
    assert score == 0.0


def test_summary_attr_path_recall_reward_excludes_dynamic_child_keys():
    reward = SummaryAttrPathRecallReward()
    # The digit-only key "263" and key=value token "a=b" should not be required.
    ref = '{"统计":[{"类别":"电线","捆扎":{"整齐":1,"263":1,"a=b":1}}]}'
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_template": "summary_bbu",
        "summary_ref": ref,
    }

    pred_only_stable = '{"统计":[{"类别":"电线","捆扎":{"整齐":1}}]}'
    assert reward([_completion_with_json(pred_only_stable)], metadata=[meta])[0] == 1.0


def test_summary_structured_content_tversky_reward_missing_vs_spurious():
    reward = SummaryStructuredContentTverskyReward()
    ref = '{"统计":[{"类别":"BBU设备","品牌":{"华为":1},"可见性":{"部分":1}}]}'
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_template": "summary_bbu",
        "summary_ref": ref,
    }

    pred_missing = '{"统计":[{"类别":"BBU设备","品牌":{"华为":1}}]}'
    pred_spurious = (
        '{"统计":[{"类别":"BBU设备","品牌":{"华为":1,"中兴":1},"可见性":{"部分":1}}]}'
    )
    score_missing = reward([_completion_with_json(pred_missing)], metadata=[meta])[0]
    score_spurious = reward([_completion_with_json(pred_spurious)], metadata=[meta])[0]

    assert math.isclose(score_missing, 0.5, rel_tol=1e-9, abs_tol=1e-12)
    assert score_spurious > score_missing


def test_summary_text_bbu_reward_normalizes_fullwidth_parentheses_and_penalizes_overflow():
    reward = SummaryTextBBUReward()
    ref = '{"统计":[{"类别":"标签","文本":{"5G-BBU-(正极)":1}}]}'
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_template": "summary_bbu",
        "summary_ref": ref,
    }

    pred_norm = '{"统计":[{"类别":"标签","文本":{"5G-BBU-（正极）":1}}]}'
    assert reward([_completion_with_json(pred_norm)], metadata=[meta])[0] == 1.0

    pred_overflow = (
        '{"统计":[{"类别":"标签","文本":{"5G-BBU-(正极)":1,"A":1,"B":1,"C":1}}]}'
    )
    assert math.isclose(
        reward([_completion_with_json(pred_overflow)], metadata=[meta])[0],
        0.9,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )

    ref_empty = '{"统计":[{"类别":"标签","可读性":{"不可读":1}}]}'
    meta_empty = {
        "_fusion_source": "bbu_summary",
        "_fusion_template": "summary_bbu",
        "summary_ref": ref_empty,
    }
    pred_spam = '{"统计":[{"类别":"标签","文本":{"A":1,"B":1,"C":1}}]}'
    assert math.isclose(
        reward([_completion_with_json(pred_spam)], metadata=[meta_empty])[0],
        -0.1,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


def test_summary_notes_bbu_reward_recall_and_spurious_penalty():
    reward = SummaryNotesBBUReward()
    ref_with_notes = '{"统计":[],"备注":["无法判断品牌"]}'
    meta_with = {
        "_fusion_source": "bbu_summary",
        "_fusion_template": "summary_bbu",
        "summary_ref": ref_with_notes,
    }
    pred_ok = '{"统计":[],"备注":["无法判断品牌"]}'
    pred_missing = '{"统计":[]}'
    assert reward([_completion_with_json(pred_ok)], metadata=[meta_with])[0] == 1.0
    assert reward([_completion_with_json(pred_missing)], metadata=[meta_with])[0] == 0.0

    ref_empty = '{"统计":[]}'
    meta_empty = {
        "_fusion_source": "bbu_summary",
        "_fusion_template": "summary_bbu",
        "summary_ref": ref_empty,
    }
    assert reward([_completion_with_json(pred_ok)], metadata=[meta_empty])[0] == -1.0


def test_summary_strict_penalty_reward_penalizes_extra_lines():
    reward = SummaryStrictPenaltyReward()
    meta = {"_fusion_source": "bbu_summary", "_fusion_template": "summary_bbu"}
    completion = '<DOMAIN=BBU>, <TASK=SUMMARY>\n{"统计":[]}\nextra'
    assert reward([completion], metadata=[meta])[0] == -1.0


def test_summary_rewards_require_strict_two_lines_for_content():
    reward = SummaryDatasetReward()
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_template": "summary_bbu",
        "summary_ref": '{"统计":[]}',
    }
    completion = '<DOMAIN=BBU>, <TASK=SUMMARY>\n{"统计":[]}\nextra'
    assert reward([completion], metadata=[meta])[0] == 0.0


def test_summary_group_stats_presence_reward_rru_only_and_ref_gated():
    reward = SummaryGroupStatsPresenceReward()
    ref = '{"统计":[{"类别":"标签","组":{"1":2}}],"分组统计":{"1":2}}'
    meta = {
        "_fusion_source": "rru_summary",
        "_fusion_template": "summary_rru",
        "summary_ref": ref,
    }

    completion_no_group = _completion_with_json(
        '{"统计":[{"类别":"标签","组":{"1":2}}]}',
        domain="RRU",
    )
    completion_with_group = _completion_with_json(
        '{"统计":[{"类别":"标签","组":{"1":2}}],"分组统计":{"1":2}}',
        domain="RRU",
    )
    assert reward([completion_no_group], metadata=[meta])[0] == 0.0
    assert reward([completion_with_group], metadata=[meta])[0] == 1.0
