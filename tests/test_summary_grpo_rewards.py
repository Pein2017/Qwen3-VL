import pytest

from src.rlhf.summary_grpo_rewards import (
    SummaryContentF1Reward,
    SummaryDatasetReward,
    SummaryObjectsTotalReward,
)


def _completion_with_json(payload: str, *, domain: str = "BBU") -> str:
    return f"<DOMAIN={domain}>, <TASK=SUMMARY>\n{payload}"


def test_summary_content_f1_reward_exact_match():
    reward = SummaryContentF1Reward()
    ref = '{"统计":[{"类别":"标签","文本":{"ABC":1}}],"备注":["需要补拍"]}'
    completion = _completion_with_json(ref)
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_domain_token": "BBU",
        "summary_ref": ref,
    }
    assert reward([completion], metadata=[meta])[0] == 1.0


def test_summary_content_f1_reward_partial_match():
    reward = SummaryContentF1Reward()
    ref = '{"统计":[{"类别":"标签","文本":{"ABC":1,"DEF":1}}]}'
    pred = '{"统计":[{"类别":"标签","文本":{"ABC":1}}]}'
    completion = _completion_with_json(pred)
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_domain_token": "BBU",
        "summary_ref": ref,
    }
    assert reward([completion], metadata=[meta])[0] == pytest.approx(2.0 / 3.0)


def test_summary_content_f1_reward_no_overlap_zero():
    reward = SummaryContentF1Reward()
    ref = '{"统计":[{"类别":"标签","文本":{"ABC":1}}]}'
    pred = '{"统计":[{"类别":"标签","文本":{"XYZ":1}}]}'
    completion = _completion_with_json(pred)
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_domain_token": "BBU",
        "summary_ref": ref,
    }
    assert reward([completion], metadata=[meta])[0] == 0.0


def test_summary_content_f1_reward_count_weighted_overlap():
    reward = SummaryContentF1Reward()
    ref = '{"统计":[{"类别":"标签","文本":{"ABC":3}}]}'
    pred = '{"统计":[{"类别":"标签","文本":{"ABC":1}}]}'
    completion = _completion_with_json(pred)
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_domain_token": "BBU",
        "summary_ref": ref,
    }
    # overlap=1, pred_total=1, ref_total=3 -> precision=1, recall=1/3 -> F1=0.5
    assert reward([completion], metadata=[meta])[0] == pytest.approx(0.5)


def test_summary_content_f1_reward_penalizes_spurious_extras():
    reward = SummaryContentF1Reward()
    ref = '{"统计":[{"类别":"标签","文本":{"ABC":1}}]}'
    pred = '{"统计":[{"类别":"标签","文本":{"ABC":1,"DEF":1}}]}'
    completion = _completion_with_json(pred)
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_domain_token": "BBU",
        "summary_ref": ref,
    }
    assert reward([completion], metadata=[meta])[0] == pytest.approx(2.0 / 3.0)


def test_summary_content_f1_reward_rejects_rru_notes():
    reward = SummaryContentF1Reward()
    ref = '{"统计":[{"类别":"标签","文本":{"ABC":1}}]}'
    pred = '{"统计":[{"类别":"标签","文本":{"ABC":1}}],"备注":["不应出现"]}'
    completion = _completion_with_json(pred, domain="RRU")
    meta = {
        "_fusion_source": "rru_summary",
        "_fusion_domain_token": "RRU",
        "summary_ref": ref,
    }
    assert reward([completion], metadata=[meta])[0] == 0.0


def test_summary_content_f1_reward_includes_rru_group_stats():
    reward = SummaryContentF1Reward()
    ref = (
        '{"dataset":"RRU","objects_total":2,"统计":[{"类别":"标签","组":{"1":2}}],'
        '"分组统计":{"1":2}}'
    )
    pred = '{"dataset":"RRU","objects_total":2,"统计":[{"类别":"标签","组":{"1":2}}]}'
    completion = _completion_with_json(pred, domain="RRU")
    meta = {
        "_fusion_source": "rru_summary",
        "_fusion_domain_token": "RRU",
        "summary_ref": ref,
    }
    assert reward([completion], metadata=[meta])[0] == pytest.approx(2.0 / 3.0)


def test_summary_content_f1_reward_irrelevant_is_zero():
    reward = SummaryContentF1Reward()
    completion = "无关图片"
    meta = {"_fusion_source": "irrelevant_summary"}
    assert reward([completion], metadata=[meta])[0] == 0.0


def test_summary_dataset_reward_matches_domain_token():
    reward = SummaryDatasetReward()
    completion = _completion_with_json('{"dataset":"BBU","objects_total":1,"统计":[]}')
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_domain_token": "BBU",
        "summary_ref": '{"dataset":"BBU","objects_total":1,"统计":[]}',
    }
    assert reward([completion], metadata=[meta])[0] == 1.0


def test_summary_dataset_reward_mismatch_is_zero():
    reward = SummaryDatasetReward()
    completion = _completion_with_json('{"dataset":"RRU","objects_total":1,"统计":[]}')
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_domain_token": "BBU",
        "summary_ref": '{"dataset":"BBU","objects_total":1,"统计":[]}',
    }
    assert reward([completion], metadata=[meta])[0] == 0.0


def test_summary_objects_total_reward_dense_decay():
    reward = SummaryObjectsTotalReward()
    meta = {
        "_fusion_source": "bbu_summary",
        "_fusion_domain_token": "BBU",
        "summary_ref": '{"dataset":"BBU","objects_total":3,"统计":[]}',
    }
    completion_same = _completion_with_json('{"dataset":"BBU","objects_total":3,"统计":[]}')
    completion_off_by_one = _completion_with_json('{"dataset":"BBU","objects_total":2,"统计":[]}')
    assert reward([completion_same], metadata=[meta])[0] == 1.0
    assert reward([completion_off_by_one], metadata=[meta])[0] == pytest.approx(0.5)
