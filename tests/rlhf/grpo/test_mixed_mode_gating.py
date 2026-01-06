from src.rlhf.grpo.rewards.dense.rewards import DenseLocalizationMeanFBetaReward
from src.rlhf.grpo.rewards.summary.rewards import SummaryFormatReward


def test_dense_rewards_noop_on_summary_mode():
    reward = DenseLocalizationMeanFBetaReward()
    meta: dict[str, object] = {"_fusion_mode": "summary"}
    obj_payload: dict[str, object] = {"desc": "类别=设备", "bbox_2d": [0, 0, 10, 10]}
    gt_payload: dict[str, object] = {"object_1": obj_payload}
    pred = "<DOMAIN=BBU>, <TASK=DETECTION>\n{}"
    out = reward([pred], metadata=[meta], assistant_payload=[gt_payload])
    assert out == [0.0]


def test_summary_rewards_noop_on_dense_mode():
    reward = SummaryFormatReward()
    meta: dict[str, object] = {"_fusion_mode": "dense"}
    pred = "<DOMAIN=BBU>, <TASK=SUMMARY>\n{}"
    out = reward([pred], metadata=[meta])
    assert out == [0.0]
