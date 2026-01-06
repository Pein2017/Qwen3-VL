from src.rlhf.grpo.rewards.dense.rewards import (
    DenseAttrWeightedRecallReward,
    DenseLocalizationMeanFBetaReward,
)


def test_dense_attr_weighted_recall_downweights_visibility_and_ocr_bonus_only():
    reward = DenseAttrWeightedRecallReward()

    meta = {
        "_fusion_mode": "dense",
        "_fusion_template": "target_dense_bbu",
        "_fusion_source": "bbu_dense",
    }

    gt_payload: dict[str, object] = {
        "object_1": {
            "desc": "类别=设备,可见性=模糊,品牌=华为,文本=ABC,备注=OK",
            "bbox_2d": [0, 0, 10, 10],
        },
        "object_2": {
            "desc": "类别=站点距离,站点距离=123",
            "bbox_2d": [20, 20, 30, 30],
        },
    }

    pred_json = (
        "<DOMAIN=BBU>, <TASK=DETECTION>\n"
        '{"object_1":{"desc":"类别=设备,可见性=清晰,品牌=华为,文本=XXX,备注=BAD","bbox_2d":[0,0,10,10]},'
        '"object_2":{"desc":"类别=站点距离,站点距离=123","bbox_2d":[20,20,30,30]}}'
    )

    out = reward([pred_json], metadata=[meta], assistant_payload=[gt_payload])
    assert len(out) == 1

    # object_1: core keys are 品牌 (1.0) and 可见性 (0.1). Only 品牌 matches -> 1/1.1
    # object_2: 站点距离 exact match -> 1.0
    expected = (1.0 / 1.1 + 1.0) / 2.0
    assert abs(out[0] - expected) < 1e-6


def test_dense_attr_weighted_recall_station_distance_requires_exact_int_match():
    reward = DenseAttrWeightedRecallReward()

    meta = {
        "_fusion_mode": "dense",
        "_fusion_template": "target_dense_rru",
        "_fusion_source": "rru_dense",
    }

    gt_payload: dict[str, object] = {
        "object_1": {
            "desc": "类别=站点距离,站点距离=123",
            "bbox_2d": [0, 0, 10, 10],
        }
    }

    pred_json = (
        "<DOMAIN=RRU>, <TASK=DETECTION>\n"
        '{"object_1":{"desc":"类别=站点距离,站点距离=124","bbox_2d":[0,0,10,10]}}'
    )

    out = reward([pred_json], metadata=[meta], assistant_payload=[gt_payload])
    assert len(out) == 1
    assert out[0] == 0.0


def test_dense_loc_mean_fbeta_uses_assistant_payload_gt():
    reward = DenseLocalizationMeanFBetaReward()

    meta = {
        "_fusion_mode": "dense",
        "_fusion_template": "target_dense_bbu",
        "_fusion_source": "bbu_dense",
    }

    gt_payload: dict[str, object] = {
        "object_1": {
            "desc": "类别=设备",
            "bbox_2d": [0, 0, 10, 10],
        }
    }

    pred_json = (
        "<DOMAIN=BBU>, <TASK=DETECTION>\n"
        '{"object_1":{"desc":"类别=设备","bbox_2d":[0,0,10,10]}}'
    )

    out = reward([pred_json], metadata=[meta], assistant_payload=[gt_payload])
    assert len(out) == 1
    assert abs(out[0] - 1.0) < 1e-12
