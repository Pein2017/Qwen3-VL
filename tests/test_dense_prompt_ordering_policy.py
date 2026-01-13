from src.config.prompts import build_dense_system_prompt, build_dense_user_prompt


def test_dense_prompt_default_policy_mentions_aabb_center() -> None:
    prompt = build_dense_system_prompt("standard")
    assert "AABB" in prompt
    assert "中心坐标" in prompt
    # Center policy should not claim bbox/poly/line reference points are used for ordering.
    assert "bbox_2d 排序参考点" not in prompt


def test_dense_prompt_reference_policy_mentions_reference_points() -> None:
    prompt = build_dense_system_prompt(
        "standard", object_ordering_policy="reference_tlbr"
    )
    assert "bbox_2d 排序参考点" in prompt
    assert "poly 排序参考点" in prompt
    assert "line 排序参考点" in prompt

    user = build_dense_user_prompt("center_tlbr")
    assert "中心点" in user
