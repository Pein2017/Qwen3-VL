from typing import Any, cast

from src.config.loader import ConfigLoader


def test_smoke_dense_summary_grpo_config_loads():
    cfg = cast(
        dict[str, Any],
        ConfigLoader.load_yaml_with_extends("configs/smoke/grpo_dense_summary_mixed.yaml"),
    )
    custom = cast(dict[str, Any], cfg.get("custom") or {})
    rlhf = cast(dict[str, Any], cfg.get("rlhf") or {})

    assert (
        custom.get("fusion_config")
        == "configs/fusion/variants/bbu_rru_dense_grpo_mixed_2048.yaml"
    )
    assert rlhf.get("rlhf_type") == "grpo"

    reward_funcs = cast(list[str], rlhf.get("reward_funcs") or [])

    # Dense rewards
    assert "dense.loc_mean_fbeta" in reward_funcs
    assert "dense.attr_weighted_recall" in reward_funcs

    # Summary rewards are present for mixed-mode regularization
    assert "summary.format" in reward_funcs
    assert "summary.group_stats_presence" in reward_funcs


def test_smoke_dense_summary_grpo_tiny_config_loads():
    cfg = cast(
        dict[str, Any],
        ConfigLoader.load_yaml_with_extends(
            "configs/smoke/grpo_dense_summary_mixed_tiny.yaml"
        ),
    )
    custom = cast(dict[str, Any], cfg.get("custom") or {})
    training = cast(dict[str, Any], cfg.get("training") or {})

    assert (
        custom.get("fusion_config")
        == "configs/fusion/variants/bbu_rru_dense_grpo_mixed_2048_tiny.yaml"
    )
    assert training.get("logging_steps") == 1
    assert training.get("eval_steps") == 1
