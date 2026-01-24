from typing import Any, cast

import pytest

from src.config.loader import ConfigLoader
from src.rlhf.grpo.rollout_server_config import extract_rollout_server_launch_config


def test_rollout_server_config_smoke_extracts_from_summary_1024():
    cfg = cast(
        dict[str, Any],
        ConfigLoader.load_yaml_with_extends("configs/train/grpo/summary_1024.yaml"),
    )

    launch = extract_rollout_server_launch_config(cfg, visible_gpu_count=6)

    assert launch.connectivity.host == "127.0.0.1"
    assert launch.connectivity.port == 8080
    assert launch.model_path

    assert launch.rollout.vllm_tensor_parallel_size == 1
    assert launch.rollout.vllm_data_parallel_size == 6
    assert launch.rollout.vllm_max_model_len == cfg["global_max_length"]

    args = " ".join(launch.rollout_args)
    assert "--vllm_tensor_parallel_size 1" in args
    assert "--vllm_data_parallel_size 6" in args
    assert "--vllm_max_model_len 12000" in args


def _minimal_cfg(rollout_server: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_max_length": 12000,
        "model": {"model": "output/fake/ckpt"},
        "rlhf": {
            "use_vllm": True,
            "vllm_mode": "server",
            "vllm_server_host": ["127.0.0.1"],
            "vllm_server_port": [8080],
            "vllm_server_timeout": 240,
        },
        "custom": {"extra": {"rollout_server": rollout_server}},
    }


def test_rollout_server_config_rejects_enable_lora_true():
    cfg = _minimal_cfg(
        {
            "vllm_tensor_parallel_size": 1,
            "vllm_data_parallel_size": 1,
            "vllm_max_model_len": 12000,
            "vllm_enable_lora": True,
        }
    )
    with pytest.raises(ValueError, match=r"vllm_enable_lora.*must be false"):
        extract_rollout_server_launch_config(cfg, visible_gpu_count=1)


def test_rollout_server_config_rejects_max_model_len_below_global_max_length():
    cfg = _minimal_cfg(
        {
            "vllm_tensor_parallel_size": 1,
            "vllm_data_parallel_size": 1,
            "vllm_max_model_len": 8192,
            "vllm_enable_lora": False,
        }
    )
    with pytest.raises(
        ValueError, match=r"vllm_max_model_len must be >= global_max_length"
    ):
        extract_rollout_server_launch_config(cfg, visible_gpu_count=1)


def test_rollout_server_config_requires_tp_dp_match_visible_gpus():
    cfg = _minimal_cfg(
        {
            "vllm_tensor_parallel_size": 2,
            "vllm_data_parallel_size": 1,
            "vllm_max_model_len": 12000,
            "vllm_enable_lora": False,
        }
    )
    with pytest.raises(
        ValueError, match=r"vllm_tensor_parallel_size \* vllm_data_parallel_size"
    ):
        extract_rollout_server_launch_config(cfg, visible_gpu_count=6)
