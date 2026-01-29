from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from src.config.loader import ConfigLoader
from src.rlhf.grpo.rollout_server_config import extract_rollout_server_launch_config


def _write_yaml(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def test_grpo_batch_plan_expands_into_legacy_knobs(tmp_path):
    cfg_path = str(tmp_path / "cfg.yaml")
    _write_yaml(
        cfg_path,
        {
            "training": {},
            "rlhf": {"num_generations": 4},
            "custom": {
                "grpo": {
                    "batch_plan": {
                        "enabled": True,
                        "per_device_train_batch_size": 8,
                        "per_device_eval_batch_size": 8,
                        "unified_batch_size": 48,
                        "rollout_server": {
                            "force_vllm_tensor_parallel_size": 1,
                            "force_vllm_data_parallel_size": 2,
                            "max_num_seqs_per_gpu": 4,
                        },
                    }
                }
            },
        },
    )

    resolved = cast(dict[str, Any], ConfigLoader.load_yaml_with_extends(cfg_path))
    training = cast(dict[str, Any], resolved["training"])
    rlhf = cast(dict[str, Any], resolved["rlhf"])
    custom = cast(dict[str, Any], resolved["custom"])
    extra = cast(dict[str, Any], custom["extra"])
    rollout_server = cast(dict[str, Any], extra["rollout_server"])

    assert training["per_device_train_batch_size"] == 8
    assert training["per_device_eval_batch_size"] == 8
    assert training["effective_batch_size"] == 48
    assert rlhf["generation_batch_size"] == 48

    assert rollout_server["vllm_tensor_parallel_size"] == 1
    assert rollout_server["vllm_data_parallel_size"] == 2
    assert rollout_server["vllm_max_num_seqs"] == 4


def test_grpo_batch_plan_rejects_conflicting_legacy_knobs(tmp_path):
    cfg_path = str(tmp_path / "cfg.yaml")
    _write_yaml(
        cfg_path,
        {
            "training": {"effective_batch_size": 32},
            "custom": {
                "grpo": {
                    "batch_plan": {
                        "enabled": True,
                        "per_device_train_batch_size": 8,
                        "unified_batch_size": 48,
                    }
                }
            },
        },
    )

    with pytest.raises(
        ValueError,
        match=r"training\.effective_batch_size=.*conflicts with .*unified_batch_size",
    ):
        _ = ConfigLoader.load_yaml_with_extends(cfg_path)


def test_grpo_batch_plan_world_size_divisibility_validation(tmp_path, monkeypatch):
    cfg_path = str(tmp_path / "cfg.yaml")
    _write_yaml(
        cfg_path,
        {
            "custom": {
                "grpo": {
                    "batch_plan": {
                        "enabled": True,
                        "per_device_train_batch_size": 8,
                        "unified_batch_size": 64,
                    }
                }
            },
        },
    )

    # Outside a distributed context, WORLD_SIZE may be unknown; validation is deferred.
    _ = ConfigLoader.load_yaml_with_extends(cfg_path)

    monkeypatch.setenv("WORLD_SIZE", "6")
    with pytest.raises(
        ValueError, match=r"custom\.grpo\.batch_plan\.unified_batch_size.*WORLD_SIZE"
    ):
        _ = ConfigLoader.load_yaml_with_extends(cfg_path)


def test_grpo_batch_plan_parses_enabled_string_false_as_disabled(tmp_path):
    cfg_path = str(tmp_path / "cfg.yaml")
    _write_yaml(
        cfg_path,
        {
            "custom": {
                "grpo": {
                    "batch_plan": {
                        "enabled": "false",
                        "per_device_train_batch_size": 8,
                        "unified_batch_size": 48,
                    }
                }
            }
        },
    )

    with pytest.raises(
        ValueError, match=r"custom\.grpo\.batch_plan is configured but disabled"
    ):
        _ = ConfigLoader.load_yaml_with_extends(cfg_path)


def test_grpo_batch_plan_rejects_non_integer_types(tmp_path):
    cfg_path = str(tmp_path / "cfg.yaml")
    _write_yaml(
        cfg_path,
        {
            "custom": {
                "grpo": {
                    "batch_plan": {
                        "enabled": True,
                        "per_device_train_batch_size": {},
                        "unified_batch_size": 48,
                    }
                }
            }
        },
    )

    with pytest.raises(
        TypeError,
        match=r"custom\.grpo\.batch_plan\.per_device_train_batch_size must be an integer",
    ):
        _ = ConfigLoader.load_yaml_with_extends(cfg_path)


def test_grpo_batch_plan_rejects_non_integer_float(tmp_path):
    cfg_path = str(tmp_path / "cfg.yaml")
    _write_yaml(
        cfg_path,
        {
            "custom": {
                "grpo": {
                    "batch_plan": {
                        "enabled": True,
                        "per_device_train_batch_size": 1.5,
                        "unified_batch_size": 48,
                    }
                }
            }
        },
    )

    with pytest.raises(
        ValueError,
        match=r"custom\.grpo\.batch_plan\.per_device_train_batch_size must be an integer",
    ):
        _ = ConfigLoader.load_yaml_with_extends(cfg_path)


def test_grpo_batch_plan_forces_rollout_server_config_for_launcher(tmp_path):
    cfg_path = str(tmp_path / "cfg.yaml")
    _write_yaml(
        cfg_path,
        {
            "global_max_length": 12000,
            "model": {"model": "output/fake/ckpt"},
            "rlhf": {
                "use_vllm": True,
                "vllm_mode": "server",
                "vllm_server_host": ["127.0.0.1"],
                "vllm_server_port": [8080],
                "vllm_server_timeout": 240,
            },
            "custom": {
                "extra": {"rollout_server": {"vllm_max_model_len": 12000}},
                "grpo": {
                    "batch_plan": {
                        "enabled": True,
                        "per_device_train_batch_size": 8,
                        "unified_batch_size": 48,
                        "rollout_server": {
                            "force_vllm_tensor_parallel_size": 1,
                            "force_vllm_data_parallel_size": 2,
                            "max_num_seqs_per_gpu": 4,
                        },
                    }
                },
            },
        },
    )

    resolved = cast(dict[str, Any], ConfigLoader.load_yaml_with_extends(cfg_path))
    launch = extract_rollout_server_launch_config(resolved, visible_gpu_count=2)

    assert launch.rollout.vllm_tensor_parallel_size == 1
    assert launch.rollout.vllm_data_parallel_size == 2
    assert launch.rollout.vllm_max_num_seqs == 4

    args = " ".join(launch.rollout_args)
    assert "--vllm_tensor_parallel_size 1" in args
    assert "--vllm_data_parallel_size 2" in args
    assert "--vllm_max_num_seqs 4" in args
