"""Server-mode GRPO rollout configuration helpers.

This module defines and validates the Qwen3-VL-owned config surface under
`custom.extra.rollout_server`, consumed by the server-only rollout launcher.

Motivation:
- In ms-swift GRPO server mode, the trainer (`scripts/train.sh` → `src/sft.py`) connects
  to an external `swift rollout` server via `rlhf.vllm_server_host`/`rlhf.vllm_server_port`.
- Server-only vLLM knobs (TP/DP/max_model_len/gpu_memory_utilization/...) must be
  provided to the rollout server process, not the trainer.

This file intentionally keeps the rollout-server config parsing separate from
`TrainingConfig` because `custom.extra` is an explicit escape hatch for
launcher-owned extensions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast


def _require_mapping(value: object, field_path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_path} must be a mapping")
    # Treat as Mapping[str, Any] after validation.
    return value  # type: ignore[return-value]


def _as_int(value: object, field_path: str) -> int:
    if value is None:
        raise ValueError(f"{field_path} is required")
    if isinstance(value, bool):
        raise TypeError(f"{field_path} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field_path} must be an integer, got {value!r}")
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_path} is required")
        try:
            return int(stripped)
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"{field_path} must be an integer, got {value!r}") from exc
    raise TypeError(f"{field_path} must be an integer, got {type(value).__name__}")


def _as_optional_int(value: object, field_path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_path} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field_path} must be an integer, got {value!r}")
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"{field_path} must be an integer, got {value!r}") from exc
    raise TypeError(f"{field_path} must be an integer, got {type(value).__name__}")


def _as_optional_float(value: object, field_path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_path} must be a float")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"{field_path} must be a float, got {value!r}") from exc
    raise TypeError(f"{field_path} must be a float, got {type(value).__name__}")


def _as_optional_bool(value: object, field_path: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in ("true", "1", "yes", "y"):
            return True
        if normalized in ("false", "0", "no", "n"):
            return False
    raise TypeError(f"{field_path} must be a boolean")


def _normalize_to_list(value: object, field_path: str) -> list[object]:
    if value is None:
        raise ValueError(f"{field_path} is required")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(cast(Sequence[object], value))
    return [value]


@dataclass(frozen=True)
class RolloutServerConnectivity:
    """Trainer-to-server connectivity settings (ms-swift `RLHFArguments`)."""

    host: str
    port: int

    @classmethod
    def from_training_config(
        cls, raw_config: Mapping[str, Any]
    ) -> "RolloutServerConnectivity":
        rlhf = _require_mapping(raw_config.get("rlhf") or {}, "rlhf")

        hosts_raw = rlhf.get("vllm_server_host")
        hosts = _normalize_to_list(hosts_raw, "rlhf.vllm_server_host")
        if len(hosts) != 1:
            raise ValueError(
                "rlhf.vllm_server_host must contain exactly 1 entry for single-node server-mode rollout"
            )
        host = str(hosts[0]).strip()
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError(
                "rlhf.vllm_server_host[0] must be one of {'127.0.0.1', 'localhost'} for local-only rollout"
            )

        ports_raw = rlhf.get("vllm_server_port")
        ports = _normalize_to_list(ports_raw, "rlhf.vllm_server_port")
        if len(ports) != 1:
            raise ValueError(
                "rlhf.vllm_server_port must contain exactly 1 entry for single-node server-mode rollout"
            )
        port = _as_int(ports[0], "rlhf.vllm_server_port[0]")
        if port <= 0 or port > 65535:
            raise ValueError("rlhf.vllm_server_port[0] must be in [1, 65535]")

        return cls(host=host, port=port)


@dataclass(frozen=True)
class RolloutServerConfig:
    """Qwen3-VL-owned rollout server settings (consumed by the launcher)."""

    vllm_tensor_parallel_size: int
    vllm_data_parallel_size: int
    vllm_max_model_len: int

    # Common optional knobs we support as passthrough flags to `swift rollout`.
    vllm_gpu_memory_utilization: float | None = None
    vllm_max_num_seqs: int | None = None
    vllm_enable_prefix_caching: bool | None = None
    vllm_disable_custom_all_reduce: bool | None = None
    vllm_enforce_eager: bool | None = None
    vllm_limit_mm_per_prompt: object | None = (
        None  # vLLM accepts JSON for multimodal limits.
    )
    vllm_enable_lora: bool | None = None
    vllm_max_lora_rank: int | None = None
    vllm_use_async_engine: bool | None = None
    vllm_engine_kwargs: object | None = None
    vllm_mm_processor_cache_gb: float | None = None
    vllm_disable_cascade_attn: bool | None = None
    vllm_quantization: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RolloutServerConfig":
        if not isinstance(raw, Mapping):
            raise TypeError("custom.extra.rollout_server must be a mapping")

        allowed_keys = {
            "vllm_tensor_parallel_size",
            "vllm_data_parallel_size",
            "vllm_gpu_memory_utilization",
            "vllm_max_model_len",
            "vllm_max_num_seqs",
            "vllm_enable_prefix_caching",
            "vllm_disable_custom_all_reduce",
            "vllm_enforce_eager",
            "vllm_limit_mm_per_prompt",
            "vllm_enable_lora",
            "vllm_max_lora_rank",
            "vllm_use_async_engine",
            "vllm_engine_kwargs",
            "vllm_mm_processor_cache_gb",
            "vllm_disable_cascade_attn",
            "vllm_quantization",
        }
        unknown = sorted(set(raw.keys()) - allowed_keys)
        if unknown:
            rendered = ", ".join(str(k) for k in unknown)
            raise ValueError(
                "custom.extra.rollout_server contains unsupported keys: "
                f"{rendered}. Supported keys: {', '.join(sorted(allowed_keys))}"
            )

        tp = _as_int(
            raw.get("vllm_tensor_parallel_size"),
            "custom.extra.rollout_server.vllm_tensor_parallel_size",
        )
        dp = _as_int(
            raw.get("vllm_data_parallel_size"),
            "custom.extra.rollout_server.vllm_data_parallel_size",
        )
        if tp <= 0:
            raise ValueError(
                "custom.extra.rollout_server.vllm_tensor_parallel_size must be > 0"
            )
        if dp <= 0:
            raise ValueError(
                "custom.extra.rollout_server.vllm_data_parallel_size must be > 0"
            )

        max_model_len = _as_int(
            raw.get("vllm_max_model_len"),
            "custom.extra.rollout_server.vllm_max_model_len",
        )
        if max_model_len <= 0:
            raise ValueError(
                "custom.extra.rollout_server.vllm_max_model_len must be > 0"
            )

        enable_lora = _as_optional_bool(
            raw.get("vllm_enable_lora"), "custom.extra.rollout_server.vllm_enable_lora"
        )
        if enable_lora is True:
            raise ValueError(
                "custom.extra.rollout_server.vllm_enable_lora must be false for this workflow (vLLM LoRA is forbidden)"
            )

        max_lora_rank = _as_optional_int(
            raw.get("vllm_max_lora_rank"),
            "custom.extra.rollout_server.vllm_max_lora_rank",
        )
        if max_lora_rank is not None:
            raise ValueError(
                "custom.extra.rollout_server.vllm_max_lora_rank is unsupported in this workflow (vLLM LoRA is forbidden)"
            )

        gpu_mem = _as_optional_float(
            raw.get("vllm_gpu_memory_utilization"),
            "custom.extra.rollout_server.vllm_gpu_memory_utilization",
        )
        if gpu_mem is not None and not (0.0 < gpu_mem <= 1.0):
            raise ValueError(
                "custom.extra.rollout_server.vllm_gpu_memory_utilization must be in (0, 1]"
            )

        quant = raw.get("vllm_quantization")
        if quant is not None and not isinstance(quant, str):
            raise TypeError(
                "custom.extra.rollout_server.vllm_quantization must be a string"
            )

        return cls(
            vllm_tensor_parallel_size=tp,
            vllm_data_parallel_size=dp,
            vllm_max_model_len=max_model_len,
            vllm_gpu_memory_utilization=gpu_mem,
            vllm_max_num_seqs=_as_optional_int(
                raw.get("vllm_max_num_seqs"),
                "custom.extra.rollout_server.vllm_max_num_seqs",
            ),
            vllm_enable_prefix_caching=_as_optional_bool(
                raw.get("vllm_enable_prefix_caching"),
                "custom.extra.rollout_server.vllm_enable_prefix_caching",
            ),
            vllm_disable_custom_all_reduce=_as_optional_bool(
                raw.get("vllm_disable_custom_all_reduce"),
                "custom.extra.rollout_server.vllm_disable_custom_all_reduce",
            ),
            vllm_enforce_eager=_as_optional_bool(
                raw.get("vllm_enforce_eager"),
                "custom.extra.rollout_server.vllm_enforce_eager",
            ),
            vllm_limit_mm_per_prompt=raw.get("vllm_limit_mm_per_prompt"),
            vllm_enable_lora=enable_lora,
            vllm_max_lora_rank=None,
            vllm_use_async_engine=_as_optional_bool(
                raw.get("vllm_use_async_engine"),
                "custom.extra.rollout_server.vllm_use_async_engine",
            ),
            vllm_engine_kwargs=raw.get("vllm_engine_kwargs"),
            vllm_mm_processor_cache_gb=_as_optional_float(
                raw.get("vllm_mm_processor_cache_gb"),
                "custom.extra.rollout_server.vllm_mm_processor_cache_gb",
            ),
            vllm_disable_cascade_attn=_as_optional_bool(
                raw.get("vllm_disable_cascade_attn"),
                "custom.extra.rollout_server.vllm_disable_cascade_attn",
            ),
            vllm_quantization=quant,
        )


@dataclass(frozen=True)
class RolloutServerLaunchConfig:
    """Fully-resolved inputs needed to launch `swift rollout` for GRPO."""

    model_path: str
    connectivity: RolloutServerConnectivity
    rollout: RolloutServerConfig
    rollout_args: list[str]


def extract_rollout_server_launch_config(
    raw_config: Mapping[str, Any],
    *,
    visible_gpu_count: int,
) -> RolloutServerLaunchConfig:
    """Extract and validate rollout-server launch configuration from a merged YAML mapping."""

    if visible_gpu_count <= 0:
        raise ValueError("visible_gpu_count must be > 0")

    rlhf = _require_mapping(raw_config.get("rlhf") or {}, "rlhf")
    if rlhf.get("use_vllm") is not True:
        raise ValueError("rlhf.use_vllm must be true for server-mode rollout")
    if rlhf.get("vllm_mode") != "server":
        raise ValueError(
            f"rlhf.vllm_mode must be 'server', got {rlhf.get('vllm_mode')!r}"
        )

    connectivity = RolloutServerConnectivity.from_training_config(raw_config)

    model_block = _require_mapping(raw_config.get("model") or {}, "model")
    model_path = str(model_block.get("model") or "").strip()
    if not model_path:
        raise ValueError("model.model must be set (rollout model path)")

    custom = _require_mapping(raw_config.get("custom") or {}, "custom")
    extra = _require_mapping(custom.get("extra") or {}, "custom.extra")
    rollout_raw = extra.get("rollout_server")
    if rollout_raw is None:
        raise ValueError(
            "custom.extra.rollout_server is required when rlhf.vllm_mode == 'server'"
        )
    rollout = RolloutServerConfig.from_mapping(
        _require_mapping(rollout_raw, "custom.extra.rollout_server")
    )

    required_device_count = (
        rollout.vllm_tensor_parallel_size * rollout.vllm_data_parallel_size
    )
    if required_device_count != visible_gpu_count:
        raise ValueError(
            "custom.extra.rollout_server requires vllm_tensor_parallel_size * vllm_data_parallel_size "
            f"== visible rollout GPU count, got {required_device_count} != {visible_gpu_count}"
        )

    global_max_length_raw = raw_config.get("global_max_length")
    if global_max_length_raw is not None:
        global_max_length = _as_int(global_max_length_raw, "global_max_length")
        if rollout.vllm_max_model_len < global_max_length:
            raise ValueError(
                "custom.extra.rollout_server.vllm_max_model_len must be >= global_max_length "
                f"({rollout.vllm_max_model_len} < {global_max_length})"
            )

    # Build vLLM args for `swift rollout`. For structured objects, we JSON-encode to
    # match the behavior of the legacy combined launcher.
    args: list[str] = [
        "--vllm_tensor_parallel_size",
        str(rollout.vllm_tensor_parallel_size),
        "--vllm_data_parallel_size",
        str(rollout.vllm_data_parallel_size),
        "--vllm_max_model_len",
        str(rollout.vllm_max_model_len),
    ]

    optional: dict[str, tuple[object | None, str]] = {
        "vllm_gpu_memory_utilization": (
            rollout.vllm_gpu_memory_utilization,
            "--vllm_gpu_memory_utilization",
        ),
        "vllm_max_num_seqs": (rollout.vllm_max_num_seqs, "--vllm_max_num_seqs"),
        "vllm_enable_prefix_caching": (
            rollout.vllm_enable_prefix_caching,
            "--vllm_enable_prefix_caching",
        ),
        "vllm_disable_custom_all_reduce": (
            rollout.vllm_disable_custom_all_reduce,
            "--vllm_disable_custom_all_reduce",
        ),
        "vllm_enforce_eager": (rollout.vllm_enforce_eager, "--vllm_enforce_eager"),
        "vllm_limit_mm_per_prompt": (
            rollout.vllm_limit_mm_per_prompt,
            "--vllm_limit_mm_per_prompt",
        ),
        # LoRA is forbidden. If explicitly provided, it must be false.
        "vllm_enable_lora": (rollout.vllm_enable_lora, "--vllm_enable_lora"),
        "vllm_use_async_engine": (
            rollout.vllm_use_async_engine,
            "--vllm_use_async_engine",
        ),
        "vllm_engine_kwargs": (rollout.vllm_engine_kwargs, "--vllm_engine_kwargs"),
        "vllm_mm_processor_cache_gb": (
            rollout.vllm_mm_processor_cache_gb,
            "--vllm_mm_processor_cache_gb",
        ),
        "vllm_disable_cascade_attn": (
            rollout.vllm_disable_cascade_attn,
            "--vllm_disable_cascade_attn",
        ),
        "vllm_quantization": (rollout.vllm_quantization, "--vllm_quantization"),
    }

    for key, (value, flag) in optional.items():
        if value is None:
            continue
        if isinstance(value, (Mapping, Sequence)) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            rendered = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        # Include the key name in errors if something ends up being empty.
        if not rendered:
            raise ValueError(
                f"custom.extra.rollout_server.{key} must not be empty when provided"
            )
        args.extend([flag, rendered])

    return RolloutServerLaunchConfig(
        model_path=model_path,
        connectivity=connectivity,
        rollout=rollout,
        rollout_args=args,
    )


__all__ = [
    "RolloutServerConnectivity",
    "RolloutServerConfig",
    "RolloutServerLaunchConfig",
    "extract_rollout_server_launch_config",
]
