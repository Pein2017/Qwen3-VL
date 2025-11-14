#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration schema for the Stage-B reflection pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml

from .types import DecodeConfig


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required key '{key}' in {context}")
    return mapping[key]


@dataclass(frozen=True)
class GuidanceConfig:
    path: Path
    retention: int


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    run_name: str  # All outputs go under {root}/{run_name}/
    trajectories_path: Path
    selections_jsonl: Path


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str
    torch_dtype: str
    device_map: str


@dataclass(frozen=True)
class SamplerConfig:
    grid: Tuple[DecodeConfig, ...]
    samples_per_decode: int
    format_filter: bool


@dataclass(frozen=True)
class SignalsConfig:
    store_confidence: bool = False
    enable_consistency: bool = False


@dataclass(frozen=True)
class CriticConfig:
    """Configuration for the CriticEngine LLM-based evaluation component."""

    enabled: bool
    prompt_path: Path
    temperature: float
    top_p: float
    max_new_tokens: int
    max_candidates: int
    summary_max_chars: int
    critique_max_chars: int


@dataclass(frozen=True)
class ReflectionConfig:
    prompt_path: Path
    batch_size: int
    # apply_if_delta: Optional[float]  # DEFERRED: Holdout uplift gating not yet implemented
    allow_uncertain: bool
    rapid_mode: bool = False  # Debug flag to disable guardrails
    # Eligibility and budget controls
    eligibility_policy: str = (
        "selected_mismatch_or_all_wrong"  # or "contradictions_only" or "contradictions_or_all_wrong"
    )
    max_operations: Optional[int] = None  # Per reflection cycle cap; None = unlimited
    change_cap_per_epoch: Optional[int] = (
        None  # Per mission per-epoch cap; None = unlimited
    )
    # Generation parameters (keep defaults for diversity)
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 1024
    max_reflection_length: int = 4096
    # P2.13: Token-budget for prompt packing (cap the prompt, not the generated tokens)
    token_budget: int = 1536
    # Optional safety valve for deployments
    all_wrong_strategy: str = "reflect_diagnose"


@dataclass(frozen=True)
class SelectionConfig:
    policy: str
    tie_break: str


@dataclass(frozen=True)
class RunnerConfig:
    epochs: int


@dataclass(frozen=True)
class StageBConfig:
    seed: int
    stage_a_paths: Tuple[Path, ...]
    guidance: GuidanceConfig
    output: OutputConfig
    sampler: SamplerConfig
    signals: SignalsConfig
    reflection: ReflectionConfig
    selection: SelectionConfig
    model: ModelConfig
    runner: RunnerConfig
    critic: CriticConfig


def _decode_config(section: Mapping[str, Any]) -> DecodeConfig:
    temperature = float(_require(section, "temperature", "sampler.grid entry"))
    top_p = float(_require(section, "top_p", "sampler.grid entry"))
    max_new_tokens = int(_require(section, "max_new_tokens", "sampler.grid entry"))

    seed_value: Optional[int] = None
    if "seed" in section and section["seed"] is not None:
        seed_value = int(section["seed"])

    stop_tokens: Sequence[str]
    raw_stop = section.get("stop")
    if raw_stop is None:
        stop_tokens = ()
    elif isinstance(raw_stop, Sequence):
        stop_tokens = tuple(str(token) for token in raw_stop)
    else:
        raise TypeError("sampler.grid entry 'stop' must be a sequence or null")

    return DecodeConfig(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed=seed_value,
        stop=tuple(stop_tokens),
    )


def _load_sampler(section: Mapping[str, Any]) -> SamplerConfig:
    grid_section = _require(section, "grid", "sampler section")
    if not isinstance(grid_section, Sequence) or not grid_section:
        raise ValueError("sampler.grid must be a non-empty sequence")
    grid = tuple(_decode_config(entry) for entry in grid_section)

    samples_per_decode = int(_require(section, "samples_per_decode", "sampler section"))
    if samples_per_decode <= 0:
        raise ValueError("sampler.samples_per_decode must be > 0")

    format_filter = bool(_require(section, "format_filter", "sampler section"))

    return SamplerConfig(
        grid=grid,
        samples_per_decode=samples_per_decode,
        format_filter=format_filter,
    )


def _load_stage_a_paths(raw: Any) -> Tuple[Path, ...]:
    if raw is None:
        raise ValueError("stage_a_paths must be provided")
    if isinstance(raw, (str, Path)):
        return (Path(raw),)
    if isinstance(raw, Sequence):
        return tuple(Path(item) for item in raw)
    raise TypeError("stage_a_paths must be string or sequence")


def _load_guidance(section: Mapping[str, Any]) -> GuidanceConfig:
    path = Path(_require(section, "path", "guidance section"))
    retention = int(_require(section, "retention", "guidance section"))
    if retention <= 0:
        raise ValueError("guidance.retention must be > 0")

    return GuidanceConfig(path=path, retention=retention)


def _load_output(section: Mapping[str, Any]) -> OutputConfig:
    root = Path(_require(section, "root", "output section"))
    run_name = str(_require(section, "run_name", "output section"))

    # Base directory for this run: {root}/{run_name}
    # Mission-specific paths will be created dynamically in runner
    # These are placeholders - actual paths will be {root}/{run_name}/{mission}/*
    base_dir = root / run_name
    trajectories_path = (
        base_dir / "trajectories.jsonl"
    )  # Placeholder, will be mission-specific
    selections_jsonl = (
        base_dir / "selections.jsonl"
    )  # Placeholder, will be mission-specific

    return OutputConfig(
        root=root,
        run_name=run_name,
        trajectories_path=trajectories_path,
        selections_jsonl=selections_jsonl,
    )


def _load_model(section: Mapping[str, Any]) -> ModelConfig:
    model_name = str(_require(section, "model_name_or_path", "model section"))
    dtype_raw = _require(section, "torch_dtype", "model section")
    dtype_value = str(dtype_raw)
    device_map_raw = _require(section, "device_map", "model section")
    device_map_value = str(device_map_raw)
    return ModelConfig(
        model_name_or_path=model_name,
        torch_dtype=dtype_value,
        device_map=device_map_value,
    )


def _load_signals(section: Mapping[str, Any]) -> SignalsConfig:
    store_confidence = bool(_require(section, "store_confidence", "signals section"))
    enable_consistency = bool(
        _require(section, "enable_consistency", "signals section")
    )
    return SignalsConfig(
        store_confidence=store_confidence,
        enable_consistency=enable_consistency,
    )


def _load_critic(section: Optional[Mapping[str, Any]]) -> CriticConfig:
    """Load CriticConfig with sensible defaults."""
    if section is None:
        # Default: critic disabled
        return CriticConfig(
            enabled=False,
            prompt_path=Path("configs/prompts/stage_b_critic.txt"),
            temperature=0.2,
            top_p=0.9,
            max_new_tokens=512,
            max_candidates=6,
            summary_max_chars=200,
            critique_max_chars=200,
        )

    enabled = bool(section.get("enabled", True))
    prompt_path = Path(_require(section, "prompt_path", "critic section"))

    temperature = float(section.get("temperature", 0.2))
    if not (0.1 <= temperature <= 0.3):
        raise ValueError("critic.temperature must be in range [0.1, 0.3]")

    top_p = float(section.get("top_p", 0.9))
    max_new_tokens = int(section.get("max_new_tokens", 512))
    max_candidates = int(section.get("max_candidates", 6))
    if max_candidates > 6:
        raise ValueError("critic.max_candidates must be <= 6")

    summary_max_chars = int(section.get("summary_max_chars", 200))
    critique_max_chars = int(section.get("critique_max_chars", 200))

    # Validate prompt_path exists if critic is enabled
    if enabled and not prompt_path.exists():
        raise FileNotFoundError(f"critic.prompt_path does not exist: {prompt_path}")

    return CriticConfig(
        enabled=enabled,
        prompt_path=prompt_path,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        max_candidates=max_candidates,
        summary_max_chars=summary_max_chars,
        critique_max_chars=critique_max_chars,
    )


def _load_reflection(section: Mapping[str, Any]) -> ReflectionConfig:
    prompt_path = Path(_require(section, "prompt_path", "reflection section"))
    batch_size = int(_require(section, "batch_size", "reflection section"))
    if batch_size <= 0:
        raise ValueError("reflection.batch_size must be > 0")

    # DEFERRED: apply_if_delta and holdout uplift gating not yet implemented
    # Ignore apply_if_delta in config if present
    # apply_if_delta_raw = section.get("apply_if_delta")
    # apply_if_delta_val: Optional[float]
    # if apply_if_delta_raw is None:
    #     apply_if_delta_val = None
    # else:
    #     apply_if_delta_val = float(apply_if_delta_raw)

    allow_uncertain = bool(_require(section, "allow_uncertain", "reflection section"))

    # rapid_mode flag for disabling guardrails
    rapid_mode = bool(section.get("rapid_mode", False))

    # Eligibility & budgets
    eligibility_policy = str(
        section.get("eligibility_policy", "selected_mismatch_or_all_wrong")
    )
    if eligibility_policy not in {
        "selected_mismatch_or_all_wrong",
        "contradictions_only",
        "contradictions_or_all_wrong",
    }:
        raise ValueError(
            "reflection.eligibility_policy must be 'selected_mismatch_or_all_wrong', 'contradictions_only', or 'contradictions_or_all_wrong'"
        )
    max_operations_raw = section.get("max_operations")
    max_operations_val: Optional[int]
    if max_operations_raw is None:
        max_operations_val = None
    else:
        max_operations_val = int(max_operations_raw)
        if max_operations_val <= 0:
            raise ValueError("reflection.max_operations must be > 0 if set")
    change_cap_raw = section.get("change_cap_per_epoch")
    change_cap_val: Optional[int]
    if change_cap_raw is None:
        change_cap_val = None
    else:
        change_cap_val = int(change_cap_raw)
        if change_cap_val <= 0:
            raise ValueError("reflection.change_cap_per_epoch must be > 0 if set")

    # Generation parameters with defaults for diversity
    temperature = float(section.get("temperature", 1.0))
    top_p = float(section.get("top_p", 0.95))
    max_new_tokens = int(section.get("max_new_tokens", 1024))
    max_reflection_length = int(section.get("max_reflection_length", 4096))
    if max_reflection_length <= 0:
        raise ValueError("reflection.max_reflection_length must be > 0")

    # P2.13: token budget for prompt packing
    token_budget = int(section.get("token_budget", 1536))
    if token_budget <= 0:
        raise ValueError("reflection.token_budget must be > 0")

    # Optional strategy for all-wrong handling
    all_wrong_strategy = str(section.get("all_wrong_strategy", "reflect_diagnose"))
    if all_wrong_strategy not in {"reflect_diagnose", "manual_review"}:
        raise ValueError("reflection.all_wrong_strategy must be 'reflect_diagnose' or 'manual_review'")

    return ReflectionConfig(
        prompt_path=prompt_path,
        batch_size=batch_size,
        # apply_if_delta=apply_if_delta_val,  # DEFERRED: not in schema
        allow_uncertain=allow_uncertain,
        rapid_mode=rapid_mode,
        eligibility_policy=eligibility_policy,
        max_operations=max_operations_val,
        change_cap_per_epoch=change_cap_val,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        max_reflection_length=max_reflection_length,
        token_budget=token_budget,
        all_wrong_strategy=all_wrong_strategy,
    )


def _load_selection(section: Mapping[str, Any]) -> SelectionConfig:
    policy = str(_require(section, "policy", "selection section"))
    if policy not in {"top_label", "top_semantic"}:
        raise ValueError("selection.policy must be 'top_label' or 'top_semantic'")

    tie_break = str(_require(section, "tie_break", "selection section"))
    if tie_break not in {"confidence", "temperature"}:
        raise ValueError("selection.tie_break must be 'confidence' or 'temperature'")

    return SelectionConfig(policy=policy, tie_break=tie_break)



def _load_seed(raw_config: Mapping[str, Any]) -> int:
    seed_value = raw_config.get("seed")
    if seed_value is None:
        raise KeyError("Stage-B config must include top-level 'seed'")
    return int(seed_value)


def _load_runner(section: Mapping[str, Any]) -> RunnerConfig:
    epochs = int(_require(section, "epochs", "runner section"))
    if epochs <= 0:
        raise ValueError("runner.epochs must be > 0")
    return RunnerConfig(epochs=epochs)


def load_stage_b_config(path: str | Path) -> StageBConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw_config = yaml.safe_load(fh) or {}

    if not isinstance(raw_config, MutableMapping):
        raise TypeError("Stage-B config must be a mapping")

    stage_a_paths = _load_stage_a_paths(
        _require(raw_config, "stage_a_paths", "Stage-B config")
    )
    guidance = _load_guidance(_require(raw_config, "guidance", "Stage-B config"))
    output = _load_output(_require(raw_config, "output", "Stage-B config"))
    sampler = _load_sampler(_require(raw_config, "sampler", "Stage-B config"))
    signals = _load_signals(_require(raw_config, "signals", "Stage-B config"))
    reflection = _load_reflection(_require(raw_config, "reflection", "Stage-B config"))
    selection = _load_selection(_require(raw_config, "selection", "Stage-B config"))
    model = _load_model(_require(raw_config, "model", "Stage-B config"))
    runner = _load_runner(_require(raw_config, "runner", "Stage-B config"))
    critic = _load_critic(raw_config.get("critic"))
    seed_value = _load_seed(raw_config)

    return StageBConfig(
        seed=seed_value,
        stage_a_paths=stage_a_paths,
        guidance=guidance,
        output=output,
        sampler=sampler,
        signals=signals,
        reflection=reflection,
        selection=selection,
        model=model,
        runner=runner,
        critic=critic,
    )


__all__ = [
    "CriticConfig",
    "GuidanceConfig",
    "RunnerConfig",
    "ModelConfig",
    "OutputConfig",
    "ReflectionConfig",
    "SamplerConfig",
    "SelectionConfig",
    "SignalsConfig",
    "StageBConfig",
    "load_stage_b_config",
]
