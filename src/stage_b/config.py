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
    group_report: bool = True


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
    enable_consistency: bool = False


@dataclass(frozen=True)
class ReflectionConfig:
    prompt_path: Path
    batch_size: int
    max_operations: Optional[int] = None  # Per reflection cycle cap; None = unlimited
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 1024
    max_reflection_length: int = 4096
    token_budget: int = 4096  # Token budget for reflection prompt packing
    # Reflection conflict policy:
    # - require_rule_for_conflicts: when True, each conflicting group (label vs model)
    #   that lacks an explicit add_rule/ask_more_info/abstain_noise entry will receive
    #   a conservative auto rule instead of being treated as pure noise.
    # - treat_keep_conflict_as_noise: when True, legacy behaviour is preserved and
    #   `keep` on conflict groups is logged to the noise queue; when False (default),
    #   such cases are left for the conflict policy to handle (no auto noise).
    require_rule_for_conflicts: bool = True
    treat_keep_conflict_as_noise: bool = False


@dataclass(frozen=True)
class ManualReviewConfig:
    """Gating thresholds for deferring to manual review based on agreement/self-consistency."""

    enabled: bool = True
    # Minimum fraction of candidates sharing the majority verdict (0.0–1.0).
    min_verdict_agreement: float = 0.8
    # Optional minimum average self-consistency for majority verdict candidates (0.0–1.0).
    min_self_consistency: float = 0.8


@dataclass(frozen=True)
class SelectionConfig:
    policy: str
    tie_break: str


@dataclass(frozen=True)
class RunnerConfig:
    epochs: int
    rollout_batch_size: int = 1


@dataclass(frozen=True)
class StageBConfig:
    seed: int
    stage_a_paths: Tuple[Path, ...]
    guidance: GuidanceConfig
    output: OutputConfig
    sampler: SamplerConfig
    reflection: ReflectionConfig
    selection: SelectionConfig
    model: ModelConfig
    runner: RunnerConfig
    manual_review: ManualReviewConfig


def _decode_config(section: Mapping[str, Any]) -> DecodeConfig:
    temperature = float(_require(section, "temperature", "sampler.grid entry"))
    top_p = float(_require(section, "top_p", "sampler.grid entry"))
    max_new_tokens = int(_require(section, "max_new_tokens", "sampler.grid entry"))
    repetition_penalty = float(section.get("repetition_penalty", 1.0))
    no_repeat_ngram_size = section.get("no_repeat_ngram_size")
    no_repeat_ngram_size = (
        int(no_repeat_ngram_size) if no_repeat_ngram_size is not None else None
    )

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
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
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
    group_report = bool(section.get("group_report", True))

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
        group_report=group_report,
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
    enable_consistency = bool(
        _require(section, "enable_consistency", "signals section")
    )
    return SignalsConfig(enable_consistency=enable_consistency)


def _load_reflection(section: Mapping[str, Any]) -> ReflectionConfig:
    prompt_path = Path(_require(section, "prompt_path", "reflection section"))
    batch_size = int(_require(section, "batch_size", "reflection section"))
    if batch_size <= 0:
        raise ValueError("reflection.batch_size must be > 0")

    max_operations_raw = section.get("max_operations")
    max_operations_val: Optional[int]
    if max_operations_raw is None:
        max_operations_val = None
    else:
        max_operations_val = int(max_operations_raw)
        if max_operations_val <= 0:
            raise ValueError("reflection.max_operations must be > 0 if set")
    # Generation parameters with defaults for diversity
    temperature = float(section.get("temperature", 1.0))
    top_p = float(section.get("top_p", 0.95))
    max_new_tokens = int(section.get("max_new_tokens", 1024))
    max_reflection_length = int(section.get("max_reflection_length", 4096))
    if max_reflection_length <= 0:
        raise ValueError("reflection.max_reflection_length must be > 0")
    token_budget = int(section.get("token_budget", max_reflection_length))
    if token_budget <= 0:
        raise ValueError("reflection.token_budget must be > 0")
    require_rule_for_conflicts = bool(
        section.get("require_rule_for_conflicts", True)
    )
    treat_keep_conflict_as_noise = bool(
        section.get("treat_keep_conflict_as_noise", False)
    )

    return ReflectionConfig(
        prompt_path=prompt_path,
        batch_size=batch_size,
        max_operations=max_operations_val,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        max_reflection_length=max_reflection_length,
        token_budget=token_budget,
        require_rule_for_conflicts=require_rule_for_conflicts,
        treat_keep_conflict_as_noise=treat_keep_conflict_as_noise,
    )


def _load_manual_review(
    section: Optional[Mapping[str, Any]],
) -> ManualReviewConfig:
    """Load ManualReviewConfig controlling manual-review gating."""

    if section is None:
        return ManualReviewConfig()

    enabled = bool(section.get("enabled", True))
    min_verdict_agreement = float(section.get("min_verdict_agreement", 0.8))
    min_self_consistency = float(section.get("min_self_consistency", 0.8))

    for name, value in (
        ("min_verdict_agreement", min_verdict_agreement),
        ("min_self_consistency", min_self_consistency),
    ):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"manual_review.{name} must be in [0.0, 1.0]")

    return ManualReviewConfig(
        enabled=enabled,
        min_verdict_agreement=min_verdict_agreement,
        min_self_consistency=min_self_consistency,
    )


def _load_selection(section: Mapping[str, Any]) -> SelectionConfig:
    policy = str(_require(section, "policy", "selection section"))
    if policy not in {"top_label", "top_semantic"}:
        raise ValueError("selection.policy must be 'top_label' or 'top_semantic'")

    tie_break = str(_require(section, "tie_break", "selection section"))
    if tie_break not in {"temperature"}:
        raise ValueError("selection.tie_break must be 'temperature'")

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
    rollout_batch_size = int(section.get("rollout_batch_size", 1))
    if rollout_batch_size <= 0:
        raise ValueError("runner.rollout_batch_size must be > 0")
    return RunnerConfig(epochs=epochs, rollout_batch_size=rollout_batch_size)


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
    reflection = _load_reflection(_require(raw_config, "reflection", "Stage-B config"))
    selection = _load_selection(_require(raw_config, "selection", "Stage-B config"))
    manual_review = _load_manual_review(raw_config.get("manual_review"))
    model = _load_model(_require(raw_config, "model", "Stage-B config"))
    runner = _load_runner(_require(raw_config, "runner", "Stage-B config"))
    seed_value = _load_seed(raw_config)

    return StageBConfig(
        seed=seed_value,
        stage_a_paths=stage_a_paths,
        guidance=guidance,
        output=output,
        sampler=sampler,
        reflection=reflection,
        selection=selection,
        model=model,
        runner=runner,
        manual_review=manual_review,
    )


__all__ = [
    "GuidanceConfig",
    "RunnerConfig",
    "ModelConfig",
    "OutputConfig",
    "ReflectionConfig",
    "ManualReviewConfig",
    "SamplerConfig",
    "SelectionConfig",
    "StageBConfig",
    "load_stage_b_config",
]
