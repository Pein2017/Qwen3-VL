#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration schema for the Stage-B rule-search pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

from src.prompts.domain_packs import get_domain_pack

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
    reset_on_rerun: bool = False


@dataclass(frozen=True)
class OutputConfig:
    root: Path  # Base output directory
    run_name: str  # All outputs go under {root}/{mission_name}/{run_name}/


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str
    torch_dtype: str
    device_map: str
    attn_implementation: Optional[str] = None


@dataclass(frozen=True)
class SamplerConfig:
    grid: Tuple[DecodeConfig, ...]
    samples_per_decode: int


@dataclass(frozen=True)
class ReflectionConfig:
    decision_prompt_path: Path
    ops_prompt_path: Path
    batch_size: int
    max_operations: Optional[int] = None  # Per reflection cycle cap; None = unlimited
    allow_uncertain: bool = False
    eligibility_policy: str = "selected_mismatch_or_all_wrong"
    all_wrong_strategy: str = "learn"
    change_cap_per_epoch: Optional[int] = None
    hypothesis_min_support_cycles: int = 2
    hypothesis_min_unique_ticket_keys: int = 12
    temperature: float = 1.0
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    max_new_tokens: int = 1024
    max_reflection_length: int = 4096
    token_budget: int = 4096  # Token budget for reflection prompt packing
    retry_budget_per_group_per_epoch: int = 2
    max_calls_per_epoch: Optional[int] = None  # decision+ops calls per epoch; None = unlimited


@dataclass(frozen=True)
class RuleSearchBootstrapConfig:
    iterations: int = 200
    min_prob: float = 0.8
    seed: int = 0


@dataclass(frozen=True)
class RuleSearchGateConfig:
    """Rule admission gate for rule-search mode."""

    # Relative error reduction threshold, e.g. 0.1 = 10% relative error reduction.
    min_relative_error_reduction: float = 0.1
    # Maximum fraction of tickets whose majority prediction changes (churn cap).
    max_changed_fraction: float = 0.05
    # Allowable fp_rate increase for lifecycle operations.
    max_fp_rate_increase: float = 0.01
    bootstrap: RuleSearchBootstrapConfig = field(default_factory=RuleSearchBootstrapConfig)


@dataclass(frozen=True)
class RuleSearchEarlyStopConfig:
    patience: int = 3


@dataclass(frozen=True)
class RuleSearchConfig:
    """Rule-search (tree-growth) configuration."""

    proposer_prompt_path: Path
    proposer_temperature: float = 0.4
    proposer_top_p: float = 0.9
    proposer_repetition_penalty: float = 1.05
    proposer_max_new_tokens: int = 2048
    proposer_max_prompt_tokens: int = 4096
    reflect_size: int = 16
    num_candidate_rules: int = 3
    train_pool_size: int = 512
    train_pool_fraction: Optional[float] = None
    train_with_replacement: bool = False
    eval_pool_fraction: float = 0.2
    gate: RuleSearchGateConfig = field(default_factory=RuleSearchGateConfig)
    early_stop: RuleSearchEarlyStopConfig = field(default_factory=RuleSearchEarlyStopConfig)
    train_sampler: Optional[SamplerConfig] = None
    eval_sampler: Optional[SamplerConfig] = None
    mining_sampler: Optional[SamplerConfig] = None


@dataclass(frozen=True)
class RunnerConfig:
    epochs: int
    per_rank_rollout_batch_size: int
    logging_steps: int = 256


@dataclass(frozen=True)
class StageBDistillationConfig:
    """Optional distillation logging."""

    enabled: bool = True
    log_chatml_path: Optional[Path] = None
    distill_size: Optional[int] = None
    distill_seed: Optional[int] = None
    distill_temperature: Optional[float] = None


@dataclass(frozen=True)
class StageBConfig:
    seed: int
    stage_a_paths: Tuple[Path, ...]
    guidance: GuidanceConfig
    output: OutputConfig
    reflection: ReflectionConfig
    model: ModelConfig
    runner: RunnerConfig
    stage_b_distillation: Optional[StageBDistillationConfig] = None
    domain_map: Mapping[str, str] = field(default_factory=dict)
    default_domain: Optional[str] = None
    rule_search: Optional[RuleSearchConfig] = None


def _load_domain_map(raw: Any) -> Mapping[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("domain_map must be a mapping of mission -> domain")
    return {str(key): str(value) for key, value in raw.items()}


def _load_default_domain(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw).strip()
    return value if value else None


def resolve_domain_for_mission(config: StageBConfig, mission: str) -> str:
    domain_raw = None
    if config.domain_map:
        domain_raw = config.domain_map.get(mission)
    if domain_raw is None:
        domain_raw = config.default_domain
    if not domain_raw:
        raise ValueError(
            f"Stage-B domain is not configured for mission '{mission}'. "
            "Provide domain_map or default_domain in the Stage-B config."
        )
    pack = get_domain_pack(domain_raw)
    return pack.domain


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

    return SamplerConfig(
        grid=grid,
        samples_per_decode=samples_per_decode,
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
    reset_on_rerun = bool(section.get("reset_on_rerun", False))

    return GuidanceConfig(
        path=path,
        retention=retention,
        reset_on_rerun=reset_on_rerun,
    )


def _load_output(section: Mapping[str, Any]) -> OutputConfig:
    root = Path(_require(section, "root", "output section"))
    run_name = str(_require(section, "run_name", "output section"))
    return OutputConfig(
        root=root,
        run_name=run_name,
    )


def _load_model(section: Mapping[str, Any]) -> ModelConfig:
    model_name = str(_require(section, "model_name_or_path", "model section"))
    dtype_raw = _require(section, "torch_dtype", "model section")
    dtype_value = str(dtype_raw)
    device_map_raw = _require(section, "device_map", "model section")
    device_map_value = str(device_map_raw)
    attn_impl_raw = section.get("attn_implementation") or section.get("attn_impl")
    attn_impl_value = str(attn_impl_raw) if attn_impl_raw is not None else None
    return ModelConfig(
        model_name_or_path=model_name,
        torch_dtype=dtype_value,
        device_map=device_map_value,
        attn_implementation=attn_impl_value,
    )


def _load_reflection(section: Mapping[str, Any]) -> ReflectionConfig:
    decision_prompt_path = Path(
        _require(section, "decision_prompt_path", "reflection section")
    )
    ops_prompt_path = Path(_require(section, "ops_prompt_path", "reflection section"))
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
    allow_uncertain = bool(section.get("allow_uncertain", False))
    eligibility_policy = str(
        section.get("eligibility_policy", "selected_mismatch_or_all_wrong")
    )
    all_wrong_strategy = str(section.get("all_wrong_strategy", "learn"))
    change_cap_raw = section.get("change_cap_per_epoch")
    change_cap_per_epoch = int(change_cap_raw) if change_cap_raw is not None else None
    if change_cap_per_epoch is not None and change_cap_per_epoch <= 0:
        raise ValueError("reflection.change_cap_per_epoch must be > 0 if set")
    hypothesis_min_support_cycles = int(
        section.get("hypothesis_min_support_cycles", 2)
    )
    if hypothesis_min_support_cycles <= 0:
        raise ValueError("reflection.hypothesis_min_support_cycles must be > 0")
    hypothesis_min_unique_ticket_keys = int(
        section.get("hypothesis_min_unique_ticket_keys", 12)
    )
    if hypothesis_min_unique_ticket_keys <= 0:
        raise ValueError("reflection.hypothesis_min_unique_ticket_keys must be > 0")
    temperature = float(section.get("temperature", 1.0))
    top_p = float(section.get("top_p", 0.95))
    repetition_penalty = float(section.get("repetition_penalty", 1.0))
    max_new_tokens = int(section.get("max_new_tokens", 1024))
    max_reflection_length = int(section.get("max_reflection_length", 4096))
    if max_reflection_length <= 0:
        raise ValueError("reflection.max_reflection_length must be > 0")
    token_budget = int(section.get("token_budget", max_reflection_length))
    if token_budget <= 0:
        raise ValueError("reflection.token_budget must be > 0")

    retry_budget_raw = section.get("retry_budget_per_group_per_epoch", 2)
    retry_budget_per_group_per_epoch = int(retry_budget_raw)
    if retry_budget_per_group_per_epoch < 0:
        raise ValueError("reflection.retry_budget_per_group_per_epoch must be >= 0")

    max_calls_raw = section.get("max_calls_per_epoch")
    max_calls_per_epoch = int(max_calls_raw) if max_calls_raw is not None else None
    if max_calls_per_epoch is not None and max_calls_per_epoch <= 0:
        raise ValueError("reflection.max_calls_per_epoch must be > 0 if set")

    return ReflectionConfig(
        decision_prompt_path=decision_prompt_path,
        ops_prompt_path=ops_prompt_path,
        batch_size=batch_size,
        max_operations=max_operations_val,
        allow_uncertain=allow_uncertain,
        eligibility_policy=eligibility_policy,
        all_wrong_strategy=all_wrong_strategy,
        change_cap_per_epoch=change_cap_per_epoch,
        hypothesis_min_support_cycles=hypothesis_min_support_cycles,
        hypothesis_min_unique_ticket_keys=hypothesis_min_unique_ticket_keys,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        max_reflection_length=max_reflection_length,
        token_budget=token_budget,
        retry_budget_per_group_per_epoch=retry_budget_per_group_per_epoch,
        max_calls_per_epoch=max_calls_per_epoch,
    )


def _load_seed(raw_config: Mapping[str, Any]) -> int:
    seed_value = raw_config.get("seed")
    if seed_value is None:
        raise KeyError("Stage-B config must include top-level 'seed'")
    return int(seed_value)


def _load_runner(section: Mapping[str, Any]) -> RunnerConfig:
    epochs = int(_require(section, "epochs", "runner section"))
    if epochs <= 0:
        raise ValueError("runner.epochs must be > 0")
    per_rank_rollout_batch_size = int(section.get("per_rank_rollout_batch_size", 1))
    if per_rank_rollout_batch_size <= 0:
        raise ValueError("runner.per_rank_rollout_batch_size must be > 0")
    logging_steps = int(section.get("logging_steps", 256))
    if logging_steps <= 0:
        raise ValueError("runner.logging_steps must be > 0")
    return RunnerConfig(
        epochs=epochs,
        per_rank_rollout_batch_size=per_rank_rollout_batch_size,
        logging_steps=logging_steps,
    )


def _load_distillation(
    section: Optional[Mapping[str, Any]],
) -> Optional[StageBDistillationConfig]:
    if section is None:
        return StageBDistillationConfig()
    enabled = bool(section.get("enabled", True))
    raw_path = section.get("log_chatml_path")
    log_chatml_path = Path(raw_path) if raw_path else None
    distill_size = section.get("distill_size")
    if distill_size is not None:
        distill_size = int(distill_size)
        if distill_size <= 0:
            raise ValueError("stage_b_distillation.distill_size must be > 0")
    distill_seed = section.get("distill_seed")
    if distill_seed is not None:
        distill_seed = int(distill_seed)
    distill_temperature = section.get("distill_temperature")
    if distill_temperature is not None:
        distill_temperature = float(distill_temperature)
    return StageBDistillationConfig(
        enabled=enabled,
        log_chatml_path=log_chatml_path,
        distill_size=distill_size,
        distill_seed=distill_seed,
        distill_temperature=distill_temperature,
    )


def load_stage_b_config(path: str | Path) -> StageBConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw_config = yaml.safe_load(fh) or {}

    if not isinstance(raw_config, MutableMapping):
        raise TypeError("Stage-B config must be a mapping")

    seed_value = _load_seed(raw_config)

    stage_a_paths = _load_stage_a_paths(
        _require(raw_config, "stage_a_paths", "Stage-B config")
    )
    guidance = _load_guidance(_require(raw_config, "guidance", "Stage-B config"))
    output = _load_output(_require(raw_config, "output", "Stage-B config"))
    reflection = _load_reflection(_require(raw_config, "reflection", "Stage-B config"))
    model = _load_model(_require(raw_config, "model", "Stage-B config"))
    runner = _load_runner(_require(raw_config, "runner", "Stage-B config"))
    stage_b_distillation = _load_distillation(raw_config.get("stage_b_distillation"))
    domain_map = _load_domain_map(raw_config.get("domain_map"))
    default_domain = _load_default_domain(raw_config.get("default_domain"))

    rule_search_section = raw_config.get("rule_search")
    if rule_search_section is None:
        raise ValueError("Stage-B requires rule_search configuration")
    rule_search: Optional[RuleSearchConfig] = None
    if rule_search_section is not None:
        if not isinstance(rule_search_section, Mapping):
            raise TypeError("rule_search must be a mapping")

        proposer_prompt_path = Path(
            _require(rule_search_section, "proposer_prompt_path", "rule_search section")
        )
        proposer_temperature = float(rule_search_section.get("proposer_temperature", 0.4))
        proposer_top_p = float(rule_search_section.get("proposer_top_p", 0.9))
        proposer_repetition_penalty = float(
            rule_search_section.get("proposer_repetition_penalty", 1.05)
        )
        proposer_max_new_tokens = int(
            rule_search_section.get("proposer_max_new_tokens", 2048)
        )
        if proposer_max_new_tokens <= 0:
            raise ValueError("rule_search.proposer_max_new_tokens must be > 0")
        proposer_max_prompt_tokens = int(
            rule_search_section.get("proposer_max_prompt_tokens", 4096)
        )
        if proposer_max_prompt_tokens <= 0:
            raise ValueError("rule_search.proposer_max_prompt_tokens must be > 0")
        reflect_size = int(rule_search_section.get("reflect_size", 16))
        if reflect_size <= 0:
            raise ValueError("rule_search.reflect_size must be > 0")
        num_candidate_rules = int(rule_search_section.get("num_candidate_rules", 3))
        if num_candidate_rules <= 0:
            raise ValueError("rule_search.num_candidate_rules must be > 0")

        legacy_keys = {
            "validate_size",
            "validate_fraction",
            "validate_with_replacement",
            "holdout",
        }
        legacy_present = legacy_keys & set(rule_search_section.keys())
        if legacy_present:
            raise ValueError(
                "rule_search legacy keys are no longer supported: "
                f"{', '.join(sorted(legacy_present))}. "
                "Use train_pool_* and eval_pool_fraction instead."
            )

        train_pool_size = int(rule_search_section.get("train_pool_size", 512))
        if train_pool_size <= 0:
            raise ValueError("rule_search.train_pool_size must be > 0")
        train_pool_fraction_raw = rule_search_section.get("train_pool_fraction")
        train_pool_fraction = (
            float(train_pool_fraction_raw)
            if train_pool_fraction_raw is not None
            else None
        )
        if train_pool_fraction is not None and not (0.0 < train_pool_fraction <= 1.0):
            raise ValueError("rule_search.train_pool_fraction must be in (0, 1]")

        train_with_replacement = bool(
            rule_search_section.get("train_with_replacement", False)
        )

        eval_pool_fraction = float(rule_search_section.get("eval_pool_fraction", 0.2))
        if not (0.0 <= eval_pool_fraction < 1.0):
            raise ValueError("rule_search.eval_pool_fraction must be in [0, 1)")

        gate_section = rule_search_section.get("gate") or {}
        if not isinstance(gate_section, Mapping):
            raise TypeError("rule_search.gate must be a mapping when provided")
        min_rer = float(gate_section.get("min_relative_error_reduction", 0.1))
        if min_rer < 0.0:
            raise ValueError("rule_search.gate.min_relative_error_reduction must be >= 0")
        max_changed_fraction = gate_section.get("max_changed_fraction")
        if max_changed_fraction is None and "min_changed_fraction" in gate_section:
            max_changed_fraction = gate_section.get("min_changed_fraction")
        max_changed_fraction = float(max_changed_fraction or 0.05)
        if not (0.0 <= max_changed_fraction <= 1.0):
            raise ValueError("rule_search.gate.max_changed_fraction must be in [0, 1]")
        max_fp_rate_increase = float(gate_section.get("max_fp_rate_increase", 0.01))
        if max_fp_rate_increase < 0.0:
            raise ValueError("rule_search.gate.max_fp_rate_increase must be >= 0")

        bootstrap_section = gate_section.get("bootstrap") or {}
        if not isinstance(bootstrap_section, Mapping):
            raise TypeError("rule_search.gate.bootstrap must be a mapping when provided")
        bootstrap_iterations = int(bootstrap_section.get("iterations", 200))
        if bootstrap_iterations <= 0:
            raise ValueError("rule_search.gate.bootstrap.iterations must be > 0")
        bootstrap_min_prob = float(bootstrap_section.get("min_prob", 0.8))
        if not (0.0 <= bootstrap_min_prob <= 1.0):
            raise ValueError("rule_search.gate.bootstrap.min_prob must be in [0, 1]")
        bootstrap_seed = int(bootstrap_section.get("seed", seed_value))
        bootstrap = RuleSearchBootstrapConfig(
            iterations=bootstrap_iterations,
            min_prob=bootstrap_min_prob,
            seed=bootstrap_seed,
        )
        gate = RuleSearchGateConfig(
            min_relative_error_reduction=min_rer,
            max_changed_fraction=max_changed_fraction,
            max_fp_rate_increase=max_fp_rate_increase,
            bootstrap=bootstrap,
        )

        early_stop_section = rule_search_section.get("early_stop") or {}
        if not isinstance(early_stop_section, Mapping):
            raise TypeError("rule_search.early_stop must be a mapping when provided")
        patience = int(early_stop_section.get("patience", 3))
        if patience <= 0:
            raise ValueError("rule_search.early_stop.patience must be > 0")
        early_stop = RuleSearchEarlyStopConfig(patience=patience)

        train_sampler = _load_sampler(
            _require(rule_search_section, "train_sampler", "rule_search section")
        )
        eval_sampler = _load_sampler(
            _require(rule_search_section, "eval_sampler", "rule_search section")
        )
        mining_sampler = None
        if "mining_sampler" in rule_search_section and rule_search_section["mining_sampler"] is not None:
            mining_sampler = _load_sampler(
                _require(rule_search_section, "mining_sampler", "rule_search section")
            )

        rule_search = RuleSearchConfig(
            proposer_prompt_path=proposer_prompt_path,
            proposer_temperature=proposer_temperature,
            proposer_top_p=proposer_top_p,
            proposer_repetition_penalty=proposer_repetition_penalty,
            proposer_max_new_tokens=proposer_max_new_tokens,
            proposer_max_prompt_tokens=proposer_max_prompt_tokens,
            reflect_size=reflect_size,
            num_candidate_rules=num_candidate_rules,
            train_pool_size=train_pool_size,
            train_pool_fraction=train_pool_fraction,
            train_with_replacement=train_with_replacement,
            eval_pool_fraction=eval_pool_fraction,
            gate=gate,
            early_stop=early_stop,
            train_sampler=train_sampler,
            eval_sampler=eval_sampler,
            mining_sampler=mining_sampler,
        )

    return StageBConfig(
        seed=seed_value,
        stage_a_paths=stage_a_paths,
        guidance=guidance,
        output=output,
        reflection=reflection,
        model=model,
        runner=runner,
        stage_b_distillation=stage_b_distillation,
        domain_map=domain_map,
        default_domain=default_domain,
        rule_search=rule_search,
    )


__all__ = [
    "GuidanceConfig",
    "RunnerConfig",
    "StageBDistillationConfig",
    "ModelConfig",
    "OutputConfig",
    "ReflectionConfig",
    "SamplerConfig",
    "RuleSearchConfig",
    "RuleSearchGateConfig",
    "RuleSearchBootstrapConfig",
    "StageBConfig",
    "load_stage_b_config",
    "resolve_domain_for_mission",
]
