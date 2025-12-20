#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration schema for the Stage-B reflection pipeline."""

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
    trajectories_path: Path
    selections_jsonl: Path
    group_report: bool = True


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
class SignalsConfig:
    store_confidence: bool = False
    enable_consistency: bool = False


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
class RuleSearchHoldoutConfig:
    """Train/holdout split configuration for rule-search mode."""

    default_fraction: float = 0.2
    per_mission: Mapping[str, float] = field(default_factory=dict)
    seed: int = 0
    stratify_by_label: bool = True


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
    # Fraction of tickets whose majority prediction changes (sanity / coverage).
    min_changed_fraction: float = 0.01
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
    validate_size: Optional[int] = None
    validate_fraction: Optional[float] = None
    validate_with_replacement: bool = False
    holdout: RuleSearchHoldoutConfig = field(default_factory=RuleSearchHoldoutConfig)
    gate: RuleSearchGateConfig = field(default_factory=RuleSearchGateConfig)
    early_stop: RuleSearchEarlyStopConfig = field(default_factory=RuleSearchEarlyStopConfig)
    eval_sampler: Optional[SamplerConfig] = None
    mining_sampler: Optional[SamplerConfig] = None


@dataclass(frozen=True)
class GuidanceLifecycleConfig:
    """Guidance lifecycle management configuration.

    Controls automatic cleanup of low-confidence experiences.
    """

    confidence_drop_threshold: float = 0.35
    min_miss_before_drop: int = 3
    enable_auto_cleanup: bool = True  # Auto-cleanup at each epoch end


@dataclass(frozen=True)
class ManualReviewConfig:
    """Low-agreement threshold for flagging tickets."""

    # Minimum fraction of candidates sharing the majority verdict (0.0–1.0).
    min_verdict_agreement: float = 0.8


@dataclass(frozen=True)
class SelectionConfig:
    policy: str
    tie_break: str


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


@dataclass(frozen=True)
class StageBConfig:
    seed: int
    stage_a_paths: Tuple[Path, ...]
    guidance: GuidanceConfig
    output: OutputConfig
    sampler: Optional[SamplerConfig]  # Optional in rule_search mode
    reflection: ReflectionConfig
    selection: SelectionConfig
    model: ModelConfig
    runner: RunnerConfig
    manual_review: ManualReviewConfig
    guidance_lifecycle: Optional[GuidanceLifecycleConfig] = None
    stage_b_distillation: Optional[StageBDistillationConfig] = None
    domain_map: Mapping[str, str] = field(default_factory=dict)
    default_domain: Optional[str] = None
    mode: str = "legacy_reflection"
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
    group_report = bool(section.get("group_report", True))

    # Base directory for this run: {root}/{mission_name}/{run_name}
    # Mission-specific paths will be created dynamically in runner
    # These are placeholders - actual paths will be {root}/{mission_name}/{run_name}/*
    base_dir = root / "placeholder" / run_name  # Placeholder, actual path set in runner
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
    attn_impl_raw = section.get("attn_implementation") or section.get("attn_impl")
    attn_impl_value = str(attn_impl_raw) if attn_impl_raw is not None else None
    return ModelConfig(
        model_name_or_path=model_name,
        torch_dtype=dtype_value,
        device_map=device_map_value,
        attn_implementation=attn_impl_value,
    )


def _load_signals(section: Mapping[str, Any]) -> SignalsConfig:
    store_confidence = bool(section.get("store_confidence", False))
    enable_consistency = bool(
        _require(section, "enable_consistency", "signals section")
    )
    return SignalsConfig(
        store_confidence=store_confidence, enable_consistency=enable_consistency
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


def _load_manual_review(
    section: Optional[Mapping[str, Any]],
) -> ManualReviewConfig:
    """Load ManualReviewConfig controlling low-agreement gating."""

    if section is None:
        return ManualReviewConfig()

    min_verdict_agreement = float(section.get("min_verdict_agreement", 0.8))

    if not (0.0 <= min_verdict_agreement <= 1.0):
        raise ValueError("manual_review.min_verdict_agreement must be in [0.0, 1.0]")

    return ManualReviewConfig(min_verdict_agreement=min_verdict_agreement)


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


def _load_guidance_lifecycle(
    section: Optional[Mapping[str, Any]],
) -> Optional[GuidanceLifecycleConfig]:
    """Load optional guidance lifecycle configuration."""
    if section is None:
        return None
    confidence_drop = float(section.get("confidence_drop_threshold", 0.35))
    min_miss = int(section.get("min_miss_before_drop", 3))
    enable_cleanup = bool(section.get("enable_auto_cleanup", True))
    return GuidanceLifecycleConfig(
        confidence_drop_threshold=confidence_drop,
        min_miss_before_drop=min_miss,
        enable_auto_cleanup=enable_cleanup,
    )


def _load_distillation(
    section: Optional[Mapping[str, Any]],
) -> Optional[StageBDistillationConfig]:
    if section is None:
        return StageBDistillationConfig()
    enabled = bool(section.get("enabled", True))
    raw_path = section.get("log_chatml_path")
    log_chatml_path = Path(raw_path) if raw_path else None
    return StageBDistillationConfig(enabled=enabled, log_chatml_path=log_chatml_path)


def load_stage_b_config(path: str | Path) -> StageBConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw_config = yaml.safe_load(fh) or {}

    if not isinstance(raw_config, MutableMapping):
        raise TypeError("Stage-B config must be a mapping")

    seed_value = _load_seed(raw_config)

    mode_raw = str(raw_config.get("mode", "legacy_reflection")).strip().lower()
    if mode_raw not in {"legacy_reflection", "rule_search"}:
        raise ValueError(
            "Stage-B config 'mode' must be one of: legacy_reflection, rule_search"
        )

    stage_a_paths = _load_stage_a_paths(
        _require(raw_config, "stage_a_paths", "Stage-B config")
    )
    guidance = _load_guidance(_require(raw_config, "guidance", "Stage-B config"))
    output = _load_output(_require(raw_config, "output", "Stage-B config"))
    # sampler is required for legacy_reflection mode, optional for rule_search mode
    sampler = None
    if "sampler" in raw_config:
        sampler = _load_sampler(raw_config["sampler"])
    elif mode_raw == "legacy_reflection":
        raise ValueError("sampler is required for legacy_reflection mode")
    reflection = _load_reflection(_require(raw_config, "reflection", "Stage-B config"))
    selection = _load_selection(_require(raw_config, "selection", "Stage-B config"))
    manual_review = _load_manual_review(raw_config.get("manual_review"))
    model = _load_model(_require(raw_config, "model", "Stage-B config"))
    runner = _load_runner(_require(raw_config, "runner", "Stage-B config"))
    guidance_lifecycle = _load_guidance_lifecycle(raw_config.get("guidance_lifecycle"))
    stage_b_distillation = _load_distillation(raw_config.get("stage_b_distillation"))
    domain_map = _load_domain_map(raw_config.get("domain_map"))
    default_domain = _load_default_domain(raw_config.get("default_domain"))

    rule_search_section = raw_config.get("rule_search")
    rule_search: Optional[RuleSearchConfig] = None
    if rule_search_section is not None:
        if not isinstance(rule_search_section, Mapping):
            raise TypeError("rule_search must be a mapping when provided")

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

        validate_size_raw = rule_search_section.get("validate_size")
        validate_size = int(validate_size_raw) if validate_size_raw is not None else None
        if validate_size is not None and validate_size <= 0:
            raise ValueError("rule_search.validate_size must be > 0 if set")

        validate_fraction_raw = rule_search_section.get("validate_fraction")
        validate_fraction = (
            float(validate_fraction_raw) if validate_fraction_raw is not None else None
        )
        if validate_fraction is not None and not (0.0 < validate_fraction <= 1.0):
            raise ValueError("rule_search.validate_fraction must be in (0, 1]")

        validate_with_replacement = bool(
            rule_search_section.get("validate_with_replacement", False)
        )

        holdout_section = rule_search_section.get("holdout") or {}
        if not isinstance(holdout_section, Mapping):
            raise TypeError("rule_search.holdout must be a mapping when provided")
        default_fraction = float(holdout_section.get("default_fraction", 0.2))
        if not (0.0 <= default_fraction < 1.0):
            raise ValueError("rule_search.holdout.default_fraction must be in [0, 1)")
        per_mission_raw = holdout_section.get("per_mission") or {}
        if not isinstance(per_mission_raw, Mapping):
            raise TypeError("rule_search.holdout.per_mission must be a mapping")
        per_mission = {str(k): float(v) for k, v in per_mission_raw.items()}
        for mission, frac in per_mission.items():
            if not (0.0 <= frac < 1.0):
                raise ValueError(
                    f"rule_search.holdout.per_mission[{mission!r}] must be in [0, 1)"
                )
        holdout_seed = int(holdout_section.get("seed", seed_value))
        stratify_by_label = bool(holdout_section.get("stratify_by_label", True))
        holdout = RuleSearchHoldoutConfig(
            default_fraction=default_fraction,
            per_mission=per_mission,
            seed=holdout_seed,
            stratify_by_label=stratify_by_label,
        )

        gate_section = rule_search_section.get("gate") or {}
        if not isinstance(gate_section, Mapping):
            raise TypeError("rule_search.gate must be a mapping when provided")
        min_rer = float(gate_section.get("min_relative_error_reduction", 0.1))
        if min_rer < 0.0:
            raise ValueError("rule_search.gate.min_relative_error_reduction must be >= 0")
        min_changed_fraction = float(gate_section.get("min_changed_fraction", 0.01))
        if not (0.0 <= min_changed_fraction <= 1.0):
            raise ValueError("rule_search.gate.min_changed_fraction must be in [0, 1]")

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
            min_changed_fraction=min_changed_fraction,
            bootstrap=bootstrap,
        )

        early_stop_section = rule_search_section.get("early_stop") or {}
        if not isinstance(early_stop_section, Mapping):
            raise TypeError("rule_search.early_stop must be a mapping when provided")
        patience = int(early_stop_section.get("patience", 3))
        if patience <= 0:
            raise ValueError("rule_search.early_stop.patience must be > 0")
        early_stop = RuleSearchEarlyStopConfig(patience=patience)

        eval_sampler = None
        mining_sampler = None
        if "eval_sampler" in rule_search_section and rule_search_section["eval_sampler"] is not None:
            eval_sampler = _load_sampler(
                _require(rule_search_section, "eval_sampler", "rule_search section")
            )
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
            validate_size=validate_size,
            validate_fraction=validate_fraction,
            validate_with_replacement=validate_with_replacement,
            holdout=holdout,
            gate=gate,
            early_stop=early_stop,
            eval_sampler=eval_sampler,
            mining_sampler=mining_sampler,
        )

    if mode_raw == "rule_search" and rule_search is None:
        raise ValueError("mode=rule_search requires a rule_search section")

    return StageBConfig(
        mode=mode_raw,
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
        guidance_lifecycle=guidance_lifecycle,
        stage_b_distillation=stage_b_distillation,
        domain_map=domain_map,
        default_domain=default_domain,
        rule_search=rule_search,
    )


__all__ = [
    "GuidanceConfig",
    "GuidanceLifecycleConfig",
    "RunnerConfig",
    "StageBDistillationConfig",
    "ModelConfig",
    "OutputConfig",
    "ReflectionConfig",
    "ManualReviewConfig",
    "SamplerConfig",
    "SelectionConfig",
    "RuleSearchConfig",
    "RuleSearchHoldoutConfig",
    "RuleSearchGateConfig",
    "RuleSearchBootstrapConfig",
    "StageBConfig",
    "load_stage_b_config",
    "resolve_domain_for_mission",
]
