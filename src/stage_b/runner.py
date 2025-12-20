#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entrypoint for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)

from ..utils import configure_logging, get_logger
from .config import StageBConfig, load_stage_b_config, resolve_domain_for_mission
from .distributed import (
    barrier,
    broadcast_int,
    broadcast_object,
    gather_object,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
)
from .ingest import ingest_stage_a
from .io.export import serialize_selection, serialize_trajectory
from .io.group_report import build_group_report
from .io.guidance import GuidanceRepository
from .io.hypotheses import HypothesisPool
from .reflection import ReflectionEngine
from .rollout import RolloutSampler
from .rule_search import (
    EvalMetrics,
    build_gate_stats,
    build_ticket_stats,
    compute_metrics as compute_rule_search_metrics,
    normalize_rule_signature,
    pick_reflection_ticket_keys,
)
from .sampling.prompts import build_messages
from .types import (
    DeterministicSignals,
    ExperienceBundle,
    ExperienceOperation,
    ExperienceRecord,
    GroupTicket,
    MissionGuidance,
    ReflectionAction,
    ReflectionOutcome,
    ReflectionProposal,
    TrajectoryWithSignals,
)
from .utils.chinese import normalize_spaces, to_simplified
from .utils.perf import enable_tf32, maybe_empty_cache
from .utils.seed import seed_everything

logger = get_logger("stage_b.runner")


@dataclass(frozen=True)
class _PendingRuleFeedback:
    """Buffered hit/miss feedback awaiting stop-gradient classification."""

    experience_keys: Tuple[str, ...]
    label_match: bool


def _batch_size_for_retry(base_batch_size: int, *, attempt: int) -> int:
    """Deterministic retry batch shrink policy (attempt=0 is the initial batch)."""
    if attempt <= 0:
        return base_batch_size
    return max(1, base_batch_size // (2**attempt))


def _is_gradient_candidate(
    *,
    label_match: Optional[bool],
    low_agreement: bool,
    conflict_flag: bool,
    needs_manual_review: bool,
    candidate_verdicts: List[Optional[str]],
) -> bool:
    """Return True if the ticket is eligible for reflection (gradient candidate)."""
    verdicts = {v for v in candidate_verdicts if v is not None}
    rollout_contradiction = "pass" in verdicts and "fail" in verdicts
    return bool(
        label_match is False
        or rollout_contradiction
        or low_agreement
        or conflict_flag
        or needs_manual_review
    )


def _compute_learnability_coverage(
    learnable_ids: Iterable[str],
    evidence_ops: Iterable[str],
    evidence_hypotheses: Iterable[str],
) -> Tuple[set[str], set[str]]:
    contributors = set(evidence_ops) | set(evidence_hypotheses)
    uncovered = set(learnable_ids) - contributors
    return contributors, uncovered


def _drain_buffered_feedback(
    pending_feedback: Dict[str, _PendingRuleFeedback],
    *,
    stop_gradient_ticket_keys: set[str],
    contributor_ticket_keys: set[str],
) -> Tuple[List[_PendingRuleFeedback], List[_PendingRuleFeedback]]:
    """Pop buffered feedback entries into (to_commit, to_drop) buckets."""

    committed: List[_PendingRuleFeedback] = []
    dropped: List[_PendingRuleFeedback] = []

    for ticket_key in sorted(stop_gradient_ticket_keys):
        feedback = pending_feedback.pop(ticket_key, None)
        if feedback is not None:
            dropped.append(feedback)

    for ticket_key in sorted(contributor_ticket_keys):
        feedback = pending_feedback.pop(ticket_key, None)
        if feedback is not None:
            committed.append(feedback)

    return committed, dropped


def _dtype_from_str(name: str):
    lowered = name.lower()
    if not hasattr(torch, lowered):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return getattr(torch, lowered)


def _detect_model_variant(model_path: str) -> str:
    """Return 'vl' for multi-modal Qwen3-VL checkpoints, else 'lm'."""

    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to read model config at {model_path}: {exc}"
        ) from exc

    model_type = getattr(cfg, "model_type", "") or ""
    has_vision = getattr(cfg, "vision_config", None) is not None

    if "vl" in str(model_type).lower() or has_vision:
        return "vl"
    return "lm"


def _safe_model_max_length(tokenizer) -> int | None:
    value = getattr(tokenizer, "model_max_length", None)
    if value is None:
        return None
    try:
        length = int(value)
    except Exception:  # noqa: BLE001
        return None
    if length <= 0 or length > 1_000_000:
        return None
    return length


def _load_model(config: StageBConfig):
    variant = _detect_model_variant(config.model.model_name_or_path)
    logger.info(
        "Detected model variant '%s' for %s",
        variant,
        config.model.model_name_or_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    dtype = _dtype_from_str(config.model.torch_dtype)
    device_map: object = config.model.device_map
    if get_world_size() > 1 and torch.cuda.is_available():
        local_rank = get_local_rank()
        device_map = {"": local_rank}
        if is_main_process():
            logger.info(
                "Distributed mode: forcing per-rank single-GPU model placement (LOCAL_RANK=%d, device_map=%r -> %r)",
                local_rank,
                config.model.device_map,
                device_map,
            )
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
    }
    # Enable Flash Attention 2 if specified in config (similar to training pipeline)
    if config.model.attn_implementation is not None:
        # Normalize 'flash_attn' to 'flash_attention_2' for compatibility
        attn_impl = config.model.attn_implementation
        if attn_impl.lower() in ("flash_attn", "flash_attention_2"):
            attn_impl = "flash_attention_2"
        model_kwargs["attn_implementation"] = attn_impl
        if is_main_process():
            logger.info("Using attention implementation: %s", attn_impl)
    if variant == "vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model.model_name_or_path,
            **model_kwargs,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            config.model.model_name_or_path, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name_or_path,
            **model_kwargs,
            trust_remote_code=True,
        )
        processor = None

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, processor


def _prepare_mission_output_paths(mission_dir: Path) -> Tuple[Path, Path]:
    """Prepare output paths for a specific mission.

    Args:
        mission_dir: Directory for this mission ({root}/{mission_name}/{run_name})

    Returns:
        Tuple of (trajectories_path, selections_path)
    """
    mission_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = mission_dir / "trajectories.jsonl"
    selections_path = mission_dir / "selections.jsonl"

    trajectories_path.write_text("", encoding="utf-8")
    selections_path.write_text("", encoding="utf-8")

    return trajectories_path, selections_path


def _reset_mission_artifacts(mission_dir: Path) -> None:
    """Clear per-run artifacts to avoid cross-run contamination."""

    mission_dir.mkdir(parents=True, exist_ok=True)
    # Fresh-run artifacts.
    for filename in (
        "trajectories.jsonl",
        "selections.jsonl",
        "need_review_queue.jsonl",
        "failure_malformed.jsonl",
        "reflection.jsonl",
        "hypothesis_events.jsonl",
        "rule_candidates.jsonl",
        "benchmarks.jsonl",
        "metrics.jsonl",
        "group_report_delta.jsonl",
    ):
        path = mission_dir / filename
        if path.exists():
            path.unlink()
        path.write_text("", encoding="utf-8")

    hypotheses_path = mission_dir / "hypotheses.json"
    if hypotheses_path.exists():
        hypotheses_path.unlink()
    hypotheses_path.write_text("{}", encoding="utf-8")

    need_review_summary_path = mission_dir / "need_review.json"
    if need_review_summary_path.exists():
        need_review_summary_path.unlink()

    cache_dir = mission_dir / "reflection_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def _shuffle_indices(count: int, *, epoch: int, base_seed: int) -> List[int]:
    indices = list(range(count))
    seed_value = base_seed + epoch
    random.Random(seed_value).shuffle(indices)
    return indices


def _holdout_fraction_for_mission(config: StageBConfig, mission: str) -> float:
    if config.rule_search is None:
        return 0.0
    holdout_cfg = config.rule_search.holdout
    if mission in holdout_cfg.per_mission:
        return float(holdout_cfg.per_mission[mission])
    return float(holdout_cfg.default_fraction)


def _split_train_holdout(
    tickets: Sequence[GroupTicket],
    *,
    fraction: float,
    seed: int,
    stratify_by_label: bool,
) -> Tuple[List[GroupTicket], List[GroupTicket]]:
    if not tickets:
        return [], []
    if fraction <= 0.0:
        return list(tickets), []
    if fraction >= 1.0:
        return [], list(tickets)

    rng = random.Random(int(seed))

    if stratify_by_label:
        by_label: Dict[str, List[GroupTicket]] = defaultdict(list)
        for ticket in tickets:
            by_label[str(ticket.label)].append(ticket)

        train: List[GroupTicket] = []
        holdout: List[GroupTicket] = []
        for label in sorted(by_label.keys()):
            bucket = sorted(by_label[label], key=lambda t: t.key)
            indices = list(range(len(bucket)))
            rng.shuffle(indices)
            cutoff = int(round(len(bucket) * float(fraction)))
            holdout.extend([bucket[i] for i in indices[:cutoff]])
            train.extend([bucket[i] for i in indices[cutoff:]])
        return train, holdout

    ordered = sorted(tickets, key=lambda t: t.key)
    indices = list(range(len(ordered)))
    rng.shuffle(indices)
    cutoff = int(round(len(ordered) * float(fraction)))
    holdout = [ordered[i] for i in indices[:cutoff]]
    train = [ordered[i] for i in indices[cutoff:]]
    return train, holdout


def _sample_validation_tickets(
    tickets: Sequence[GroupTicket],
    *,
    validate_size: Optional[int],
    validate_fraction: Optional[float],
    with_replacement: bool,
    seed: int,
) -> List[GroupTicket]:
    if not tickets:
        return []

    total = len(tickets)
    rng = random.Random(int(seed))
    if validate_fraction is not None:
        target = int(round(total * float(validate_fraction)))
    elif validate_size is not None:
        target = int(validate_size)
    else:
        target = total

    if with_replacement:
        return [rng.choice(tickets) for _ in range(max(1, target))]

    target = min(max(1, target), total)
    indices = list(range(total))
    rng.shuffle(indices)
    return [tickets[i] for i in indices[:target]]


def _extract_verdict_samples(
    trajectories: Sequence,
) -> List[Optional[str]]:
    verdicts: List[Optional[str]] = []
    for cand in trajectories:
        verdict_val = getattr(cand, "verdict", None)
        format_ok = bool(getattr(cand, "format_ok", False))
        if not format_ok:
            verdicts.append(None)
        else:
            verdicts.append(str(verdict_val) if verdict_val is not None else None)
    return verdicts


def _extract_reason_samples(
    trajectories: Sequence,
) -> List[Optional[str]]:
    reasons: List[Optional[str]] = []
    for cand in trajectories:
        format_ok = bool(getattr(cand, "format_ok", False))
        reason = getattr(cand, "reason", None)
        if format_ok and reason:
            reasons.append(str(reason))
        else:
            reasons.append(None)
    return reasons


def _distributed_rollout_payloads(
    *,
    tickets: Sequence[GroupTicket],
    sampler: RolloutSampler,
    guidance: MissionGuidance,
    mission: str,
    domain: str,
    per_rank_batch_size: int,
) -> Dict[str, List[object]]:
    """Run distributed rollout for a single mission and return per-ticket trajectories.

    Returns a mapping only on rank 0; other ranks return an empty dict.
    """

    world_size = get_world_size()
    rank = get_rank()
    global_batch_size = max(1, int(per_rank_batch_size)) * max(1, int(world_size))

    guidance_map = {mission: guidance}
    domain_map = {mission: domain}

    merged: Dict[str, List[object]] = {}
    for batch in _chunked(list(tickets), global_batch_size):
        start, end = _shard_bounds(len(batch), world_size=world_size, rank=rank)
        shard = batch[start:end]
        local = sampler.generate_for_batch(shard, guidance_map, domain_map)
        gathered = gather_object(local, dst=0)
        if not is_main_process():
            continue
        assert gathered is not None
        for part in gathered:
            for ticket_key, trajectories in part.items():
                merged.setdefault(ticket_key, []).extend(trajectories)

    return merged if is_main_process() else {}


def _distributed_rollout_verdicts(
    *,
    tickets: Sequence[GroupTicket],
    sampler: RolloutSampler,
    guidance: MissionGuidance,
    mission: str,
    domain: str,
    per_rank_batch_size: int,
) -> Dict[str, List[Optional[str]]]:
    """Run distributed rollout for a single mission and return per-ticket verdict samples.

    Returns a mapping only on rank 0; other ranks return an empty dict.
    """

    world_size = get_world_size()
    rank = get_rank()
    global_batch_size = max(1, int(per_rank_batch_size)) * max(1, int(world_size))

    guidance_map = {mission: guidance}
    domain_map = {mission: domain}

    merged: Dict[str, List[Optional[str]]] = {}
    for batch in _chunked(list(tickets), global_batch_size):
        start, end = _shard_bounds(len(batch), world_size=world_size, rank=rank)
        shard = batch[start:end]
        local = sampler.generate_for_batch(shard, guidance_map, domain_map)
        gathered = gather_object(local, dst=0)
        if not is_main_process():
            continue
        assert gathered is not None
        for part in gathered:
            for ticket_key, trajectories in part.items():
                merged.setdefault(ticket_key, []).extend(
                    _extract_verdict_samples(trajectories)
                )

    return merged if is_main_process() else {}


def _build_rule_proposer_user_prompt(
    *,
    mission: str,
    guidance: MissionGuidance,
    examples: Sequence[GroupTicket],
    stats_by_ticket: Mapping[str, object],
) -> str:
    g0 = guidance.experiences.get("G0") or ""
    scaffold_lines = [
        f"[{key}]. {value}"
        for key, value in sorted(guidance.experiences.items())
        if key.startswith("S")
    ]

    lines: List[str] = [
        f"任务: {mission}",
        f"G0: {g0}",
    ]
    if scaffold_lines:
        lines.extend(["", "SCAFFOLD (S*):", *scaffold_lines])
    lines.append("")
    lines.append("以下为用于提出候选规则的错例样本（仅供提案，不用于最终评估）：")
    lines.append("")

    for idx, ticket in enumerate(examples, start=1):
        stats = stats_by_ticket.get(ticket.key)
        majority_pred = getattr(stats, "majority_pred", None)
        agreement = getattr(stats, "agreement", 0.0)
        difficulty = getattr(stats, "difficulty", 0.0)
        lines.append(
            f"样本{idx}: ticket_key={ticket.key}; gt_label={ticket.label}; "
            f"baseline_majority={majority_pred}; agreement={agreement:.3f}; difficulty={difficulty:.3f}"
        )
        summaries = ticket.summaries.as_dict()
        lines.append("stage_a_summaries:")
        for key in sorted(summaries.keys()):
            lines.append(f"  - {key}: {summaries[key]}")
        lines.append("")

    lines.append("请基于上述错例样本提出候选规则。只输出 JSON。")
    return "\n".join(lines).strip()


def _propose_rules(
    *,
    model: torch.nn.Module,
    tokenizer,
    config: StageBConfig,
    mission: str,
    guidance: MissionGuidance,
    examples: Sequence[GroupTicket],
    stats_by_ticket: Mapping[str, object],
    reflection_engine: ReflectionEngine,
    iteration: int,
    log_dir: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Run proposer LLM once and return candidate rules (validated + de-duplicated)."""

    assert config.rule_search is not None
    system_prompt = Path(config.rule_search.proposer_prompt_path).read_text(
        encoding="utf-8"
    )
    max_prompt_tokens = int(config.rule_search.proposer_max_prompt_tokens)
    example_list = list(examples)
    prompt_tokens: Optional[int] = None

    while True:
        user_prompt = _build_rule_proposer_user_prompt(
            mission=mission,
            guidance=guidance,
            examples=example_list,
            stats_by_ticket=stats_by_ticket,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            chat_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            chat_prompt = tokenizer.apply_chat_template(  # type: ignore[call-arg]
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        assert isinstance(chat_prompt, str)

        try:
            encoded_probe = tokenizer(
                chat_prompt,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            prompt_tokens = int(encoded_probe["input_ids"].size(1))
        except Exception:  # noqa: BLE001
            prompt_tokens = None

        if prompt_tokens is None or prompt_tokens <= max_prompt_tokens:
            break
        if len(example_list) <= 1:
            break
        example_list.pop()

    if prompt_tokens is not None and prompt_tokens > max_prompt_tokens:
        logger.warning(
            "rule_search proposer prompt exceeds token budget after trimming: tokens=%d budget=%d examples=%d",
            prompt_tokens,
            max_prompt_tokens,
            len(example_list),
        )

    model_max_length = _safe_model_max_length(tokenizer)
    if prompt_tokens is not None and model_max_length is not None:
        total_requested = prompt_tokens + int(config.rule_search.proposer_max_new_tokens)
        if prompt_tokens > model_max_length:
            logger.error(
                "rule_search proposer prompt exceeds model_max_length: prompt=%d model_max_length=%d",
                prompt_tokens,
                model_max_length,
            )
        elif total_requested > model_max_length:
            logger.warning(
                "rule_search proposer prompt+generation exceeds model_max_length: prompt=%d new=%d total=%d model_max_length=%d",
                prompt_tokens,
                int(config.rule_search.proposer_max_new_tokens),
                total_requested,
                model_max_length,
            )

    encoded = tokenizer(
        chat_prompt,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    inputs = {k: v.to(model.device) for k, v in encoded.items()}

    torch.manual_seed(int(config.seed))
    with torch.inference_mode():
        output = model.generate(  # type: ignore[call-overload]
            **inputs,
            max_new_tokens=config.rule_search.proposer_max_new_tokens,
            temperature=config.rule_search.proposer_temperature,
            top_p=config.rule_search.proposer_top_p,
            repetition_penalty=config.rule_search.proposer_repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    prompt_length = inputs["input_ids"].size(1)
    generated_tokens = output[0, prompt_length:]
    raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    response = normalize_spaces(to_simplified(raw_response))

    try:
        payload = ReflectionEngine._loads_first_json(response)
    except Exception as exc:  # noqa: BLE001
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            _append_jsonl(
                log_dir / "rule_search_proposer_failures.jsonl",
                {
                    "timestamp": time.time(),
                    "iteration": iteration,
                    "mission": mission,
                    "examples": len(example_list),
                    "prompt_tokens": prompt_tokens,
                    "error": str(exc),
                    "raw_response_prefix": raw_response[:400],
                    "raw_response_suffix": raw_response[-400:],
                },
            )
        raise ValueError(
            f"No valid JSON found in rule_search proposer response (examples={len(example_list)}, prompt_tokens={prompt_tokens})"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Rule proposer must return a JSON object")

    rules_raw = payload.get("rules")
    if not isinstance(rules_raw, Sequence) or isinstance(rules_raw, (str, bytes)):
        raise ValueError("rules must be a list")

    scaffold_texts = [
        text for key, text in guidance.experiences.items() if key.startswith("S")
    ]
    existing_norms = {
        normalize_rule_signature(v) for v in guidance.experiences.values()
    }

    candidates: List[Dict[str, str]] = []
    seen_signatures: set[str] = set()
    for entry in rules_raw:
        if not isinstance(entry, Mapping):
            continue
        text_raw = entry.get("text")
        text = str(text_raw).strip() if text_raw is not None else ""
        if not text:
            continue
        if reflection_engine._reject_experience_text(text):
            continue
        if reflection_engine._contains_forbidden_phrase(text):
            continue
        if reflection_engine._contains_ambiguous_negation(text):
            continue
        if not reflection_engine._is_binary_hypothesis(text):
            continue
        if reflection_engine._hypothesis_conflicts_scaffold(text, scaffold_texts):
            continue

        dim_raw = entry.get("dimension")
        dim = str(dim_raw).strip() if dim_raw is not None else ""
        if dim and reflection_engine._is_brand_dimension(dim):
            continue

        signature = normalize_rule_signature(text)
        if not signature:
            continue
        if signature in existing_norms:
            continue
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        rationale_raw = entry.get("rationale")
        rationale = str(rationale_raw).strip() if rationale_raw is not None else ""
        candidates.append({"text": text, "rationale": rationale})
        if len(candidates) >= int(config.rule_search.num_candidate_rules):
            break

    return candidates


def _load_rejected_rule_signatures(rule_candidates_path: Path) -> set[str]:
    rejected: set[str] = set()
    for row in _load_jsonl(rule_candidates_path):
        decision = str(row.get("decision") or "").strip().lower()
        signature = str(row.get("signature") or "").strip()
        if signature and decision == "rejected":
            rejected.add(signature)
    return rejected


def _run_rule_search_mission(
    *,
    config: StageBConfig,
    model,
    tokenizer,
    mission: str,
    mission_tickets: Sequence[GroupTicket],
    mission_dir: Path,
    domain: str,
) -> None:
    assert config.rule_search is not None

    rule_candidates_path = mission_dir / "rule_candidates.jsonl"
    benchmarks_path = mission_dir / "benchmarks.jsonl"
    hard_cases_path = mission_dir / "rule_search_hard_cases.jsonl"
    regressions_path = mission_dir / "rule_search_candidate_regressions.jsonl"

    mission_guidance_repo: Optional[GuidanceRepository] = None
    reflection_engine: Optional[ReflectionEngine] = None
    if is_main_process():
        mission_guidance_repo = _setup_mission_guidance(
            startup_path=config.guidance.path,
            mission_dir=mission_dir,
            mission=mission,
            retention=config.guidance.retention,
            reset=config.guidance.reset_on_rerun,
        )
        reflection_engine = ReflectionEngine(
            model=model,
            tokenizer=tokenizer,
            config=config.reflection,
            guidance_repo=mission_guidance_repo,
            reflection_log=None,
        )

    rejected_signatures = (
        _load_rejected_rule_signatures(rule_candidates_path)
        if is_main_process()
        else set()
    )
    rejected_signatures = broadcast_object(
        rejected_signatures if is_main_process() else None, src=0
    )

    # Deterministic mission ticket lookup.
    ticket_by_key = {ticket.key: ticket for ticket in mission_tickets}

    holdout_fraction = _holdout_fraction_for_mission(config, mission)
    train_tickets, holdout_tickets = _split_train_holdout(
        list(mission_tickets),
        fraction=holdout_fraction,
        seed=config.rule_search.holdout.seed,
        stratify_by_label=config.rule_search.holdout.stratify_by_label,
    )
    logger.info(
        "rule_search split: mission=%s train=%d holdout=%d holdout_fraction=%.3f",
        mission,
        len(train_tickets),
        len(holdout_tickets),
        holdout_fraction,
    )

    if config.rule_search.eval_sampler is None:
        raise ValueError("rule_search.eval_sampler is required in rule_search mode")
    eval_sampler_cfg = config.rule_search.eval_sampler
    # mining_sampler is optional; if not provided, use eval_sampler
    mining_sampler_cfg = config.rule_search.mining_sampler or config.rule_search.eval_sampler
    eval_sampler = RolloutSampler(
        model=model, tokenizer=tokenizer, config=eval_sampler_cfg
    )
    mining_sampler = RolloutSampler(
        model=model, tokenizer=tokenizer, config=mining_sampler_cfg
    )

    patience = int(config.rule_search.early_stop.patience)
    no_gain_rounds = 0
    hard_case_limit = max(100, int(config.rule_search.reflect_size) * 4)

    # Use runner.epochs as the iteration budget for rule-search.
    for iteration in range(1, config.runner.epochs + 1):
        current_guidance = None
        if is_main_process():
            assert mission_guidance_repo is not None
            current_guidance = mission_guidance_repo.get(mission)
        current_guidance = broadcast_object(current_guidance, src=0)

        # Sample validation tickets from train set.
        validate_seed = int(config.seed) + iteration
        validate_tickets = _sample_validation_tickets(
            train_tickets,
            validate_size=config.rule_search.validate_size,
            validate_fraction=config.rule_search.validate_fraction,
            with_replacement=config.rule_search.validate_with_replacement,
            seed=validate_seed,
        )

        # Baseline rollout for gate evaluation (paired with candidate eval sampler).
        base_payloads_eval = _distributed_rollout_payloads(
            tickets=validate_tickets,
            sampler=eval_sampler,
            guidance=current_guidance,
            mission=mission,
            domain=domain,
            per_rank_batch_size=config.runner.per_rank_rollout_batch_size,
        )
        base_samples_eval: Dict[str, List[Optional[str]]] = {}
        base_reasons_eval: Dict[str, List[Optional[str]]] = {}
        if is_main_process():
            for ticket_key, trajectories in base_payloads_eval.items():
                base_samples_eval[ticket_key] = _extract_verdict_samples(trajectories)
                base_reasons_eval[ticket_key] = _extract_reason_samples(trajectories)
        base_stats_by_ticket: Dict[str, object] = {}
        stats_for_proposer: Dict[str, object] = {}
        base_metrics = None
        reflect_keys: List[str] = []
        candidates: List[Dict[str, str]] = []

        if is_main_process():
            for ticket in validate_tickets:
                verdicts = base_samples_eval.get(ticket.key, [])
                base_stats_by_ticket[ticket.key] = build_ticket_stats(
                    ticket_key=ticket.key,
                    gt_label=ticket.label,
                    verdicts=verdicts,
                )
            base_metrics = compute_rule_search_metrics(
                base_stats_by_ticket.values()  # type: ignore[arg-type]
            )
            stats_for_proposer = dict(base_stats_by_ticket)
            for row in _hard_case_rows(
                base_stats_by_ticket,
                mission=mission,
                iteration=iteration,
                sampler="eval",
                limit=hard_case_limit,
                reason_samples_by_ticket=base_reasons_eval,
            ):
                _append_jsonl(hard_cases_path, row)

            # Optional mining sampler for selecting harder mismatches (does not affect gate metrics).
            if config.rule_search.mining_sampler is not None:
                base_samples_mining = _distributed_rollout_verdicts(
                    tickets=validate_tickets,
                    sampler=mining_sampler,
                    guidance=current_guidance,
                    mission=mission,
                    domain=domain,
                    per_rank_batch_size=config.runner.per_rank_rollout_batch_size,
                )
                mining_stats: Dict[str, object] = {}
                for ticket in validate_tickets:
                    verdicts = base_samples_mining.get(ticket.key, [])
                    mining_stats[ticket.key] = build_ticket_stats(
                        ticket_key=ticket.key,
                        gt_label=ticket.label,
                        verdicts=verdicts,
                    )
                stats_for_proposer = mining_stats
                for row in _hard_case_rows(
                    mining_stats,
                    mission=mission,
                    iteration=iteration,
                    sampler="mining",
                    limit=hard_case_limit,
                ):
                    _append_jsonl(hard_cases_path, row)

            reflect_keys = pick_reflection_ticket_keys(
                stats_for_proposer,  # type: ignore[arg-type]
                reflect_size=config.rule_search.reflect_size,
            )

            examples = [
                ticket_by_key[key] for key in reflect_keys if key in ticket_by_key
            ]

            if examples:
                assert reflection_engine is not None
                try:
                        candidates = _propose_rules(
                            model=model,
                            tokenizer=tokenizer,
                            config=config,
                            mission=mission,
                            guidance=current_guidance,
                            examples=examples,
                            stats_by_ticket=stats_for_proposer,
                            reflection_engine=reflection_engine,
                            iteration=iteration,
                            log_dir=mission_dir / "reflection_cache",
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("rule_search proposer failed: %s", exc)
                    candidates = []

            # Filter already-rejected signatures.
            filtered: List[Dict[str, str]] = []
            for cand in candidates:
                sig = normalize_rule_signature(cand.get("text", ""))
                if not sig:
                    continue
                if sig in rejected_signatures:
                    continue
                filtered.append(cand)
            candidates = filtered

        # Broadcast proposer results.
        candidates = broadcast_object(candidates if is_main_process() else None, src=0)
        reflect_keys = broadcast_object(
            reflect_keys if is_main_process() else None, src=0
        )
        base_metrics = broadcast_object(
            base_metrics if is_main_process() else None, src=0
        )

        if not candidates:
            if is_main_process():
                logger.info(
                    "rule_search iteration %d: no candidates proposed (reflect=%d, validate=%d); treating as no-gain",
                    iteration,
                    len(reflect_keys),
                    len(validate_tickets),
                )
            no_gain_rounds = broadcast_int(
                no_gain_rounds + 1 if is_main_process() else 0, src=0
            )
            should_stop = no_gain_rounds >= patience
            should_stop = bool(
                broadcast_int(1 if should_stop and is_main_process() else 0, src=0)
            )
            if should_stop:
                break
            continue

        # Candidate evaluation (A/B on the same validation set, paired seeds by sampler config).
        best_candidate: Optional[Dict[str, object]] = None
        for cand_idx, cand in enumerate(candidates, start=1):
            cand_text = str(cand.get("text") or "").strip()
            cand_sig = normalize_rule_signature(cand_text)

            candidate_guidance = None
            candidate_operation = None
            if is_main_process():
                assert mission_guidance_repo is not None
                reflection_id = f"rule_search_preview_{iteration}_{cand_idx}"
                candidate_operation = ExperienceOperation(
                    op="upsert",
                    key=None,
                    text=cand_text,
                    rationale="rule_search_candidate",
                    evidence=tuple(reflect_keys),
                    merged_from=None,
                )
                proposal = ReflectionProposal(
                    action=ReflectionAction("refine"),
                    summary=None,
                    critique=None,
                    operations=(candidate_operation,),
                    hypotheses=tuple(),
                    evidence_group_ids=tuple(reflect_keys),
                    uncertainty_note=None,
                    no_evidence_group_ids=tuple(),
                    text=None,
                )
                candidate_guidance = mission_guidance_repo.preview_reflection(
                    mission,
                    proposal=proposal,
                    reflection_id=reflection_id,
                    source_group_ids=list(reflect_keys),
                    operations=(candidate_operation,),
                )

            candidate_guidance = broadcast_object(
                candidate_guidance if is_main_process() else None, src=0
            )

            cand_samples = _distributed_rollout_verdicts(
                tickets=validate_tickets,
                sampler=eval_sampler,
                guidance=candidate_guidance,
                mission=mission,
                domain=domain,
                per_rank_batch_size=config.runner.per_rank_rollout_batch_size,
            )

            if not is_main_process():
                continue

            # Build candidate stats and gate decision (rank 0 only).
            cand_stats_by_ticket: Dict[str, object] = {}
            for ticket in validate_tickets:
                verdicts = cand_samples.get(ticket.key, [])
                cand_stats_by_ticket[ticket.key] = build_ticket_stats(
                    ticket_key=ticket.key,
                    gt_label=ticket.label,
                    verdicts=verdicts,
                )

            gate_stats, passed = build_gate_stats(
                base_stats=base_stats_by_ticket,  # type: ignore[arg-type]
                new_stats=cand_stats_by_ticket,  # type: ignore[arg-type]
                rer_threshold=config.rule_search.gate.min_relative_error_reduction,
                bootstrap_iterations=config.rule_search.gate.bootstrap.iterations,
                bootstrap_min_prob=config.rule_search.gate.bootstrap.min_prob,
                bootstrap_seed=config.rule_search.gate.bootstrap.seed + iteration,
                min_changed_fraction=config.rule_search.gate.min_changed_fraction,
            )
            cand_metrics = compute_rule_search_metrics(
                cand_stats_by_ticket.values()  # type: ignore[arg-type]
            )

            decision = "accepted" if passed else "rejected"
            regressions = _candidate_regressions(
                base_stats_by_ticket,  # type: ignore[arg-type]
                cand_stats_by_ticket,  # type: ignore[arg-type]
                limit=50,
            )
            if regressions:
                _append_jsonl(
                    regressions_path,
                    {
                        "timestamp": time.time(),
                        "iteration": iteration,
                        "candidate_index": cand_idx,
                        "mission": mission,
                        "signature": cand_sig,
                        "decision": decision,
                        "regression_count": len(regressions),
                        "regressions": regressions,
                    },
                )
            _append_jsonl(
                rule_candidates_path,
                {
                    "timestamp": time.time(),
                    "iteration": iteration,
                    "candidate_index": cand_idx,
                    "mission": mission,
                    "signature": cand_sig,
                    "text": cand_text,
                    "rationale": str(cand.get("rationale") or "").strip() or None,
                    "validate_n": len(validate_tickets),
                    "base_acc": base_metrics.acc if base_metrics else None,
                    "base_fn_rate": base_metrics.fn_rate if base_metrics else None,
                    "base_fp_rate": base_metrics.fp_rate if base_metrics else None,
                    "base_fn_over_tp": base_metrics.fn_over_tp if base_metrics else None,
                    "base_fp_over_tp": base_metrics.fp_over_tp if base_metrics else None,
                    "cand_acc": cand_metrics.acc,
                    "cand_fn_rate": cand_metrics.fn_rate,
                    "cand_fp_rate": cand_metrics.fp_rate,
                    "cand_fn_over_tp": cand_metrics.fn_over_tp,
                    "cand_fp_over_tp": cand_metrics.fp_over_tp,
                    "relative_error_reduction": gate_stats.relative_error_reduction,
                    "changed_fraction": gate_stats.changed_fraction,
                    "bootstrap_prob": gate_stats.bootstrap_prob,
                    "decision": decision,
                },
            )

            if passed:
                if (
                    best_candidate is None
                    or gate_stats.relative_error_reduction
                    > float(
                        best_candidate.get("relative_error_reduction", -1.0)  # type: ignore[union-attr]
                    )
                ):
                    best_candidate = {
                        "candidate_index": cand_idx,
                        "signature": cand_sig,
                        "text": cand_text,
                        "rationale": str(cand.get("rationale") or "").strip() or None,
                        "gate": gate_stats,
                        "cand_metrics": cand_metrics,
                        "operation": candidate_operation,
                    }

        # Apply best candidate if any.
        should_stop = False
        if is_main_process():
            if best_candidate is None:
                no_gain_rounds += 1
                logger.info(
                    "rule_search iteration %d: no candidate passed gate (no_gain_rounds=%d/%d)",
                    iteration,
                    no_gain_rounds,
                    patience,
                )
            else:
                assert mission_guidance_repo is not None
                op = best_candidate["operation"]
                assert isinstance(op, ExperienceOperation)
                reflection_id = uuid.uuid4().hex[:12]
                proposal = ReflectionProposal(
                    action=ReflectionAction("refine"),
                    summary=None,
                    critique=None,
                    operations=(op,),
                    hypotheses=tuple(),
                    evidence_group_ids=tuple(reflect_keys),
                    uncertainty_note=None,
                    no_evidence_group_ids=tuple(),
                    text=None,
                )
                before = mission_guidance_repo.get(mission)
                updated = mission_guidance_repo.apply_reflection(
                    mission=mission,
                    proposal=proposal,
                    reflection_id=reflection_id,
                    source_group_ids=list(reflect_keys),
                    operations=(op,),
                    applied_epoch=iteration,
                )
                new_keys = sorted(
                    set(updated.experiences.keys()) - set(before.experiences.keys())
                )
                accepted_key = new_keys[0] if new_keys else None
                gate: object = best_candidate["gate"]
                cand_metrics = best_candidate["cand_metrics"]
                assert isinstance(cand_metrics, EvalMetrics), "cand_metrics must be EvalMetrics"
                _append_jsonl(
                    benchmarks_path,
                    {
                        "timestamp": time.time(),
                        "iteration": iteration,
                        "mission": mission,
                        "accepted_key": accepted_key,
                        "signature": best_candidate["signature"],
                        "text": best_candidate["text"],
                        "rationale": best_candidate["rationale"],
                        "validate_n": len(validate_tickets),
                        "base_acc": base_metrics.acc if base_metrics else None,
                        "base_fn_rate": base_metrics.fn_rate if base_metrics else None,
                        "base_fp_rate": base_metrics.fp_rate if base_metrics else None,
                        "base_fn_over_tp": base_metrics.fn_over_tp if base_metrics else None,
                        "base_fp_over_tp": base_metrics.fp_over_tp if base_metrics else None,
                        "after_acc": cand_metrics.acc,
                        "after_fn_rate": cand_metrics.fn_rate,
                        "after_fp_rate": cand_metrics.fp_rate,
                        "after_fn_over_tp": cand_metrics.fn_over_tp,
                        "after_fp_over_tp": cand_metrics.fp_over_tp,
                        "relative_error_reduction": getattr(
                            gate, "relative_error_reduction", None
                        ),
                        "changed_fraction": getattr(gate, "changed_fraction", None),
                        "bootstrap_prob": getattr(gate, "bootstrap_prob", None),
                        "guidance_step_before": before.step,
                        "guidance_step_after": updated.step,
                        "reflect_ticket_keys": list(reflect_keys),
                    },
                )
                no_gain_rounds = 0
                logger.info(
                    "rule_search iteration %d: accepted %s (key=%s, RER=%.3f)",
                    iteration,
                    best_candidate["signature"],
                    accepted_key,
                    getattr(gate, "relative_error_reduction", 0.0),
                )

            should_stop = no_gain_rounds >= patience

        should_stop = bool(
            broadcast_int(1 if should_stop and is_main_process() else 0, src=0)
        )
        no_gain_rounds = broadcast_int(
            no_gain_rounds if is_main_process() else 0, src=0
        )
        if should_stop:
            break


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", path)
                continue
            if isinstance(payload, dict):
                items.append(payload)
    return items


def _write_need_review_summary(
    *,
    mission: str,
    need_review_queue_path: Path,
    output_path: Path,
) -> None:
    items = _load_jsonl(need_review_queue_path)
    items_sorted = sorted(
        items,
        key=lambda item: (
            str(item.get("ticket_key") or ""),
            str(item.get("group_id") or ""),
            str(item.get("reason_code") or ""),
        ),
    )
    by_reason: Dict[str, int] = defaultdict(int)
    for item in items_sorted:
        reason_code = str(item.get("reason_code") or item.get("reason") or "unknown")
        by_reason[reason_code] += 1

    summary = {
        "mission": mission,
        "n": len(items_sorted),
        "by_reason_code": dict(sorted(by_reason.items(), key=lambda kv: kv[0])),
        "items": items_sorted,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _unexplainable_need_review_items(
    records: List[ExperienceRecord],
    *,
    outcome: ReflectionOutcome,
    epoch: int,
    reflection_cycle: int,
) -> List[Tuple[ExperienceRecord, dict]]:
    """Build need-review entries for tickets that reflection marked as unexplainable after GT.

    Semantics (Stage-B training-free runtime):
    - Need-review is a label-suspect queue: after seeing gt_label, reflection still
      cannot explain how to support the GT direction for some tickets.
    - Reflection MUST decide at group_id granularity via `proposal.no_evidence_group_ids`.
      A global `uncertainty_note == "no_evidence_for_label"` is treated as "all cases
      in this bundle are unexplainable" (fallback for legacy outputs).
    """

    items: List[Tuple[ExperienceRecord, dict]] = []
    no_evidence_group_ids = set(outcome.proposal.no_evidence_group_ids or ())
    global_no_evidence = outcome.proposal.uncertainty_note == "no_evidence_for_label"
    if global_no_evidence and not no_evidence_group_ids:
        no_evidence_group_ids = {rec.ticket.key for rec in records}

    for rec in records:
        # Reflection is the single source of truth for need-review routing.
        if rec.ticket.key not in no_evidence_group_ids:
            continue

        pred_verdict: Optional[str] = None
        pred_reason: Optional[str] = None
        if rec.winning_candidate is not None:
            win_idx = rec.winning_candidate
            win_cand = next(
                (c for c in rec.candidates if c.candidate_index == win_idx),
                None,
            )
            if win_cand is not None:
                pred_verdict = win_cand.verdict
                pred_reason = win_cand.reason
        if pred_verdict is None:
            for cand in rec.candidates:
                if cand.verdict is not None:
                    pred_verdict = cand.verdict
                    pred_reason = cand.reason
                    break

        entry = {
            "ticket_key": rec.ticket.key,
            "group_id": rec.ticket.group_id,
            "mission": rec.ticket.mission,
            "gt_label": rec.ticket.label,
            "pred_verdict": pred_verdict,
            "pred_reason": pred_reason,
            "reason_code": "reflection_no_evidence_after_gt",
            "epoch": epoch,
            "reflection_cycle": reflection_cycle,
            "reflection_id": outcome.reflection_id,
            "uncertainty_note": outcome.proposal.uncertainty_note,
        }
        items.append((rec, entry))

    return items


def _chunked(seq: List, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _shard_bounds(total: int, *, world_size: int, rank: int) -> Tuple[int, int]:
    """Return [start, end) bounds for stable sharding across ranks."""
    if world_size <= 1:
        return 0, total
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    base = total // world_size
    remainder = total % world_size

    start = rank * base + min(rank, remainder)
    count = base + (1 if rank < remainder else 0)
    end = start + count
    return start, end


def _compute_metrics(records: List[Dict[str, object]]) -> Dict[str, float]:
    correct = 0
    fn = 0
    fp = 0
    total = len(records)

    for record in records:
        verdict = record.get("model_verdict")
        gt_label = record.get("gt_label")
        if verdict is None:
            if gt_label == "pass":
                fn += 1
            else:
                fp += 1
            continue

        if verdict == gt_label:
            correct += 1
        elif verdict == "pass" and gt_label == "fail":
            fp += 1
        elif verdict == "fail" and gt_label == "pass":
            fn += 1

    acc = correct / total if total else 0.0

    fn_rate = fn / total if total else 0.0
    fp_rate = fp / total if total else 0.0

    return {"acc": acc, "fn": fn, "fp": fp, "fn_rate": fn_rate, "fp_rate": fp_rate, "n": total}


def _summarize_reflection_records(
    records: Sequence[ExperienceRecord],
) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for rec in records:
        verdicts: List[Optional[str]] = []
        candidate_details: List[Dict[str, object]] = []
        pass_count = 0
        fail_count = 0
        invalid = 0
        for cand in rec.candidates:
            verdicts.append(cand.verdict)
            if cand.verdict == "pass":
                pass_count += 1
            elif cand.verdict == "fail":
                fail_count += 1
            else:
                invalid += 1
            signals = getattr(cand, "signals", None)
            candidate_details.append(
                {
                    "candidate_index": cand.candidate_index,
                    "verdict": cand.verdict,
                    "reason": cand.reason,
                    "confidence": cand.confidence,
                    "signals": {
                        "label_match": getattr(signals, "label_match", None),
                        "low_agreement": getattr(signals, "low_agreement", None),
                        "conflict_flag": getattr(signals, "conflict_flag", None),
                        "needs_manual_review": getattr(
                            signals, "needs_manual_review", None
                        ),
                        "vote_strength": getattr(signals, "vote_strength", None),
                        "confidence": getattr(signals, "confidence", None),
                    },
                    "raw_text": cand.raw_text,
                }
            )

        winning = None
        if rec.winning_candidate is not None:
            for cand in rec.candidates:
                if cand.candidate_index == rec.winning_candidate:
                    winning = cand
                    break

        summaries.append(
            {
                "ticket_key": rec.ticket.key,
                "group_id": rec.ticket.group_id,
                "gt_label": rec.ticket.label,
                "winning_candidate": rec.winning_candidate,
                "winning_verdict": getattr(winning, "verdict", None),
                "winning_reason": getattr(winning, "reason", None),
                "winning_confidence": getattr(winning, "confidence", None),
                "winning_label_match": (
                    getattr(getattr(winning, "signals", None), "label_match", None)
                ),
                "verdict_counts": {
                    "pass": pass_count,
                    "fail": fail_count,
                    "invalid": invalid,
                },
                "candidate_verdicts": verdicts,
                "candidates": candidate_details,
                "stage_a_summaries": rec.ticket.summaries.as_dict(),
                "guidance_step": rec.guidance_step,
                "epoch_step": rec.epoch_step,
                "global_step": rec.global_step,
            }
        )

    return summaries


def _hard_case_rows(
    stats_by_ticket: Mapping[str, object],
    *,
    mission: str,
    iteration: int,
    sampler: str,
    limit: int,
    reason_samples_by_ticket: Optional[Mapping[str, Sequence[Optional[str]]]] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for entry in stats_by_ticket.values():
        majority = getattr(entry, "majority_pred", None)
        gt_label = getattr(entry, "gt_label", None)
        if majority is None or majority == gt_label:
            continue
        ticket_key = getattr(entry, "ticket_key", None)
        verdict_samples = list(getattr(entry, "verdict_samples", ()))
        reason_samples: List[Optional[str]] = []
        majority_reason = None
        reason_counts: List[Dict[str, object]] = []
        if reason_samples_by_ticket is not None and ticket_key is not None:
            raw_reasons = list(reason_samples_by_ticket.get(ticket_key, ()))
            if raw_reasons:
                reason_samples = raw_reasons
                if majority is not None:
                    counts = Counter()
                    for verdict, reason in zip(verdict_samples, raw_reasons):
                        if verdict == majority and reason:
                            counts[reason] += 1
                    if counts:
                        majority_reason = counts.most_common(1)[0][0]
                        reason_counts = [
                            {"reason": reason, "count": count}
                            for reason, count in counts.most_common(5)
                        ]

        rows.append(
            {
                "timestamp": time.time(),
                "iteration": iteration,
                "mission": mission,
                "sampler": sampler,
                "ticket_key": ticket_key,
                "gt_label": gt_label,
                "majority_pred": majority,
                "majority_reason": majority_reason,
                "pass_count": getattr(entry, "pass_count", None),
                "fail_count": getattr(entry, "fail_count", None),
                "invalid_count": getattr(entry, "invalid_count", None),
                "total_samples": getattr(entry, "total_samples", None),
                "agreement": getattr(entry, "agreement", None),
                "difficulty": getattr(entry, "difficulty", None),
                "hard_wrong": getattr(entry, "hard_wrong", None),
                "verdict_samples": verdict_samples,
                "reason_samples": reason_samples,
                "reason_counts": reason_counts,
            }
        )

    rows.sort(
        key=lambda row: (
            float(row.get("hard_wrong") or 0.0),  # type: ignore[arg-type]
            float(row.get("difficulty") or 0.0),  # type: ignore[arg-type]
        ),
        reverse=True,
    )
    if limit > 0:
        rows = rows[:limit]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _candidate_regressions(
    base_stats: Mapping[str, object],
    new_stats: Mapping[str, object],
    *,
    limit: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    keys = sorted(set(base_stats.keys()) & set(new_stats.keys()))
    for key in keys:
        base = base_stats[key]
        new = new_stats[key]
        base_correct = bool(getattr(base, "majority_correct", False))
        new_correct = bool(getattr(new, "majority_correct", False))
        if not base_correct or new_correct:
            continue
        rows.append(
            {
                "ticket_key": getattr(base, "ticket_key", key),
                "gt_label": getattr(base, "gt_label", None),
                "base_pred": getattr(base, "majority_pred", None),
                "cand_pred": getattr(new, "majority_pred", None),
                "base_agreement": getattr(base, "agreement", None),
                "cand_agreement": getattr(new, "agreement", None),
                "base_pass_count": getattr(base, "pass_count", None),
                "base_fail_count": getattr(base, "fail_count", None),
                "cand_pass_count": getattr(new, "pass_count", None),
                "cand_fail_count": getattr(new, "fail_count", None),
            }
        )

    rows.sort(
        key=lambda row: (
            float(row.get("cand_agreement") or 0.0),  # type: ignore[arg-type]
            float(row.get("base_agreement") or 0.0),  # type: ignore[arg-type]
        ),
        reverse=True,
    )
    if limit > 0:
        rows = rows[:limit]
    return rows


def _setup_mission_guidance(
    startup_path: Path,
    mission_dir: Path,
    mission: str,
    retention: int,
    *,
    reset: bool = False,
) -> GuidanceRepository:
    """Load initial guidance from startup path and copy to mission directory.

    NOTE: Guidance updates are applied ONLY to the mission-specific guidance.json
    under {mission_dir}/guidance.json. They are NOT written back to the global
    startup_path file. Promotion of learned guidance to the global file is
    DEFERRED for future implementation and must be done manually if needed.

    Args:
        startup_path: Original guidance.json path (unchanged)
        mission_dir: Directory for this mission ({root}/{mission_name}/{run_name})
        mission: Mission name
        retention: Retention count for snapshots

    Returns:
        GuidanceRepository initialized with mission-specific guidance.json
    """
    mission_guidance_path = mission_dir / "guidance.json"
    mission_dir.mkdir(parents=True, exist_ok=True)

    if mission_guidance_path.exists() and not reset:
        repo = GuidanceRepository(
            mission_guidance_path,
            retention=retention,
        )
        try:
            repo.load()
            logger.info(
                "Reusing existing mission guidance for %s at %s",
                mission,
                mission_guidance_path,
            )
            return repo
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Existing mission guidance %s could not be loaded (%s); re-seeding from startup",
                mission_guidance_path,
                exc,
            )

    # Load startup/global guidance and extract mission section; fail fast if missing
    startup_repo = GuidanceRepository(startup_path, retention=retention)
    startup_map = startup_repo.load()
    if mission not in startup_map:
        raise RuntimeError(
            f"Mission {mission} not found in global guidance file: {startup_path}"
        )
    seed_section = startup_map[mission].to_payload()

    if mission_guidance_path.exists() and reset:
        mission_guidance_path.unlink()

    # Write only this mission section into mission-specific guidance.json
    with mission_guidance_path.open("w", encoding="utf-8") as fh:
        json.dump({mission: seed_section}, fh, ensure_ascii=False, indent=2)

    repo = GuidanceRepository(
        mission_guidance_path,
        retention=retention,
    )

    logger.info(
        f"Initialized mission guidance for {mission} at {mission_guidance_path} (seeded from {startup_path}, step={seed_section.get('step')})"
    )

    return repo


def run_all(config: StageBConfig, log_level: str = "logging") -> None:
    level_map = {
        "debug": logging.DEBUG,
        "logging": logging.INFO,
        "warning": logging.WARNING,
    }
    normalized = log_level.strip().lower()
    if normalized not in level_map:
        raise ValueError(
            f"Unsupported log level '{log_level}'. Choose from: {', '.join(level_map)}"
        )
    configure_logging(
        level=level_map[normalized], debug=(normalized == "debug"), verbose=False
    )

    logger.info(
        "Stage-B starting (mode=%s)", getattr(config, "mode", "legacy_reflection")
    )

    init_distributed()
    enable_tf32()
    world_size = get_world_size()
    rank = get_rank()
    distributed = world_size > 1
    if distributed and is_main_process():
        logger.info(
            "Stage-B distributed ticket-parallel rollout enabled (world_size=%d)",
            world_size,
        )

    seed_everything(config.seed)

    # Ingest all tickets to discover missions (needed for directory structure)
    logger.info("Ingesting Stage-A outputs")
    tickets = list(ingest_stage_a(config.stage_a_paths))
    if not tickets:
        raise RuntimeError("No Stage-A records ingested; aborting Stage-B run")

    # Group tickets by mission
    tickets_by_mission: Dict[str, List[GroupTicket]] = defaultdict(list)
    for ticket in tickets:
        tickets_by_mission[ticket.mission].append(ticket)

    logger.info(
        f"Discovered {len(tickets_by_mission)} mission(s): {list(tickets_by_mission.keys())}"
    )

    if len(tickets_by_mission) != 1:
        raise RuntimeError(
            f"Stage-B reflection-first pipeline expects exactly one mission per run_name; got {len(tickets_by_mission)} missions"
        )

    # Get the single mission name for directory structure: {root}/{mission_name}/{run_name}/
    mission_name = list(tickets_by_mission.keys())[0]
    run_dir = config.output.root / mission_name / config.output.run_name
    if distributed:
        if is_main_process():
            if run_dir.exists():
                logger.info(
                    "Cleaning existing run directory to avoid stale artifacts: %s",
                    run_dir,
                )
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        barrier()
    else:
        if run_dir.exists():
            logger.info(
                "Cleaning existing run directory to avoid stale artifacts: %s", run_dir
            )
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    training_by_mission = tickets_by_mission

    logger.info(f"Loading model {config.model.model_name_or_path}")
    model, tokenizer, processor = _load_model(config)

    if getattr(config, "mode", "legacy_reflection") == "rule_search":
        # rule_search mode runs a metric-gated loop; legacy reflection/need-review are disabled.
        mission = mission_name
        domain = resolve_domain_for_mission(config, mission)
        _run_rule_search_mission(
            config=config,
            model=model,
            tokenizer=tokenizer,
            mission=mission,
            mission_tickets=training_by_mission[mission],
            mission_dir=run_dir,
            domain=domain,
        )
        logger.info("Completed Stage-B rule_search run for mission %s", mission)
        return

    # Legacy reflection-first pipeline (default)
    if config.sampler is None:
        raise ValueError("sampler is required for legacy_reflection mode")
    sampler = RolloutSampler(model=model, tokenizer=tokenizer, config=config.sampler)

    # Process each mission separately
    total_selections = 0
    processed_missions = 0
    for mission, mission_tickets in training_by_mission.items():
        if not mission_tickets:
            logger.warning(f"Skipping mission {mission} because no tickets available")
            continue

        logger.info(f"Processing mission: {mission} ({len(mission_tickets)} tickets)")
        domain_map = {mission: resolve_domain_for_mission(config, mission)}
        logger.info("Stage-B domain: %s", domain_map[mission])

        processed_missions += 1

        # Setup mission-specific directory structure
        # run_dir is already {mission_name}/{run_name}, so mission_dir = run_dir
        mission_dir = run_dir
        distill_cfg = config.stage_b_distillation
        distill_enabled = bool(distill_cfg.enabled) if distill_cfg else False
        default_distill_path = mission_dir / "distill_chatml.jsonl"
        distill_path = (
            Path(distill_cfg.log_chatml_path)
            if distill_enabled and distill_cfg and distill_cfg.log_chatml_path
            else default_distill_path
        )

        mission_guidance_repo: Optional[GuidanceRepository] = None
        reflection_engine: Optional[ReflectionEngine] = None
        hypothesis_pool: Optional[HypothesisPool] = None
        trajectories_path = mission_dir / "trajectories.jsonl"
        selections_path = mission_dir / "selections.jsonl"
        metrics_path = mission_dir / "metrics.jsonl"
        reflection_log_path = mission_dir / "reflection.jsonl"
        logging_steps = config.runner.logging_steps
        global_step = 0
        group_report_delta_path = mission_dir / "group_report_delta.jsonl"

        if is_main_process():
            _reset_mission_artifacts(mission_dir)

            # Setup mission-specific guidance
            mission_guidance_repo = _setup_mission_guidance(
                startup_path=config.guidance.path,
                mission_dir=mission_dir,
                mission=mission,
                retention=config.guidance.retention,
                reset=config.guidance.reset_on_rerun,
            )

            # Prepare mission-specific output paths
            trajectories_path, selections_path = _prepare_mission_output_paths(
                mission_dir
            )

            # Reflection log goes under {root}/{mission_name}/{run_name}/
            reflection_engine = ReflectionEngine(
                model=model,
                tokenizer=tokenizer,
                config=config.reflection,
                guidance_repo=mission_guidance_repo,
                reflection_log=reflection_log_path,
            )
            hypothesis_pool = HypothesisPool(
                pool_path=mission_dir / "hypotheses.json",
                events_path=mission_dir / "hypothesis_events.jsonl",
                min_support_cycles=config.reflection.hypothesis_min_support_cycles,
                min_unique_ticket_keys=config.reflection.hypothesis_min_unique_ticket_keys,
            )

        mission_selection_count = 0
        total_groups = len(mission_tickets)
        need_review_path = mission_dir / "need_review_queue.jsonl"
        failure_path = mission_dir / "failure_malformed.jsonl"
        reflection_cycle = 0
        last_reflection_id: Optional[str] = None
        last_applied_rule_keys: List[str] = []  # Track rules applied in last reflection
        group_delta_state: Dict[str, Dict[str, object]] = {}
        epoch = 0
        # Initialize variables that may be used after the epoch loop
        guidance_updated_epoch: bool = False
        distill_records_epoch: List[Dict[str, object]] = []
        for epoch in range(1, config.runner.epochs + 1):
            guidance_updated_epoch = False
            distill_records_epoch = []
            # Per-epoch reflection bookkeeping (non-sticky)
            pending_records: List[ExperienceRecord] = []  # grad-candidates (attempt=0)
            retry_queues: DefaultDict[int, List[ExperienceRecord]] = defaultdict(list)
            retry_counts: Dict[str, int] = {}
            pending_feedback: Dict[str, _PendingRuleFeedback] = {}
            reflection_calls_epoch = 0
            applied_changes_epoch = 0
            epoch_outcomes: Dict[str, Dict[str, object]] = {}
            epoch_step = 0
            window_ticket_keys: List[str] = []
            window_start_epoch_step = 1
            epoch_start_time = time.time()
            window_start_time = epoch_start_time

            def _record_group_delta(
                ticket: GroupTicket,
                *,
                guidance_step: int,
                model_verdict: Optional[str],
                in_manual_review: bool,
                manual_reason: Optional[str] = None,
                selection_payload: Optional[Dict[str, object]] = None,
                candidates_payload: Optional[List[Dict[str, object]]] = None,
            ) -> None:
                group_delta_state[ticket.key] = {
                    "ticket_key": ticket.key,
                    "group_id": ticket.group_id,
                    "mission": ticket.mission,
                    "gt_label": ticket.label,
                    "model_verdict": model_verdict,
                    "in_manual_review": in_manual_review,
                    "manual_reason": manual_reason,
                    "selection": selection_payload,
                    "candidates": candidates_payload or [],
                    "guidance_step": guidance_step,
                    "reflection_cycle": reflection_cycle,
                    "last_reflection_id": last_reflection_id,
                }

            def record_outcome(
                ticket: GroupTicket,
                *,
                model_verdict: Optional[str],
                in_manual_review: bool,
            ) -> None:
                ticket_key = ticket.key
                epoch_outcomes[ticket_key] = {
                    "group_id": ticket.group_id,
                    "mission": ticket.mission,
                    "gt_label": ticket.label,
                    "model_verdict": model_verdict,
                    "in_manual_review": in_manual_review,
                }

            def _flush_metrics_window(*, event: str) -> None:
                nonlocal window_ticket_keys, window_start_epoch_step, window_start_time
                assert is_main_process()
                if not window_ticket_keys:
                    return

                window_records: List[Dict[str, object]] = []
                for key in window_ticket_keys:
                    record = epoch_outcomes.get(key)
                    if record is not None:
                        window_records.append(record)

                clean_records = [
                    record
                    for record in window_records
                    if not record.get("in_manual_review")
                ]
                metrics_exclude_mr = _compute_metrics(clean_records)
                metrics_include_mr = _compute_metrics(window_records)

                now = time.time()
                window_secs = max(now - window_start_time, 1e-6)
                groups_per_sec = len(window_records) / window_secs
                elapsed = now - epoch_start_time
                epoch_groups_per_sec = epoch_step / max(elapsed, 1e-6)
                remaining_groups_epoch = max(total_groups - epoch_step, 0)
                eta_seconds_epoch = (
                    remaining_groups_epoch / max(epoch_groups_per_sec, 1e-9)
                    if remaining_groups_epoch
                    else 0.0
                )
                eta_hours_epoch = eta_seconds_epoch / 3600.0
                total_steps_run = total_groups * config.runner.epochs
                remaining_groups_run = (
                    remaining_groups_epoch
                    + max(config.runner.epochs - epoch, 0) * total_groups
                )
                eta_seconds_run = (
                    remaining_groups_run / max(epoch_groups_per_sec, 1e-9)
                    if remaining_groups_run
                    else 0.0
                )
                eta_hours_run = eta_seconds_run / 3600.0

                logger.info(
                    "Step %d/%d | epoch %d/%d | step %d/%d | %s (%d groups, %.2f groups/s, %.1fs elapsed, eta=%.2fh epoch, %.2fh run) | exc: acc=%.4f fn=%d fp=%d fn_rate=%.4f fp_rate=%.4f n=%d | inc: acc=%.4f fn=%d fp=%d fn_rate=%.4f fp_rate=%.4f n=%d",
                    global_step,
                    total_steps_run,
                    epoch,
                    config.runner.epochs,
                    epoch_step,
                    total_groups,
                    event,
                    len(window_records),
                    groups_per_sec,
                    elapsed,
                    eta_hours_epoch,
                    eta_hours_run,
                    metrics_exclude_mr["acc"],
                    metrics_exclude_mr["fn"],
                    metrics_exclude_mr["fp"],
                    metrics_exclude_mr["fn_rate"],
                    metrics_exclude_mr["fp_rate"],
                    metrics_exclude_mr["n"],
                    metrics_include_mr["acc"],
                    metrics_include_mr["fn"],
                    metrics_include_mr["fp"],
                    metrics_include_mr["fn_rate"],
                    metrics_include_mr["fp_rate"],
                    metrics_include_mr["n"],
                )

                _append_jsonl(
                    metrics_path,
                    {
                        "event": event,
                        "epoch": epoch,
                        "epoch_step_start": window_start_epoch_step,
                        "epoch_step_end": epoch_step,
                        "global_step": global_step,
                        "logging_steps": logging_steps,
                        "elapsed_seconds": elapsed,
                        "groups_per_second": groups_per_sec,
                        "eta_seconds_epoch": eta_seconds_epoch,
                        "eta_seconds_run": eta_seconds_run,
                        "exclude_manual_review": metrics_exclude_mr,
                        "include_manual_review": metrics_include_mr,
                    },
                )

                if config.output.group_report:
                    for ticket_key in window_ticket_keys:
                        payload = group_delta_state.get(ticket_key)
                        if payload is None:
                            continue
                        _append_jsonl(
                            group_report_delta_path,
                            {
                                "logged_event": event,
                                "logged_epoch": epoch,
                                "logged_epoch_step": epoch_step,
                                "logged_global_step": global_step,
                                "logging_steps": logging_steps,
                                **payload,
                            },
                        )

                window_ticket_keys = []
                window_start_epoch_step = epoch_step + 1
                window_start_time = time.time()

            def _mark_group_step(ticket_key: str) -> None:
                window_ticket_keys.append(ticket_key)
                if epoch_step % logging_steps == 0:
                    _flush_metrics_window(event="logging_steps")

            def _commit_pending_feedback(
                feedback: _PendingRuleFeedback,
            ) -> None:
                if not feedback.experience_keys:
                    return
                if mission_guidance_repo is not None:
                    if feedback.label_match:
                        mission_guidance_repo.increment_hit_count(
                            mission, list(feedback.experience_keys)
                        )
                    else:
                        mission_guidance_repo.increment_miss_count(
                            mission, list(feedback.experience_keys)
                        )

            def _extract_pred(
                rec: ExperienceRecord,
            ) -> Tuple[Optional[str], Optional[str]]:
                pred_verdict: Optional[str] = None
                pred_reason: Optional[str] = None
                if rec.winning_candidate is not None:
                    win_idx = rec.winning_candidate
                    win_cand = next(
                        (c for c in rec.candidates if c.candidate_index == win_idx),
                        None,
                    )
                    if win_cand is not None:
                        pred_verdict = win_cand.verdict
                        pred_reason = win_cand.reason
                if pred_verdict is None:
                    for cand in rec.candidates:
                        if cand.verdict is not None:
                            pred_verdict = cand.verdict
                            pred_reason = cand.reason
                            break
                return pred_verdict, pred_reason

            def _enqueue_need_review(
                rec: ExperienceRecord,
                *,
                reason_code: str,
                epoch: int,
                reflection_cycle: int,
                reflection_id: Optional[str],
                uncertainty_note: Optional[str] = None,
            ) -> None:
                pred_verdict, pred_reason = _extract_pred(rec)
                entry = {
                    "ticket_key": rec.ticket.key,
                    "group_id": rec.ticket.group_id,
                    "mission": rec.ticket.mission,
                    "gt_label": rec.ticket.label,
                    "pred_verdict": pred_verdict,
                    "pred_reason": pred_reason,
                    "reason_code": reason_code,
                    "epoch": epoch,
                    "reflection_cycle": reflection_cycle,
                    "reflection_id": reflection_id,
                    "uncertainty_note": uncertainty_note,
                }
                _append_jsonl(need_review_path, entry)
                record_outcome(
                    rec.ticket,
                    model_verdict=pred_verdict,
                    in_manual_review=True,
                )
                delta = group_delta_state.get(rec.ticket.key)
                if delta is not None:
                    delta["in_manual_review"] = True
                    delta["manual_reason"] = reason_code
                    delta["model_verdict"] = pred_verdict
                    delta["reflection_cycle"] = reflection_cycle
                    delta["last_reflection_id"] = reflection_id

            def _flush_gradient_candidates(
                current_epoch: int,
                context: str,
                *,
                flush_partial: bool,
            ) -> None:
                nonlocal \
                    pending_records, \
                    retry_queues, \
                    retry_counts, \
                    pending_feedback, \
                    reflection_cycle, \
                    last_reflection_id, \
                    last_applied_rule_keys, \
                    guidance_updated_epoch, \
                    reflection_calls_epoch, \
                    applied_changes_epoch, \
                    hypothesis_pool
                assert is_main_process()
                assert reflection_engine is not None
                assert mission_guidance_repo is not None

                retry_budget = config.reflection.retry_budget_per_group_per_epoch
                max_calls = config.reflection.max_calls_per_epoch

                def _pop_batch(
                    queue: List[ExperienceRecord], *, size: int
                ) -> List[ExperienceRecord]:
                    batch = queue[:size]
                    del queue[:size]
                    return batch

                def _pop_next_batch(
                    *, flush_partial: bool
                ) -> Tuple[int, List[ExperienceRecord]]:
                    # attempt=0 first (insertion order), then retry buckets (stable by group_id).
                    if pending_records:
                        size = _batch_size_for_retry(
                            config.reflection.batch_size, attempt=0
                        )
                        if len(pending_records) >= size or flush_partial:
                            take = (
                                min(len(pending_records), size)
                                if flush_partial
                                else size
                            )
                            return 0, _pop_batch(pending_records, size=take)
                    for attempt in sorted(retry_queues.keys()):
                        queue = retry_queues[attempt]
                        if not queue:
                            continue
                        queue.sort(key=lambda r: r.ticket.key)
                        size = _batch_size_for_retry(
                            config.reflection.batch_size, attempt=attempt
                        )
                        if len(queue) >= size or flush_partial:
                            take = min(len(queue), size) if flush_partial else size
                            return attempt, _pop_batch(queue, size=take)
                    return -1, []

                while True:
                    attempt, batch_records = _pop_next_batch(
                        flush_partial=flush_partial
                    )
                    if attempt < 0 or not batch_records:
                        break

                    if max_calls is not None and reflection_calls_epoch >= max_calls:
                        for rec in batch_records:
                            pending_feedback.pop(rec.ticket.key, None)
                            _enqueue_need_review(
                                rec,
                                reason_code="budget_exhausted",
                                epoch=current_epoch,
                                reflection_cycle=reflection_cycle,
                                reflection_id=None,
                            )
                        continue

                    # Snapshot guidance before any apply in this cycle.
                    guidance_step_before = batch_records[0].guidance_step
                    before_experience_keys: set[str] = set()
                    try:
                        guidance_map_before = mission_guidance_repo.load()
                        if mission in guidance_map_before:
                            guidance_step_before = guidance_map_before[mission].step
                            before_experience_keys = set(
                                guidance_map_before[mission].experiences.keys()
                            )
                    except Exception:
                        before_experience_keys = set()

                    bundle = ExperienceBundle(
                        mission=mission,
                        records=tuple(batch_records),
                        reflection_cycle=reflection_cycle,
                        guidance_step=guidance_step_before,
                    )

                    reflection_id = uuid.uuid4().hex[:12]
                    warnings: List[str] = []
                    ineligible_reason: Optional[str] = None
                    applied = False
                    guidance_step_after = guidance_step_before

                    # ----------------------
                    # Pass-1: stop-gradient decision
                    # ----------------------
                    reflection_calls_epoch += 1
                    try:
                        stop_ids_tuple, decision_analysis = (
                            reflection_engine.run_decision_pass(bundle)
                        )
                    except Exception as exc:  # noqa: BLE001
                        stop_ids_tuple = tuple()
                        decision_analysis = ""
                        ineligible_reason = f"decision_error: {exc}"
                        warnings.append("decision_error")

                    stop_ids = set(stop_ids_tuple)
                    _drain_buffered_feedback(
                        pending_feedback,
                        stop_gradient_ticket_keys=stop_ids,
                        contributor_ticket_keys=set(),
                    )
                    for rec in batch_records:
                        if rec.ticket.key in stop_ids:
                            _enqueue_need_review(
                                rec,
                                reason_code="reflection_no_evidence_after_gt",
                                epoch=current_epoch,
                                reflection_cycle=reflection_cycle,
                                reflection_id=reflection_id,
                                uncertainty_note=decision_analysis or None,
                            )

                    learnable_records = [
                        rec for rec in batch_records if rec.ticket.key not in stop_ids
                    ]
                    learnable_ids = {rec.ticket.key for rec in learnable_records}

                    operations: Tuple[ExperienceOperation, ...] = tuple()
                    hypotheses = tuple()
                    evidence_group_ids: Tuple[str, ...] = tuple()
                    evidence_analysis = ""

                    # ----------------------
                    # Pass-2: ops on learnable-only
                    # ----------------------
                    if ineligible_reason is None and learnable_records:
                        if (
                            max_calls is not None
                            and reflection_calls_epoch >= max_calls
                        ):
                            warnings.append("budget_exhausted_before_ops")
                            for rec in learnable_records:
                                pending_feedback.pop(rec.ticket.key, None)
                                _enqueue_need_review(
                                    rec,
                                    reason_code="budget_exhausted",
                                    epoch=current_epoch,
                                    reflection_cycle=reflection_cycle,
                                    reflection_id=reflection_id,
                                )
                            learnable_records = []
                            learnable_ids = set()
                        else:
                            reflection_calls_epoch += 1
                            learnable_bundle = ExperienceBundle(
                                mission=mission,
                                records=tuple(learnable_records),
                                reflection_cycle=reflection_cycle,
                                guidance_step=guidance_step_before,
                            )
                            try:
                                (
                                    operations,
                                    hypotheses,
                                    evidence_group_ids,
                                    evidence_analysis,
                                ) = reflection_engine.run_ops_pass(learnable_bundle)
                            except Exception as exc:  # noqa: BLE001
                                ineligible_reason = f"ops_error: {exc}"
                                warnings.append("ops_error")
                                operations = tuple()
                                hypotheses = tuple()
                                evidence_group_ids = tuple()

                    evidence_ops = tuple(
                        dict.fromkeys(
                            eid for op in operations for eid in (op.evidence or ())
                        )
                    )
                    evidence_hypotheses = tuple(
                        dict.fromkeys(
                            eid for hyp in hypotheses for eid in (hyp.evidence or ())
                        )
                    )
                    contributors, uncovered = _compute_learnability_coverage(
                        learnable_ids,
                        evidence_ops,
                        evidence_hypotheses,
                    )
                    if uncovered:
                        warnings.append(f"uncovered={len(uncovered)}")
                    record_by_gid = {rec.ticket.key: rec for rec in learnable_records}

                    # Apply ops (if any) with epoch-level change cap.
                    promotion_ops: List[ExperienceOperation] = []
                    promoted_signatures: List[str] = []
                    current_evidence_map: Dict[str, Tuple[str, ...]] = {}
                    if hypotheses and hypothesis_pool is not None:
                        current_evidence_map = (
                            hypothesis_pool.build_current_evidence_map(hypotheses)
                        )
                        cap = config.reflection.change_cap_per_epoch
                        allow_promote = not (
                            cap is not None and applied_changes_epoch >= cap
                        )
                        eligible = hypothesis_pool.record_proposals(
                            hypotheses,
                            reflection_cycle=reflection_cycle,
                            epoch=current_epoch,
                            allow_promote=allow_promote,
                        )
                        if eligible:
                            for rec in eligible:
                                evidence = current_evidence_map.get(rec.signature)
                                if not evidence:
                                    evidence = tuple(
                                        tk
                                        for tk in rec.support_ticket_keys
                                        if tk in learnable_ids
                                    )
                                evidence = tuple(dict.fromkeys(evidence))
                                if not evidence:
                                    warnings.append("promotion_no_current_evidence")
                                    continue
                                promotion_ops.append(
                                    ExperienceOperation(
                                        op="upsert",
                                        key=None,
                                        text=rec.text,
                                        rationale="hypothesis_promotion",
                                        evidence=evidence,
                                    )
                                )
                                promoted_signatures.append(rec.signature)
                    if promotion_ops:
                        operations = tuple(list(operations) + promotion_ops)

                    rollout_trace = _summarize_reflection_records(batch_records)
                    coverage_ratio = (
                        1.0 - (len(uncovered) / len(learnable_ids))
                        if learnable_ids
                        else 0.0
                    )
                    trace_payload = {
                        "batch_size": len(batch_records),
                        "learnable_size": len(learnable_records),
                        "stop_gradient": sorted(stop_ids),
                        "learnable_ticket_keys": sorted(learnable_ids),
                        "contributors": sorted(contributors),
                        "uncovered": sorted(uncovered),
                        "coverage_ratio": coverage_ratio,
                        "operations_count": len(operations),
                        "hypotheses_count": len(hypotheses),
                        "promotion_signatures": promoted_signatures,
                        "decision_analysis": decision_analysis or None,
                        "evidence_analysis": evidence_analysis or None,
                        "rollout_records": rollout_trace,
                    }

                    action = "refine" if operations else "noop"
                    proposal = ReflectionProposal(
                        action=ReflectionAction(action),  # type: ignore[arg-type]
                        summary=decision_analysis or None,
                        critique=evidence_analysis or None,
                        operations=tuple(operations),
                        hypotheses=tuple(hypotheses),
                        evidence_group_ids=tuple(evidence_group_ids),
                        uncertainty_note=None,
                        no_evidence_group_ids=tuple(sorted(stop_ids)),
                        text=None,
                    )

                    if operations:
                        cap = config.reflection.change_cap_per_epoch
                        if cap is not None and applied_changes_epoch >= cap:
                            warnings.append("epoch_change_cap_reached")
                        else:
                            try:
                                updated_guidance = (
                                    mission_guidance_repo.apply_reflection(
                                        mission=mission,
                                        proposal=proposal,
                                        reflection_id=reflection_id,
                                        source_group_ids=list(evidence_group_ids),
                                        applied_epoch=current_epoch,
                                        operations=operations,
                                    )
                                )
                                guidance_step_after = updated_guidance.step
                                applied = True
                                applied_changes_epoch += 1
                                guidance_updated_epoch = True
                                if promoted_signatures and hypothesis_pool is not None:
                                    hypothesis_pool.mark_promoted(
                                        promoted_signatures,
                                        reflection_cycle=reflection_cycle,
                                        epoch=current_epoch,
                                    )
                            except Exception as exc:  # noqa: BLE001
                                ineligible_reason = str(exc)
                                warnings.append("apply_failed")

                    outcome = ReflectionOutcome(
                        reflection_id=reflection_id,
                        mission=mission,
                        proposal=proposal,
                        applied=applied,
                        guidance_step_before=guidance_step_before,
                        guidance_step_after=guidance_step_after,
                        operations=tuple(operations),
                        eligible=ineligible_reason is None,
                        applied_epoch=current_epoch if applied else None,
                        ineligible_reason=ineligible_reason,
                        warnings=tuple(warnings),
                    )
                    reflection_engine._append_log(
                        outcome,
                        epoch=current_epoch,
                        trace=trace_payload,
                    )

                    if applied:
                        applied_keys: set[str] = set()
                        for op in operations:
                            if op.op in {"upsert", "merge"} and op.key:
                                applied_keys.add(op.key)
                        try:
                            guidance_map_after = mission_guidance_repo.load()
                            if mission in guidance_map_after:
                                after_keys = set(
                                    guidance_map_after[mission].experiences.keys()
                                )
                                applied_keys |= after_keys - before_experience_keys
                        except Exception:
                            pass
                        last_applied_rule_keys = sorted(applied_keys)
                        logger.debug(
                            "Reflection applied %d rule(s): %s",
                            len(applied_keys),
                            applied_keys,
                        )
                        last_reflection_id = reflection_id

                    # Closure and buffered feedback:
                    # - Normal case: commit feedback for contributor tickets (E) and retry uncovered (L\\E).
                    # - If apply failed, no learning occurred; treat all learnable tickets as retry targets
                    #   and defer feedback commit until a terminal classification (stop-gradient or final contributor).
                    retry_targets = set(uncovered)
                    if "apply_failed" in warnings and learnable_ids:
                        retry_targets |= set(learnable_ids)
                        warnings.append(f"retry_apply_failed={len(learnable_ids)}")

                    final_contributors = contributors - retry_targets
                    if final_contributors:
                        committed_feedback, _dropped_feedback = (
                            _drain_buffered_feedback(
                                pending_feedback,
                                stop_gradient_ticket_keys=set(),
                                contributor_ticket_keys=final_contributors,
                            )
                        )
                        for feedback in committed_feedback:
                            _commit_pending_feedback(feedback)

                    # Closure: uncovered learnable groups must be retried (bounded).
                    for gid in sorted(retry_targets):
                        next_retry = retry_counts.get(gid, 0) + 1
                        if next_retry <= retry_budget:
                            retry_counts[gid] = next_retry
                            retry_queues[next_retry].append(record_by_gid[gid])
                        else:
                            pending_feedback.pop(gid, None)
                            _enqueue_need_review(
                                record_by_gid[gid],
                                reason_code="budget_exhausted",
                                epoch=current_epoch,
                                reflection_cycle=reflection_cycle,
                                reflection_id=reflection_id,
                            )

                    reflection_cycle += 1

            logger.info(
                "***** Running epoch %d/%d for mission %s *****",
                epoch,
                config.runner.epochs,
                mission,
            )
            ordered_indices = broadcast_object(
                _shuffle_indices(total_groups, epoch=epoch, base_seed=config.seed)
                if is_main_process()
                else None,
                src=0,
            )

            epoch_tickets = [mission_tickets[i] for i in ordered_indices]

            # per_rank_rollout_batch_size is per-rank; global batch = per_rank × world_size
            global_batch_size = config.runner.per_rank_rollout_batch_size * max(
                world_size, 1
            )

            for batch in _chunked(epoch_tickets, global_batch_size):
                guidance_map = None
                if is_main_process():
                    assert mission_guidance_repo is not None
                    guidance_map = mission_guidance_repo.load()
                guidance_map = broadcast_object(guidance_map, src=0)

                logger.debug(
                    "Rollout batch len=%d (per_rank=%d, global=%d, world_size=%d), grid=%d, samples_per_decode=%d",
                    len(batch),
                    config.runner.per_rank_rollout_batch_size,
                    global_batch_size,
                    world_size,
                    len(config.sampler.grid),
                    config.sampler.samples_per_decode,
                )

                shard_start, shard_end = _shard_bounds(
                    len(batch), world_size=world_size, rank=rank
                )
                shard = batch[shard_start:shard_end]
                shard_parsed_map = (
                    sampler.generate_for_batch(shard, guidance_map, domain_map)
                    if shard
                    else {}
                )
                maybe_empty_cache("runner.rollout_batch")
                gathered = gather_object(shard_parsed_map, dst=0)
                if not is_main_process():
                    continue
                assert gathered is not None
                # After continue, we're guaranteed to be on main process
                assert reflection_engine is not None
                assert mission_guidance_repo is not None

                parsed_map: Dict[str, List] = {}
                for partial in gathered:
                    for ticket_key, candidates in partial.items():
                        parsed_map[ticket_key] = candidates

                for ticket in batch:
                    epoch_step += 1
                    global_step += 1
                    guidance = guidance_map[ticket.mission]

                    logger.debug(
                        f"Sampling group {ticket.group_id} (mission={ticket.mission}) at guidance.step={guidance.step}"
                    )

                    parsed_candidates = parsed_map.get(ticket.key, [])
                    if not parsed_candidates:
                        logger.warning(
                            "No parsed candidates for %s; recording failure",
                            ticket.group_id,
                        )
                        failure_entry = {
                            "ticket_key": ticket.key,
                            "group_id": ticket.group_id,
                            "mission": ticket.mission,
                            "gt_label": ticket.label,
                            "reason": "no_candidates",
                            "raw_text": None,
                        }
                        _append_jsonl(failure_path, failure_entry)
                        record_outcome(
                            ticket,
                            model_verdict=None,
                            in_manual_review=True,
                        )
                        _record_group_delta(
                            ticket,
                            guidance_step=guidance.step,
                            model_verdict=None,
                            in_manual_review=True,
                            manual_reason="no_candidates",
                        )
                        _mark_group_step(ticket.key)
                        continue

                    wrapped_candidates: List[TrajectoryWithSignals] = []
                    saw_format_error = False
                    for cand in parsed_candidates:
                        reason_text = cand.reason or ""
                        verdict_val = cand.verdict
                        format_ok = (
                            cand.format_ok
                            and verdict_val is not None
                            and bool(reason_text.strip())
                        )
                        if not format_ok:
                            failure_entry = {
                                "ticket_key": ticket.key,
                                "group_id": ticket.group_id,
                                "mission": ticket.mission,
                                "gt_label": ticket.label,
                                "reason": "format_error",
                                "raw_text": cand.base.response_text,
                            }
                            _append_jsonl(failure_path, failure_entry)
                            record_outcome(
                                ticket,
                                model_verdict=None,
                                in_manual_review=True,
                            )
                            _record_group_delta(
                                ticket,
                                guidance_step=guidance.step,
                                model_verdict=None,
                                in_manual_review=True,
                                manual_reason="format_error",
                            )
                            saw_format_error = True
                            break

                        label_match = (
                            cand.verdict == ticket.label
                            if cand.verdict is not None
                            else False
                        )
                        signals = DeterministicSignals(
                            label_match=label_match,
                            self_consistency=None,
                            conflict_flag=label_match is False,
                            needs_manual_review=False,
                        )
                        wrapped_candidates.append(
                            TrajectoryWithSignals(parsed=cand, signals=signals)
                        )

                    if saw_format_error:
                        _mark_group_step(ticket.key)
                        continue

                    if not wrapped_candidates:
                        # nothing usable for this group
                        failure_entry = {
                            "ticket_key": ticket.key,
                            "group_id": ticket.group_id,
                            "mission": ticket.mission,
                            "gt_label": ticket.label,
                            "reason": "no_valid_candidates",
                            "raw_text": None,
                        }
                        _append_jsonl(failure_path, failure_entry)
                        record_outcome(
                            ticket,
                            model_verdict=None,
                            in_manual_review=True,
                        )
                        _record_group_delta(
                            ticket,
                            guidance_step=guidance.step,
                            model_verdict=None,
                            in_manual_review=True,
                            manual_reason="no_valid_candidates",
                        )
                        _mark_group_step(ticket.key)
                        continue

                    # Select final verdict
                    from src.stage_b.scoring.selection import select_for_group

                    try:
                        selection = select_for_group(
                            ticket,
                            wrapped_candidates,
                            mission_g0=guidance.experiences.get("G0"),
                            guidance_step=guidance.step,
                            reflection_cycle=reflection_cycle,
                            reflection_change=last_reflection_id,
                            config=config.selection,
                            manual_review=config.manual_review,
                        )
                    except Exception as exc:  # noqa: BLE001
                        failure_entry = {
                            "ticket_key": ticket.key,
                            "group_id": ticket.group_id,
                            "mission": ticket.mission,
                            "gt_label": ticket.label,
                            "reason": f"selection_error: {exc}",
                        }
                        _append_jsonl(failure_path, failure_entry)
                        record_outcome(
                            ticket,
                            model_verdict=None,
                            in_manual_review=True,
                        )
                        _record_group_delta(
                            ticket,
                            guidance_step=guidance.step,
                            model_verdict=None,
                            in_manual_review=True,
                            manual_reason="selection_error",
                        )
                        _mark_group_step(ticket.key)
                        continue

                    low_agreement_flag = (
                        selection.vote_strength is not None
                        and selection.vote_strength
                        < config.manual_review.min_verdict_agreement
                    )
                    updated_candidates: List[TrajectoryWithSignals] = []
                    for candidate in wrapped_candidates:
                        sig = candidate.signals
                        if sig is None:
                            updated_candidates.append(candidate)
                            continue
                        updated_sig = replace(
                            sig,
                            vote_strength=selection.vote_strength,
                            low_agreement=low_agreement_flag,
                            needs_manual_review=(
                                sig.needs_manual_review or selection.needs_manual_review
                            ),
                        )
                        updated_candidates.append(
                            replace(candidate, signals=updated_sig)
                        )
                    wrapped_candidates = updated_candidates

                    # Write trajectories
                    for candidate in wrapped_candidates:
                        trajectory_payload = serialize_trajectory(
                            candidate,
                            reflection_cycle=reflection_cycle,
                            guidance_step=guidance.step,
                        )
                        trajectory_payload["epoch"] = epoch
                        trajectory_payload["ticket_key"] = ticket.key
                        trajectory_payload["gt_label"] = ticket.label
                        _append_jsonl(trajectories_path, trajectory_payload)

                    selection_payload = serialize_selection(selection)
                    selection_payload["epoch"] = epoch
                    selection_payload["epoch_step"] = epoch_step
                    selection_payload["global_step"] = global_step
                    selection_payload["ticket_key"] = ticket.key
                    selection_payload["gt_label"] = ticket.label
                    _append_jsonl(selections_path, selection_payload)
                    mission_selection_count += 1

                    record_outcome(
                        ticket,
                        model_verdict=selection.verdict,
                        in_manual_review=False,
                    )

                    candidates_payload = []
                    for candidate in wrapped_candidates:
                        parsed = candidate.parsed
                        sig = candidate.signals
                        if parsed is None:
                            continue
                        decode = parsed.base.decode
                        candidates_payload.append(
                            {
                                "candidate_index": parsed.base.candidate_index,
                                "verdict": parsed.verdict,
                                "reason": parsed.reason,
                                "decode": {
                                    "temperature": decode.temperature,
                                    "top_p": decode.top_p,
                                    "max_new_tokens": decode.max_new_tokens,
                                    "seed": decode.seed,
                                    "stop": list(decode.stop),
                                },
                                "format_ok": parsed.format_ok,
                                "label_match": getattr(sig, "label_match", None)
                                if sig is not None
                                else None,
                            }
                        )
                    _record_group_delta(
                        ticket,
                        guidance_step=guidance.step,
                        model_verdict=selection.verdict,
                        in_manual_review=False,
                        manual_reason=None,
                        selection_payload=selection_payload,
                        candidates_payload=candidates_payload,
                    )

                    # Buffer selected candidate for potential distillation logging
                    if distill_enabled:
                        distill_records_epoch.append(
                            {
                                "ticket": ticket,
                                "verdict": selection.verdict,
                                "reason": selection.reason,
                            }
                        )

                    candidate_verdicts = [
                        c.parsed.verdict
                        for c in wrapped_candidates
                        if c.parsed is not None
                    ]
                    grad_candidate = _is_gradient_candidate(
                        label_match=selection.label_match,
                        low_agreement=low_agreement_flag,
                        conflict_flag=bool(selection.conflict_flag),
                        needs_manual_review=bool(selection.needs_manual_review),
                        candidate_verdicts=candidate_verdicts,
                    )

                    if grad_candidate:
                        record = reflection_engine.build_record(
                            ticket,
                            wrapped_candidates,
                            selection.selected_candidate,
                            guidance.step,
                            epoch_step=epoch_step,
                            global_step=global_step,
                        )
                        pending_records.append(record)
                        if last_applied_rule_keys:
                            pending_feedback[ticket.key] = _PendingRuleFeedback(
                                experience_keys=tuple(last_applied_rule_keys),
                                label_match=bool(selection.label_match),
                            )
                        if len(pending_records) >= config.reflection.batch_size:
                            _flush_gradient_candidates(
                                epoch, "batch_full", flush_partial=False
                            )
                    elif last_applied_rule_keys:
                        try:
                            if bool(selection.label_match):
                                mission_guidance_repo.increment_hit_count(
                                    mission, list(last_applied_rule_keys)
                                )
                            else:
                                mission_guidance_repo.increment_miss_count(
                                    mission, list(last_applied_rule_keys)
                                )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "Failed to update rule hit/miss feedback for %s: %s",
                                mission,
                                exc,
                            )

                    _mark_group_step(ticket.key)

            if is_main_process() and (pending_records or any(retry_queues.values())):
                _flush_gradient_candidates(epoch, "epoch_end", flush_partial=True)

            if is_main_process():
                if window_ticket_keys:
                    _flush_metrics_window(event="epoch_end_partial")
                records_this_epoch = list(epoch_outcomes.values())
                clean_records = [
                    record
                    for record in records_this_epoch
                    if not record.get("in_manual_review")
                ]
                metrics_exclude_mr = _compute_metrics(clean_records)
                metrics_include_mr = _compute_metrics(records_this_epoch)

                logger.info(
                    "Epoch %d summary | exc: acc=%.4f fn=%d fp=%d fn_rate=%.4f fp_rate=%.4f n=%d | inc: acc=%.4f fn=%d fp=%d fn_rate=%.4f fp_rate=%.4f n=%d",
                    epoch,
                    metrics_exclude_mr["acc"],
                    metrics_exclude_mr["fn"],
                    metrics_exclude_mr["fp"],
                    metrics_exclude_mr["fn_rate"],
                    metrics_exclude_mr["fp_rate"],
                    metrics_exclude_mr["n"],
                    metrics_include_mr["acc"],
                    metrics_include_mr["fn"],
                    metrics_include_mr["fp"],
                    metrics_include_mr["fn_rate"],
                    metrics_include_mr["fp_rate"],
                    metrics_include_mr["n"],
                )

                _append_jsonl(
                    metrics_path,
                    {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "epoch_step_start": 1,
                        "epoch_step_end": epoch_step,
                        "global_step": global_step,
                        "logging_steps": logging_steps,
                        "exclude_manual_review": metrics_exclude_mr,
                        "include_manual_review": metrics_include_mr,
                    },
                )

            should_continue = 1 if epoch < config.runner.epochs else 0
            should_continue = broadcast_int(
                should_continue if is_main_process() else 0, src=0
            )
            if should_continue == 0:
                break

        if is_main_process():
            # Epoch-end cleanup: remove low-confidence rules
            if (
                config.guidance_lifecycle
                and config.guidance_lifecycle.enable_auto_cleanup
            ):
                assert mission_guidance_repo is not None
                try:
                    removed = mission_guidance_repo.cleanup_low_confidence(
                        mission,
                        confidence_threshold=config.guidance_lifecycle.confidence_drop_threshold,
                        min_miss_before_drop=config.guidance_lifecycle.min_miss_before_drop,
                    )
                    if removed:
                        logger.info(
                            f"Epoch {epoch}: cleanup_low_confidence removed {len(removed)} rule(s): {removed}"
                        )
                        guidance_updated_epoch = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Failed to perform cleanup at epoch {epoch}: {exc}")

            total_selections += mission_selection_count

            logger.info(
                f"Completed mission {mission}: {mission_selection_count} selections"
            )

            try:
                need_review_summary_path = mission_dir / "need_review.json"
                _write_need_review_summary(
                    mission=mission,
                    need_review_queue_path=need_review_path,
                    output_path=need_review_summary_path,
                )
                logger.info("Wrote need-review summary to %s", need_review_summary_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to build need_review.json for %s: %s", mission, exc
                )

            if config.output.group_report:
                try:
                    report_path = build_group_report(mission_dir, config.stage_a_paths)
                    logger.info("Wrote grouped report to %s", report_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to build group report for %s: %s", mission, exc
                    )

            # If guidance did not change this epoch and distillation is enabled, emit distill chatml and stop further epochs
            if distill_enabled and not guidance_updated_epoch:
                assert mission_guidance_repo is not None
                guidance_map = mission_guidance_repo.load()
                if mission not in guidance_map:
                    raise RuntimeError(
                        f"Mission {mission} guidance missing at convergence"
                    )
                guidance = guidance_map[mission]
                distill_path.parent.mkdir(parents=True, exist_ok=True)
                with distill_path.open("w", encoding="utf-8") as fh:
                    for record in distill_records_epoch:
                        ticket: GroupTicket = record["ticket"]  # type: ignore[assignment]
                        verdict: str = record["verdict"]  # type: ignore[assignment]
                        reason: Optional[str] = record["reason"]  # type: ignore[assignment]
                        messages = build_messages(
                            ticket, guidance, domain=domain_map[mission]
                        )
                        verdict_text = "通过" if verdict == "pass" else "不通过"
                        reason_text = reason or ""
                        assistant_content = (
                            f"Verdict: {verdict_text}\nReason: {reason_text}"
                        )
                        messages.append(
                            {"role": "assistant", "content": assistant_content}
                        )
                        payload = {
                            "ticket_key": ticket.key,
                            "group_id": ticket.group_id,
                            "mission": ticket.mission,
                            "label": ticket.label,
                            "messages": messages,
                        }
                        fh.write(json.dumps(payload, ensure_ascii=False))
                        fh.write("\n")
                logger.info(
                    "Early stop: guidance unchanged in epoch %d. Wrote %d distill records to %s; halting remaining epochs for mission %s.",
                    epoch,
                    len(distill_records_epoch),
                    distill_path,
                    mission,
                )
                break

    logger.info(
        f"Completed Stage-B pipeline across {config.runner.epochs} epoch(s) and {processed_missions} mission(s): {total_selections} total selections"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-B reflection runner")
    parser.add_argument("--config", required=True, help="Path to Stage-B YAML config")
    parser.add_argument(
        "--step",
        choices=["all"],
        default="all",
        help="Pipeline step to execute",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "logging", "warning"],
        default="logging",
        help="Logging level for the pipeline (ignored if --debug is set)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (HIGHEST PRIORITY, overrides --log-level)",
    )
    args = parser.parse_args()

    stage_b_config = load_stage_b_config(args.config)
    effective_level = "debug" if args.debug else args.log_level
    if args.step == "all":
        run_all(stage_b_config, log_level=effective_level)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported step: {args.step}")


if __name__ == "__main__":
    main()
