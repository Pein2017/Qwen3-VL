#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entrypoint for the rule-search Stage-B pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import time
import uuid
from dataclasses import replace
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence, Iterator
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import torch
from transformers import AutoConfig, PreTrainedTokenizerBase

from ..utils import configure_logging, get_logger
from src.utils import require_mapping
from src.utils.unstructured import UnstructuredMapping
from .config import (
    SamplerConfig,
    StageBConfig,
    load_stage_b_config,
    resolve_domain_for_mission,
)
from ..distributed import (
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
from .io.guidance import GuidanceRepository
from src.generation import (
    ChatTemplateOptions,
    DecodeOptions,
    GenerationEngine,
    GenerationOptions,
    ModelLoadConfig,
    QWEN_STOP_TOKENS,
    StopOptions,
    TextGenerationRequest,
    VlmPreprocessOptions,
    build_hf_engine,
)
from src.generation.chat_template import render_chat_template
from .reflection import ReflectionEngine
from .rollout import RolloutSampler
from .rule_search import (
    EvalMetrics,
    GateStats,
    TicketRolloutStats,
    build_gate_stats,
    build_ticket_stats,
    compute_metrics as compute_rule_search_metrics,
    normalize_rule_signature,
    pick_reflection_ticket_keys,
)
from .sampling.prompts import build_messages
from .types import (
    ExperienceOperation,
    ExperienceOperationKind,
    GroupTicket,
    MissionGuidance,
    ReflectionAction,
    ReflectionProposal,
)
from .utils.chinese import normalize_spaces, to_simplified
from .utils.perf import enable_tf32
from .utils.seed import seed_everything

logger = get_logger("stage_b.runner")


class RuleCandidatePayload(TypedDict):
    op: str
    text: str | None
    rationale: str | None
    signature: str | None
    target_signature: str | None
    target_signatures: list[str]
    candidate_id: str
    source: str
    confidence: NotRequired[float]
    operation: NotRequired[ExperienceOperation]
    gate: NotRequired[GateStats]
    cand_metrics: NotRequired[EvalMetrics]
    eval_metrics: NotRequired[EvalMetrics | None]
    eval_acc_drop: NotRequired[float | None]
    relative_error_reduction: NotRequired[float]


class ProposerFailurePayload(TypedDict, total=False):
    timestamp: float
    iteration: int
    mission: str
    examples: int
    prompt_tokens: int | None
    error: str
    raw_response_prefix: NotRequired[str]
    raw_response_suffix: NotRequired[str]


class ReasonCountRow(TypedDict):
    reason: str
    count: int


class HardCaseRow(TypedDict, total=False):
    timestamp: float
    iteration: int
    mission: str
    sampler: str
    ticket_key: str | None
    gt_label: object
    majority_pred: object
    majority_reason: str | None
    pass_count: object
    fail_count: object
    invalid_count: object
    total_samples: object
    agreement: object
    difficulty: object
    hard_wrong: object
    verdict_samples: list[object]
    reason_samples: list[str | None]
    reason_counts: list[ReasonCountRow]
    rank: NotRequired[int]


class CandidateRegressionRow(TypedDict, total=False):
    ticket_key: object
    gt_label: object
    base_pred: object
    cand_pred: object
    base_agreement: object
    cand_agreement: object
    base_pass_count: object
    base_fail_count: object
    cand_pass_count: object
    cand_fail_count: object


def _load_excluded_ticket_keys(config: StageBConfig) -> set[str]:
    ticket_filter = getattr(config, "ticket_filter", None)
    if ticket_filter is None:
        return set()

    excluded: set[str] = set()
    excluded.update(getattr(ticket_filter, "exclude_ticket_keys", ()) or ())

    path = getattr(ticket_filter, "exclude_ticket_keys_path", None)
    if path is None:
        return {key for key in excluded if key}

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"ticket_filter.exclude_ticket_keys_path not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            excluded.add(stripped)
    return {key for key in excluded if key}


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


def _safe_model_max_length(tokenizer: PreTrainedTokenizerBase) -> int | None:
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


def _load_generation_engine(config: StageBConfig) -> GenerationEngine:
    variant = _detect_model_variant(config.model.model_name_or_path)
    logger.info(
        "Detected model variant '%s' for %s",
        variant,
        config.model.model_name_or_path,
    )
    _dtype_from_str(config.model.torch_dtype)
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
    attn_impl = config.model.attn_implementation
    if attn_impl is not None and attn_impl.lower() in ("flash_attn", "flash_attention_2"):
        attn_impl = "flash_attention_2"
    model_config = ModelLoadConfig(
        model_name_or_path=config.model.model_name_or_path,
        torch_dtype=config.model.torch_dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        variant="vlm" if variant == "vl" else "text",
    )
    chat_template = ChatTemplateOptions(
        add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    preprocess = VlmPreprocessOptions()
    engine = build_hf_engine(
        model_config, chat_template=chat_template, preprocess=preprocess
    )
    return engine


def _require_engine_tokenizer(engine: GenerationEngine) -> PreTrainedTokenizerBase:
    tokenizer = engine.tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise RuntimeError("Engine tokenizer is unavailable or invalid")
    return tokenizer


def _shuffle_indices(count: int, *, epoch: int, base_seed: int) -> list[int]:
    indices = list(range(count))
    seed_value = base_seed + epoch
    random.Random(seed_value).shuffle(indices)
    return indices


def _resolve_train_pool_size(
    *,
    total: int,
    train_pool_size: int,
    train_pool_fraction: float | None,
) -> int:
    if total <= 0:
        return 0
    if train_pool_fraction is not None:
        target = int(round(total * float(train_pool_fraction)))
    else:
        target = int(train_pool_size)
    return min(max(1, target), total)


def _compute_learnability_coverage(
    learnable_keys: set[str],
    *candidate_keys: Sequence[str],
) -> tuple[set[str], set[str]]:
    contributors: set[str] = set()
    for keys in candidate_keys:
        contributors.update(keys)
    uncovered = set(learnable_keys) - contributors
    return contributors, uncovered


def _split_train_eval_pool(
    tickets: Sequence[GroupTicket],
    *,
    eval_size: int,
    seed: int,
    stratify_by_label: bool,
) -> tuple[list[GroupTicket], list[GroupTicket]]:
    if not tickets:
        return [], []
    if eval_size <= 0:
        return list(tickets), []
    if eval_size >= len(tickets):
        return [], list(tickets)

    rng = random.Random(int(seed))

    if stratify_by_label:
        by_label: dict[str, list[GroupTicket]] = defaultdict(list)
        for ticket in tickets:
            by_label[str(ticket.label)].append(ticket)

        total = len(tickets)
        eval_target = min(max(1, int(eval_size)), total)

        desired_counts: dict[str, int] = {}
        remainders: list[tuple[str, float]] = []
        for label in sorted(by_label.keys()):
            bucket_len = len(by_label[label])
            raw = eval_target * bucket_len / total
            count = int(raw)
            desired_counts[label] = count
            remainders.append((label, raw - count))

        allocated = sum(desired_counts.values())
        remaining = eval_target - allocated
        if remaining > 0:
            remainders.sort(key=lambda item: (-item[1], item[0]))
            for label, _ in remainders:
                if remaining <= 0:
                    break
                desired_counts[label] += 1
                remaining -= 1

        train: list[GroupTicket] = []
        holdout: list[GroupTicket] = []
        for label in sorted(by_label.keys()):
            bucket = sorted(by_label[label], key=lambda t: t.key)
            indices = list(range(len(bucket)))
            rng.shuffle(indices)
            cutoff = min(desired_counts.get(label, 0), len(bucket))
            holdout.extend([bucket[i] for i in indices[:cutoff]])
            train.extend([bucket[i] for i in indices[cutoff:]])
        return train, holdout

    ordered = sorted(tickets, key=lambda t: t.key)
    indices = list(range(len(ordered)))
    rng.shuffle(indices)
    cutoff = min(max(1, int(eval_size)), len(ordered))
    holdout = [ordered[i] for i in indices[:cutoff]]
    train = [ordered[i] for i in indices[cutoff:]]
    return train, holdout


def _sample_train_pool_tickets(
    tickets: Sequence[GroupTicket],
    *,
    pool_size: int,
    with_replacement: bool,
    seed: int,
) -> list[GroupTicket]:
    if not tickets or pool_size <= 0:
        return []

    total = len(tickets)
    target = min(max(1, int(pool_size)), total)
    rng = random.Random(int(seed))

    if with_replacement:
        return [rng.choice(tickets) for _ in range(target)]

    indices = list(range(total))
    rng.shuffle(indices)
    return [tickets[i] for i in indices[:target]]


def _extract_verdict_samples(
    trajectories: Sequence[object],
) -> list[str | None]:
    verdicts: list[str | None] = []
    for cand in trajectories:
        verdict_val = getattr(cand, "verdict", None)
        format_ok = bool(getattr(cand, "format_ok", False))
        if not format_ok:
            verdicts.append(None)
        else:
            verdicts.append(str(verdict_val) if verdict_val is not None else None)
    return verdicts


def _extract_reason_samples(
    trajectories: Sequence[object],
) -> list[str | None]:
    reasons: list[str | None] = []
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
    progress_logging_steps: int | None = None,
    progress_label: str | None = None,
    progress_gt_by_ticket_key: Mapping[str, str] | None = None,
    progress_jsonl_path: Path | None = None,
) -> tuple[dict[str, list[object]], tuple[str, ...]]:
    """Run distributed rollout for a single mission and return trajectories + dropped keys.

    Returns payloads only on rank 0; other ranks return empty payloads and drops.
    """

    world_size = get_world_size()
    rank = get_rank()
    global_batch_size = max(1, int(per_rank_batch_size)) * max(1, int(world_size))

    guidance_map = {mission: guidance}
    domain_map = {mission: domain}

    merged: dict[str, list[object]] = {}
    dropped: set[str] = set()
    batch_index = 0

    log_every = int(progress_logging_steps or 0)
    should_log_progress = bool(is_main_process() and log_every > 0)
    gt_by_key: dict[str, str] | None = None
    stats_by_key: dict[str, TicketRolloutStats] | None = None
    next_log_at = log_every
    total_batches = 0
    start_time = time.time()
    if should_log_progress:
        total_batches = int((len(tickets) + global_batch_size - 1) / global_batch_size)
        gt_by_key = (
            dict(progress_gt_by_ticket_key)
            if progress_gt_by_ticket_key is not None
            else {ticket.key: ticket.label for ticket in tickets}
        )
        stats_by_key = {}
        logger.info(
            "%s rollout progress enabled: tickets=%d log_every=%d global_batch=%d budget=%s",
            progress_label or "rollout",
            len(tickets),
            log_every,
            global_batch_size,
            sampler.config.max_prompt_tokens
            if sampler.config.max_prompt_tokens is not None
            else "default",
        )
    for batch in _chunked(list(tickets), global_batch_size):
        batch_index += 1
        start, end = _shard_bounds(len(batch), world_size=world_size, rank=rank)
        shard = batch[start:end]
        local_error: str | None = None
        try:
            local_payloads, local_dropped = sampler.generate_for_batch(
                shard, guidance_map, domain_map
            )
            local = {"payloads": local_payloads, "dropped": list(local_dropped)}
        except Exception as exc:  # noqa: BLE001
            local_error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "rollout batch failed (rank=%d batch=%d/%d size=%d)",
                rank,
                batch_index,
                int((len(tickets) + global_batch_size - 1) / global_batch_size),
                len(shard),
            )
            local = {"__error__": local_error, "payloads": {}, "dropped": []}
        gathered = gather_object(local, dst=0)
        if not is_main_process():
            error_flag = broadcast_int(0, src=0)
            if error_flag:
                raise RuntimeError("rollout aborted due to upstream error")
            continue
        assert gathered is not None
        error_flag = 0
        updated_keys: set[str] | None = set() if should_log_progress else None
        for part in gathered:
            if "__error__" in part:
                error_flag = 1
                logger.error("rollout failed on a rank: %s", part.get("__error__"))
                continue
            part_payloads = part.get("payloads", {})
            part_dropped = part.get("dropped", [])
            if isinstance(part_dropped, Sequence) and not isinstance(
                part_dropped, (str, bytes)
            ):
                for item in part_dropped:
                    dropped.add(str(item))
            if isinstance(part_payloads, Mapping):
                for ticket_key, trajectories in part_payloads.items():
                    merged.setdefault(ticket_key, []).extend(trajectories)
                    if updated_keys is not None:
                        updated_keys.add(str(ticket_key))

        error_flag = broadcast_int(error_flag, src=0)
        if error_flag:
            raise RuntimeError("rollout aborted due to upstream error")

        if should_log_progress and updated_keys and gt_by_key is not None and stats_by_key is not None:
            for ticket_key in updated_keys:
                gt_label = gt_by_key.get(ticket_key)
                if not gt_label:
                    continue
                verdicts = _extract_verdict_samples(merged.get(ticket_key, []))
                stats_by_key[ticket_key] = build_ticket_stats(
                    ticket_key=ticket_key,
                    gt_label=gt_label,
                    verdicts=verdicts,
                )

            while len(stats_by_key) >= next_log_at and next_log_at > 0:
                metrics = compute_rule_search_metrics(list(stats_by_key.values()))
                elapsed = time.time() - start_time
                logger.info(
                    "%s rollout progress: batch=%d/%d used=%d/%d dropped=%d acc=%.4f fn_rate=%.4f fp_rate=%.4f elapsed=%.1fs",
                    progress_label or "rollout",
                    batch_index,
                    total_batches,
                    metrics.n,
                    len(tickets),
                    len(dropped),
                    metrics.acc,
                    metrics.fn_rate,
                    metrics.fp_rate,
                    elapsed,
                )
                if progress_jsonl_path is not None:
                    progress_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                    with progress_jsonl_path.open("a", encoding="utf-8") as fh:
                        fh.write(
                            json.dumps(
                                {
                                    "timestamp": time.time(),
                                    "label": progress_label or "rollout",
                                    "batch_index": batch_index,
                                    "total_batches": total_batches,
                                    "n_total": len(tickets),
                                    "n_used": metrics.n,
                                    "n_dropped_overlength": len(dropped),
                                    "acc": metrics.acc,
                                    "fn": metrics.fn,
                                    "fp": metrics.fp,
                                    "tp": metrics.tp,
                                    "fn_rate": metrics.fn_rate,
                                    "fp_rate": metrics.fp_rate,
                                    "elapsed_sec": elapsed,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                next_log_at += log_every

    if is_main_process() and dropped:
        budget = sampler.config.max_prompt_tokens
        logger.info(
            "rollout dropped overlength prompts: dropped=%d kept=%d budget=%s",
            len(dropped),
            len(merged),
            budget if budget is not None else "default",
        )

    return (merged, tuple(sorted(dropped))) if is_main_process() else ({}, ())


def _distributed_rollout_verdicts(
    *,
    tickets: Sequence[GroupTicket],
    sampler: RolloutSampler,
    guidance: MissionGuidance,
    mission: str,
    domain: str,
    per_rank_batch_size: int,
) -> tuple[dict[str, list[str | None]], tuple[str, ...]]:
    """Run distributed rollout and return verdict samples plus dropped keys.

    Returns payloads only on rank 0; other ranks return empty payloads and drops.
    """

    world_size = get_world_size()
    rank = get_rank()
    global_batch_size = max(1, int(per_rank_batch_size)) * max(1, int(world_size))

    guidance_map = {mission: guidance}
    domain_map = {mission: domain}

    merged: dict[str, list[str | None]] = {}
    dropped: set[str] = set()
    for batch in _chunked(list(tickets), global_batch_size):
        start, end = _shard_bounds(len(batch), world_size=world_size, rank=rank)
        shard = batch[start:end]
        local_payloads, local_dropped = sampler.generate_for_batch(
            shard, guidance_map, domain_map
        )
        local = {"payloads": local_payloads, "dropped": list(local_dropped)}
        gathered = gather_object(local, dst=0)
        if not is_main_process():
            continue
        assert gathered is not None
        for part in gathered:
            part_payloads = part.get("payloads", {})
            part_dropped = part.get("dropped", [])
            if isinstance(part_dropped, Sequence) and not isinstance(
                part_dropped, (str, bytes)
            ):
                for item in part_dropped:
                    dropped.add(str(item))
            if isinstance(part_payloads, Mapping):
                for ticket_key, trajectories in part_payloads.items():
                    merged.setdefault(ticket_key, []).extend(
                        _extract_verdict_samples(trajectories)
                    )

    if is_main_process() and dropped:
        budget = sampler.config.max_prompt_tokens
        logger.info(
            "rollout dropped overlength prompts (verdicts): dropped=%d kept=%d budget=%s",
            len(dropped),
            len(merged),
            budget if budget is not None else "default",
        )

    return (merged, tuple(sorted(dropped))) if is_main_process() else ({}, ())


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
    modifiable_lines = [
        f"[{key}] signature={normalize_rule_signature(value)} | {value}"
        for key, value in sorted(guidance.experiences.items())
        if key.startswith("G") and key != "G0"
    ]

    lines: list[str] = [
        f"任务: {mission}",
        f"G0: {g0}",
    ]
    if scaffold_lines:
        lines.extend(["", "SCAFFOLD (S*):", *scaffold_lines])
    if modifiable_lines:
        lines.extend(
            [
                "",
                "EXISTING_RULES (modifiable G*, G0 is non-removable):",
                *modifiable_lines,
            ]
        )
    lines.append("")
    lines.append("以下为用于提出候选规则的错例样本（仅供提案，不用于最终评估）：")
    lines.append("")

    for idx, ticket in enumerate(examples, start=1):
        stats = stats_by_ticket.get(ticket.key)
        majority_pred = getattr(stats, "majority_pred", None)
        agreement = getattr(stats, "agreement", 0.0)
        difficulty = getattr(stats, "difficulty", 0.0)
        lines.append(
            f"样本{idx}: ticket_key={ticket.key}; gt_label={ticket.label}; baseline_majority={majority_pred}; agreement={agreement:.3f}; difficulty={difficulty:.3f}"
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
    engine: GenerationEngine,
    tokenizer: PreTrainedTokenizerBase,
    config: StageBConfig,
    mission: str,
    guidance: MissionGuidance,
    examples: Sequence[GroupTicket],
    stats_by_ticket: Mapping[str, object],
    reflection_engine: ReflectionEngine,
    iteration: int,
    rejected_candidate_ids: Sequence[str],
    log_dir: Path | None = None,
) -> list[RuleCandidatePayload]:
    """Run proposer LLM once and return candidate operations (validated + de-duplicated)."""

    assert config.rule_search is not None
    system_prompt = Path(config.rule_search.proposer_prompt_path).read_text(
        encoding="utf-8"
    )
    max_prompt_tokens = int(config.rule_search.proposer_max_prompt_tokens)
    example_list = list(examples)
    prompt_tokens: int | None = None

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

        chat_prompt = render_chat_template(
            tokenizer,
            messages,
            options=ChatTemplateOptions(add_generation_prompt=True, tokenize=False),
        )

        try:
            encoded_probe = tokenizer(
                chat_prompt,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            input_ids = getattr(encoded_probe, "input_ids", None)
            if input_ids is None and isinstance(encoded_probe, Mapping):
                input_ids = encoded_probe.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                prompt_tokens = int(input_ids.size(1))
            else:
                prompt_tokens = len(input_ids[0]) if input_ids else None
        except Exception:  # noqa: BLE001
            prompt_tokens = None

        if prompt_tokens is None or prompt_tokens <= max_prompt_tokens:
            break
        if len(example_list) <= 1:
            break
        example_list.pop()

    if prompt_tokens is None or prompt_tokens > max_prompt_tokens:
        logger.warning(
            "rule_search proposer prompt exceeds token budget after trimming: tokens=%s budget=%d examples=%d",
            prompt_tokens if prompt_tokens is not None else "unknown",
            max_prompt_tokens,
            len(example_list),
        )
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            failure_payload: ProposerFailurePayload = {
                "timestamp": time.time(),
                "iteration": iteration,
                "mission": mission,
                "examples": len(example_list),
                "prompt_tokens": prompt_tokens,
                "error": "prompt_overlength",
            }
            _append_jsonl(
                log_dir / "rule_search_proposer_failures.jsonl",
                failure_payload,
            )
        return []

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

    options = GenerationOptions(
        max_new_tokens=config.rule_search.proposer_max_new_tokens,
        temperature=config.rule_search.proposer_temperature,
        top_p=config.rule_search.proposer_top_p,
        repetition_penalty=config.rule_search.proposer_repetition_penalty,
        do_sample=True,
        stop=StopOptions(stop=QWEN_STOP_TOKENS),
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        seed=int(config.seed),
        decode=DecodeOptions(
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            strip_whitespace=True,
        ),
    )
    request = TextGenerationRequest(messages=messages)
    result = engine.generate_text_batch([request], options=options)[0]
    raw_response = result.raw_text
    response = normalize_spaces(to_simplified(result.text))

    try:
        payload = ReflectionEngine._loads_first_json(response)
    except Exception as exc:  # noqa: BLE001
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            failure_payload: ProposerFailurePayload = {
                "timestamp": time.time(),
                "iteration": iteration,
                "mission": mission,
                "examples": len(example_list),
                "prompt_tokens": prompt_tokens,
                "error": str(exc),
                "raw_response_prefix": raw_response[:400],
                "raw_response_suffix": raw_response[-400:],
            }
            _append_jsonl(
                log_dir / "rule_search_proposer_failures.jsonl",
                failure_payload,
            )
        raise ValueError(
            f"No valid JSON found in rule_search proposer response (examples={len(example_list)}, prompt_tokens={prompt_tokens})"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Rule proposer must return a JSON object")

    operations_raw = payload.get("operations")
    rules_raw = payload.get("rules")
    if operations_raw is not None and rules_raw is not None:
        raise ValueError("rule_search proposer must not return both operations and rules")
    if operations_raw is None:
        if not isinstance(rules_raw, Sequence) or isinstance(rules_raw, (str, bytes)):
            raise ValueError("rules must be a list")
        operations_raw = rules_raw
    if not isinstance(operations_raw, Sequence) or isinstance(
        operations_raw, (str, bytes)
    ):
        raise ValueError("operations must be a list")

    scaffold_texts = [
        text for key, text in guidance.experiences.items() if key.startswith("S")
    ]
    existing_norms = {
        normalize_rule_signature(v) for v in guidance.experiences.values()
    }

    candidates: list[RuleCandidatePayload] = []
    seen_signatures: set[str] = set()
    seen_candidate_ids: set[str] = set()
    rejected_set = {str(item).strip() for item in rejected_candidate_ids if item}

    for entry in operations_raw:
        if not isinstance(entry, Mapping):
            continue

        op_raw = entry.get("op")
        op = str(op_raw).strip().lower() if op_raw is not None else "upsert"
        if op not in {"upsert", "update", "merge", "remove"}:
            continue

        target_signature = None
        target_signatures: list[str] = []
        if op in {"update", "remove"}:
            target_raw = entry.get("target_signature")
            target_signature = (
                normalize_rule_signature(str(target_raw).strip())
                if target_raw is not None
                else ""
            )
            if not target_signature:
                continue
        if op == "merge":
            targets_raw = entry.get("target_signatures")
            if not isinstance(targets_raw, Sequence) or isinstance(
                targets_raw, (str, bytes)
            ):
                continue
            target_signatures = [
                normalize_rule_signature(str(item).strip())
                for item in targets_raw
                if item is not None
            ]
            target_signatures = [t for t in target_signatures if t]
            if len(target_signatures) < 2:
                continue

        text = ""
        signature = ""
        if op in {"upsert", "update", "merge"}:
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

        if op == "upsert":
            candidate_id = f"upsert:{signature}"
        elif op == "update":
            candidate_id = f"update:{target_signature}:{signature}"
        elif op == "remove":
            candidate_id = f"remove:{target_signature}"
        else:
            candidate_id = f"merge:{','.join(target_signatures)}:{signature}"

        if candidate_id in rejected_set:
            continue
        if candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(candidate_id)

        rationale_raw = entry.get("rationale")
        rationale = str(rationale_raw).strip() if rationale_raw is not None else ""

        candidate: RuleCandidatePayload = {
            "op": op,
            "text": text or None,
            "rationale": rationale or None,
            "signature": signature or None,
            "target_signature": target_signature,
            "target_signatures": target_signatures,
            "candidate_id": candidate_id,
            "source": "proposer",
        }
        candidates.append(candidate)
        if len(candidates) >= int(config.rule_search.num_candidate_rules):
            break

    return candidates


def _load_rejected_candidate_ids(rule_candidates_path: Path) -> set[str]:
    rejected: set[str] = set()
    for row in _load_jsonl(rule_candidates_path):
        decision = str(row.get("decision") or "").strip().lower()
        candidate_id = str(row.get("candidate_id") or row.get("signature") or "").strip()
        if candidate_id and decision in {"rejected", "rejected_eval_veto"}:
            rejected.add(candidate_id)
            if ":" not in candidate_id:
                rejected.add(f"upsert:{candidate_id}")
    return rejected


def _false_release_rate(metrics: EvalMetrics | None) -> float | None:
    if metrics is None:
        return None
    denom = metrics.tp + metrics.fp
    return metrics.fp / denom if denom else 0.0


def _false_block_rate(metrics: EvalMetrics | None) -> float | None:
    if metrics is None:
        return None
    denom = metrics.tp + metrics.fn
    return metrics.fn / denom if denom else 0.0


def _safe_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def _build_ablation_candidates(
    *,
    guidance: MissionGuidance,
    max_candidates: int,
    rejected_candidate_ids: Sequence[str],
) -> list[RuleCandidatePayload]:
    if max_candidates <= 0:
        return []
    rejected_set = {str(item).strip() for item in rejected_candidate_ids if item}

    scored: list[tuple[float, str, str]] = []
    for key, text in guidance.experiences.items():
        if not key.startswith("G") or key == "G0":
            continue
        signature = normalize_rule_signature(text)
        if not signature:
            continue
        candidate_id = f"remove:{signature}"
        if candidate_id in rejected_set:
            continue
        meta = guidance.metadata.get(key)
        confidence = float(meta.confidence) if meta is not None else 1.0
        scored.append((confidence, key, signature))

    scored.sort(key=lambda item: (item[0], item[1]))
    candidates: list[RuleCandidatePayload] = []
    for confidence, _key, signature in scored[:max_candidates]:
        candidate: RuleCandidatePayload = {
            "op": "remove",
            "text": None,
            "rationale": "ablation_candidate",
            "signature": None,
            "target_signature": signature,
            "target_signatures": [],
            "candidate_id": f"remove:{signature}",
            "source": "ablation",
            "confidence": confidence,
        }
        candidates.append(candidate)
    return candidates


def _run_rule_search_mission(
    *,
    config: StageBConfig,
    engine: GenerationEngine,
    tokenizer: PreTrainedTokenizerBase,
    mission: str,
    mission_tickets: Sequence[GroupTicket],
    mission_dir: Path,
    domain: str,
    jump_reflection: bool = False,
) -> None:
    assert config.rule_search is not None

    rule_candidates_path = mission_dir / "rule_candidates.jsonl"
    benchmarks_path = mission_dir / "benchmarks.jsonl"
    hard_cases_path = mission_dir / "rule_search_hard_cases.jsonl"
    regressions_path = mission_dir / "rule_search_candidate_regressions.jsonl"
    baseline_metrics_path = mission_dir / "baseline_metrics.json"
    baseline_ticket_stats_path = mission_dir / "baseline_ticket_stats.jsonl"
    baseline_wrong_cases_path = mission_dir / "baseline_wrong_cases.jsonl"
    baseline_metrics_steps_path = mission_dir / "baseline_metrics_steps.jsonl"
    baseline_np_cases_path = mission_dir / "baseline_np_cases.jsonl"
    baseline_ng_cases_path = mission_dir / "baseline_ng_cases.jsonl"

    mission_guidance_repo: GuidanceRepository | None = None
    reflection_engine: ReflectionEngine | None = None
    if is_main_process():
        mission_guidance_repo = _setup_mission_guidance(
            startup_path=config.guidance.path,
            mission_dir=mission_dir,
            mission=mission,
            retention=config.guidance.retention,
            reset=config.guidance.reset_on_rerun,
        )
        if not jump_reflection:
            reflection_engine = ReflectionEngine(
                engine=engine,
                tokenizer=tokenizer,
                config=config.reflection,
                guidance_repo=mission_guidance_repo,
                reflection_log=None,
            )

    rejected_candidate_ids_raw = (
        _load_rejected_candidate_ids(rule_candidates_path)
        if is_main_process()
        else set()
    )
    rejected_candidate_ids_raw = cast(
        object,
        broadcast_object(rejected_candidate_ids_raw if is_main_process() else None, src=0),
    )
    if isinstance(rejected_candidate_ids_raw, set):
        rejected_candidate_ids = [str(item) for item in rejected_candidate_ids_raw]
    elif isinstance(rejected_candidate_ids_raw, Sequence):
        rejected_candidate_ids = [str(item) for item in rejected_candidate_ids_raw]
    else:
        rejected_candidate_ids = []
    rejected_ids = sorted(rejected_candidate_ids)

    # Shuffle once to avoid label-ordered tickets (deterministic per seed).
    mission_tickets = list(mission_tickets)
    random.Random(int(config.seed)).shuffle(mission_tickets)

    # Deterministic mission ticket lookup.
    ticket_by_key = {ticket.key: ticket for ticket in mission_tickets}

    # ------------------------------------------------------------------
    # jump_reflection mode: baseline-only audit (no proposer, no gating).
    # ------------------------------------------------------------------
    if jump_reflection:
        if config.rule_search.train_sampler is None:
            raise ValueError("rule_search.train_sampler is required in rule_search mode")

        train_sampler = RolloutSampler(
            engine=engine, config=config.rule_search.train_sampler
        )

        distill_cfg = config.stage_b_distillation
        distill_requested = bool(distill_cfg and distill_cfg.enabled and distill_cfg.distill_size)

        current_guidance = None
        if is_main_process():
            assert mission_guidance_repo is not None
            current_guidance = mission_guidance_repo.get(mission)
        current_guidance = broadcast_object(current_guidance, src=0)

        tickets_filtered = None
        if is_main_process():
            tickets_filtered, _ = _filter_overlength_tickets(
                mission_tickets,
                sampler=train_sampler,
                guidance=current_guidance,
                domain=domain,
                label="jump_reflection baseline",
            )
        tickets_filtered = cast(
            list[GroupTicket] | None,
            broadcast_object(tickets_filtered if is_main_process() else None, src=0),
        )
        if tickets_filtered is None:
            tickets_filtered = []

        if is_main_process() and baseline_metrics_steps_path.exists():
            baseline_metrics_steps_path.unlink()

        base_payloads, dropped = _distributed_rollout_payloads(
            tickets=tickets_filtered,
            sampler=train_sampler,
            guidance=current_guidance,
            mission=mission,
            domain=domain,
            per_rank_batch_size=config.runner.per_rank_rollout_batch_size,
            progress_logging_steps=config.runner.logging_steps,
            progress_label="jump_reflection baseline",
            progress_gt_by_ticket_key={ticket.key: ticket.label for ticket in tickets_filtered}
            if is_main_process()
            else None,
            progress_jsonl_path=baseline_metrics_steps_path,
        )

        if is_main_process():
            if dropped:
                logger.info(
                    "jump_reflection baseline dropped tickets: dropped=%d kept=%d",
                    len(dropped),
                    len(base_payloads),
                )

            stats_by_ticket: dict[str, TicketRolloutStats] = {}
            reason_samples_by_ticket: dict[str, list[str | None]] = {}
            for ticket in tickets_filtered:
                if ticket.key not in base_payloads:
                    continue
                trajectories = base_payloads.get(ticket.key, [])
                verdicts = _extract_verdict_samples(trajectories)
                reasons = _extract_reason_samples(trajectories)
                stats_by_ticket[ticket.key] = build_ticket_stats(
                    ticket_key=ticket.key,
                    gt_label=ticket.label,
                    verdicts=verdicts,
                )
                reason_samples_by_ticket[ticket.key] = reasons

            metrics = compute_rule_search_metrics(list(stats_by_ticket.values()))

            # Write a compact baseline metrics blob for offline analysis.
            payload = {
                "timestamp": time.time(),
                "mission": mission,
                "mode": "jump_reflection",
                "n_total": len(mission_tickets),
                "n_used": len(stats_by_ticket),
                "n_dropped_overlength": len(dropped),
                "acc": metrics.acc,
                "fn": metrics.fn,
                "fp": metrics.fp,
                "tp": metrics.tp,
                "fn_rate": metrics.fn_rate,
                "fp_rate": metrics.fp_rate,
                "np": metrics.fn,  # NP: GT pass but predicted fail (false block)
                "ng": metrics.fp,  # NG: GT fail but predicted pass (false release)
            }
            baseline_metrics_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # Write per-ticket stats (rank 0 only).
            with baseline_ticket_stats_path.open("w", encoding="utf-8") as fh:
                for key, entry in sorted(
                    stats_by_ticket.items(), key=lambda item: item[0]
                ):
                    ticket = ticket_by_key.get(key)
                    fh.write(
                        json.dumps(
                            {
                                "ticket_key": entry.ticket_key,
                                "gt_label": entry.gt_label,
                                "majority_pred": entry.majority_pred,
                                "agreement": entry.agreement,
                                "hard_wrong": entry.hard_wrong,
                                "pass_count": entry.pass_count,
                                "fail_count": entry.fail_count,
                                "invalid_count": entry.invalid_count,
                                "total_samples": entry.total_samples,
                                "n_images": (
                                    len(ticket.summaries.as_dict()) if ticket else None
                                ),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            # Write joined wrong cases for fast manual/agent diagnosis.
            with baseline_wrong_cases_path.open("w", encoding="utf-8") as fh:
                for key, entry in sorted(
                    stats_by_ticket.items(), key=lambda item: item[0]
                ):
                    if entry.majority_pred is None or entry.majority_pred == entry.gt_label:
                        continue
                    ticket = ticket_by_key.get(key)
                    fh.write(
                        json.dumps(
                            {
                                "ticket_key": entry.ticket_key,
                                "group_id": key.split("::")[0],
                                "mission": mission,
                                "gt_label": entry.gt_label,
                                "majority_pred": entry.majority_pred,
                                "agreement": entry.agreement,
                                "hard_wrong": entry.hard_wrong,
                                "verdict_samples": list(entry.verdict_samples),
                                "reason_samples": reason_samples_by_ticket.get(key, [])[:3],
                                "per_image": (
                                    ticket.summaries.as_dict() if ticket else None
                                ),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            # Write split error buckets for quick analysis:
            # - NP: GT pass but predicted fail (false block)
            # - NG: GT fail but predicted pass (false release)
            with baseline_np_cases_path.open("w", encoding="utf-8") as fh_np, baseline_ng_cases_path.open(
                "w", encoding="utf-8"
            ) as fh_ng:
                for key, entry in sorted(
                    stats_by_ticket.items(), key=lambda item: item[0]
                ):
                    if entry.majority_pred is None or entry.majority_pred == entry.gt_label:
                        continue
                    ticket = ticket_by_key.get(key)
                    row = {
                        "ticket_key": entry.ticket_key,
                        "group_id": key.split("::")[0],
                        "mission": mission,
                        "gt_label": entry.gt_label,
                        "majority_pred": entry.majority_pred,
                        "agreement": entry.agreement,
                        "hard_wrong": entry.hard_wrong,
                        "verdict_samples": list(entry.verdict_samples),
                        "reason_samples": reason_samples_by_ticket.get(key, [])[:5],
                        "per_image": (ticket.summaries.as_dict() if ticket else None),
                    }
                    if entry.gt_label == "pass" and entry.majority_pred == "fail":
                        fh_np.write(json.dumps(row, ensure_ascii=False) + "\n")
                    elif entry.gt_label == "fail" and entry.majority_pred == "pass":
                        fh_ng.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Also export hard cases for quick triage (same format as rule-search).
            for row in _hard_case_rows(
                stats_by_ticket,
                mission=mission,
                iteration=1,
                sampler="baseline",
                limit=max(200, int(config.rule_search.reflect_size) * 8),
                reason_samples_by_ticket=reason_samples_by_ticket,
            ):
                _append_jsonl(hard_cases_path, row)

            logger.info(
                "jump_reflection baseline written: metrics=%s stats=%s wrong=%s hard_cases=%s",
                baseline_metrics_path,
                baseline_ticket_stats_path,
                baseline_wrong_cases_path,
                hard_cases_path,
            )
            logger.info(
                "jump_reflection baseline progress snapshots: %s",
                baseline_metrics_steps_path,
            )
            logger.info(
                "jump_reflection split buckets written: np=%s ng=%s",
                baseline_np_cases_path,
                baseline_ng_cases_path,
            )

            if distill_requested:
                logger.info(
                    "jump_reflection distill enabled; exporting ChatML for post-training"
                )

        if distill_requested:
            distill_guidance = current_guidance
            _run_rule_search_distill(
                config=config,
                engine=engine,
                tokenizer=tokenizer,
                mission=mission,
                mission_tickets=tickets_filtered,
                mission_dir=mission_dir,
                domain=domain,
                guidance=distill_guidance,
            )

        return

    eval_size = int(config.rule_search.eval_pool_size)
    train_tickets, eval_tickets = _split_train_eval_pool(
        list(mission_tickets),
        eval_size=eval_size,
        seed=config.seed,
        stratify_by_label=True,
    )
    if not train_tickets and eval_tickets:
        logger.warning(
            "rule_search eval_pool_size >= total; falling back to train-all (eval=0) to avoid empty train pool"
        )
        train_tickets = list(eval_tickets)
        eval_tickets = []
    logger.info(
        "rule_search split: mission=%s train=%d eval=%d eval_size=%d",
        mission,
        len(train_tickets),
        len(eval_tickets),
        eval_size,
    )

    if config.rule_search.train_sampler is None:
        raise ValueError("rule_search.train_sampler is required in rule_search mode")
    if config.rule_search.eval_sampler is None:
        raise ValueError("rule_search.eval_sampler is required in rule_search mode")
    train_sampler = RolloutSampler(
        engine=engine, config=config.rule_search.train_sampler
    )
    eval_sampler = RolloutSampler(
        engine=engine, config=config.rule_search.eval_sampler
    )
    # mining_sampler is optional; if not provided, use train_sampler
    mining_sampler_cfg = config.rule_search.mining_sampler or config.rule_search.train_sampler
    mining_sampler = RolloutSampler(
        engine=engine, config=mining_sampler_cfg
    )

    patience = int(config.rule_search.early_stop.patience)
    no_gain_rounds = 0
    hard_case_limit = max(100, int(config.rule_search.reflect_size) * 4)
    early_stop_triggered = False

    train_indices = _shuffle_indices(len(train_tickets), epoch=0, base_seed=config.seed)
    train_cursor = 0
    train_wraps = 0
    # Use runner.epochs as the iteration budget for rule-search.
    for iteration in range(1, config.runner.epochs + 1):
        current_guidance = None
        if is_main_process():
            assert mission_guidance_repo is not None
            current_guidance = mission_guidance_repo.get(mission)
        current_guidance = broadcast_object(current_guidance, src=0)

        train_pool_size = _resolve_train_pool_size(
            total=len(train_tickets),
            train_pool_size=config.rule_search.train_pool_size,
            train_pool_fraction=config.rule_search.train_pool_fraction,
        )
        if config.rule_search.train_with_replacement:
            train_pool_tickets = _sample_train_pool_tickets(
                train_tickets,
                pool_size=train_pool_size,
                with_replacement=True,
                seed=int(config.seed) + iteration,
            )
        else:
            if train_pool_size <= 0:
                train_pool_tickets = []
            elif train_cursor + train_pool_size <= len(train_indices):
                window = train_indices[train_cursor : train_cursor + train_pool_size]
                train_cursor += train_pool_size
                train_pool_tickets = [train_tickets[i] for i in window]
            else:
                remainder = train_indices[train_cursor:]
                train_wraps += 1
                train_indices = _shuffle_indices(
                    len(train_tickets), epoch=train_wraps, base_seed=config.seed
                )
                train_cursor = 0
                needed = train_pool_size - len(remainder)
                window = remainder + train_indices[train_cursor : train_cursor + needed]
                train_cursor += needed
                train_pool_tickets = [train_tickets[i] for i in window]
                if is_main_process():
                    logger.info(
                        "rule_search train pool wrapped (wraps=%d, pool_size=%d)",
                        train_wraps,
                        train_pool_size,
                    )

        if is_main_process():
            train_pool_tickets, _ = _filter_overlength_tickets(
                train_pool_tickets,
                sampler=train_sampler,
                guidance=current_guidance,
                domain=domain,
                label=f"train pool iter{iteration}",
            )
        train_pool_tickets = broadcast_object(
            train_pool_tickets if is_main_process() else None, src=0
        )
        if not train_pool_tickets:
            if is_main_process():
                logger.warning(
                    "rule_search iteration %d: all train tickets exceed max_prompt_tokens; skipping",
                    iteration,
                )
            no_gain_rounds = broadcast_int(
                no_gain_rounds + 1 if is_main_process() else 0, src=0
            )
            should_stop = no_gain_rounds >= patience
            should_stop = bool(
                broadcast_int(1 if should_stop and is_main_process() else 0, src=0)
            )
            if should_stop:
                early_stop_triggered = True
                break
            continue

        # Baseline rollout for gate evaluation (paired with candidate train sampler).
        base_payloads_train, dropped_train = _distributed_rollout_payloads(
            tickets=train_pool_tickets,
            sampler=train_sampler,
            guidance=current_guidance,
            mission=mission,
            domain=domain,
            per_rank_batch_size=config.runner.per_rank_rollout_batch_size,
        )
        base_samples_eval: dict[str, list[str | None]] = {}
        base_reasons_eval: dict[str, list[str | None]] = {}
        if is_main_process():
            available_keys = set(base_payloads_train.keys())
            missing_keys = {ticket.key for ticket in train_pool_tickets} - available_keys
            if dropped_train or missing_keys:
                dropped_all = set(dropped_train) | missing_keys
                logger.info(
                    "rule_search train rollout dropped tickets: dropped=%d kept=%d",
                    len(dropped_all),
                    len(available_keys),
                )
                train_pool_tickets = [
                    ticket
                    for ticket in train_pool_tickets
                    if ticket.key in available_keys
                ]
            for ticket_key, trajectories in base_payloads_train.items():
                base_samples_eval[ticket_key] = _extract_verdict_samples(trajectories)
                base_reasons_eval[ticket_key] = _extract_reason_samples(trajectories)

        train_pool_tickets = broadcast_object(
            train_pool_tickets if is_main_process() else None, src=0
        )
        base_stats_by_ticket: dict[str, TicketRolloutStats] = {}
        stats_for_proposer: dict[str, TicketRolloutStats] = {}
        base_metrics = None
        reflect_keys: list[str] = []
        candidates: list[RuleCandidatePayload] = []

        if is_main_process():
            for ticket in train_pool_tickets:
                verdicts = base_samples_eval.get(ticket.key, [])
                base_stats_by_ticket[ticket.key] = build_ticket_stats(
                    ticket_key=ticket.key,
                    gt_label=ticket.label,
                    verdicts=verdicts,
                )
            base_metrics = compute_rule_search_metrics(list(base_stats_by_ticket.values()))
            stats_for_proposer = dict(base_stats_by_ticket)
            for row in _hard_case_rows(
                base_stats_by_ticket,
                mission=mission,
                iteration=iteration,
                sampler="train",
                limit=hard_case_limit,
                reason_samples_by_ticket=base_reasons_eval,
            ):
                _append_jsonl(hard_cases_path, row)

            # Optional mining sampler for selecting harder mismatches (does not affect gate metrics).
            if config.rule_search.mining_sampler is not None:
                base_samples_mining, dropped_mining = _distributed_rollout_verdicts(
                    tickets=train_pool_tickets,
                    sampler=mining_sampler,
                    guidance=current_guidance,
                    mission=mission,
                    domain=domain,
                    per_rank_batch_size=config.runner.per_rank_rollout_batch_size,
                )
                if dropped_mining:
                    logger.info(
                        "rule_search mining rollout dropped tickets: dropped=%d kept=%d",
                        len(dropped_mining),
                        len(base_samples_mining),
                    )
                mining_stats: dict[str, TicketRolloutStats] = {}
                for ticket in train_pool_tickets:
                    verdicts = base_samples_mining.get(ticket.key, [])
                    if not verdicts and ticket.key not in base_samples_mining:
                        continue
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
                stats_for_proposer,
                reflect_size=config.rule_search.reflect_size,
                reflect_order=config.rule_search.reflect_order,
            )

            examples = [
                ticket_by_key[key] for key in reflect_keys if key in ticket_by_key
            ]

            if examples:
                assert reflection_engine is not None
                try:
                    candidates = _propose_rules(
                        engine=engine,
                        tokenizer=tokenizer,
                        config=config,
                        mission=mission,
                        guidance=current_guidance,
                        examples=examples,
                        stats_by_ticket=stats_for_proposer,
                        reflection_engine=reflection_engine,
                        iteration=iteration,
                        rejected_candidate_ids=rejected_ids,
                        log_dir=mission_dir / "reflection_cache",
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("rule_search proposer failed: %s", exc)
                    candidates = []

            remaining = max(
                0, int(config.rule_search.num_candidate_rules) - len(candidates)
            )
            if remaining > 0:
                ablation_candidates = _build_ablation_candidates(
                    guidance=current_guidance,
                    max_candidates=remaining,
                    rejected_candidate_ids=rejected_ids,
                )
                candidates.extend(ablation_candidates)

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
                        "rule_search iteration %d: no candidates proposed (reflect=%d, train_pool=%d); treating as no-gain",
                        iteration,
                        len(reflect_keys),
                        len(train_pool_tickets),
                    )
            no_gain_rounds = broadcast_int(
                no_gain_rounds + 1 if is_main_process() else 0, src=0
            )
            should_stop = no_gain_rounds >= patience
            should_stop = bool(
                broadcast_int(1 if should_stop and is_main_process() else 0, src=0)
            )
            if should_stop:
                early_stop_triggered = True
                break
            continue

        # Prepare eval pool baseline (fixed for the run).
        base_eval_metrics = None
        base_eval_stats: dict[str, TicketRolloutStats] = {}
        eval_tickets_filtered = None
        if is_main_process():
            eval_tickets_filtered, _ = _filter_overlength_tickets(
                eval_tickets,
                sampler=eval_sampler,
                guidance=current_guidance,
                domain=domain,
                label=f"eval pool iter{iteration}",
            )
        eval_tickets_filtered = cast(
            list[GroupTicket] | None,
            broadcast_object(eval_tickets_filtered if is_main_process() else None, src=0),
        )
        if eval_tickets_filtered is None:
            eval_tickets_filtered = []
        base_payloads_eval, dropped_eval = _distributed_rollout_payloads(
            tickets=eval_tickets_filtered,
            sampler=eval_sampler,
            guidance=current_guidance,
            mission=mission,
            domain=domain,
            per_rank_batch_size=config.runner.per_rank_rollout_batch_size,
        )
        base_eval_stats: dict[str, TicketRolloutStats] = {}
        if is_main_process() and eval_tickets_filtered:
            if dropped_eval:
                logger.info(
                    "rule_search eval baseline dropped tickets: dropped=%d kept=%d",
                    len(dropped_eval),
                    len(base_payloads_eval),
                )
            for ticket in eval_tickets_filtered:
                if ticket.key not in base_payloads_eval:
                    continue
                verdicts = _extract_verdict_samples(
                    base_payloads_eval.get(ticket.key, [])
                )
                base_eval_stats[ticket.key] = build_ticket_stats(
                    ticket_key=ticket.key,
                    gt_label=ticket.label,
                    verdicts=verdicts,
                )
            base_eval_metrics = compute_rule_search_metrics(list(base_eval_stats.values()))

        base_eval_metrics = cast(
            EvalMetrics | None,
            broadcast_object(base_eval_metrics if is_main_process() else None, src=0),
        )
        base_eval_stats_raw = broadcast_object(
            base_eval_stats if is_main_process() else None, src=0
        )
        if base_eval_stats_raw is None:
            base_eval_stats_raw = {}
        base_eval_stats = cast(dict[str, TicketRolloutStats], base_eval_stats_raw)

        # Candidate evaluation (A/B on the same train pool, paired seeds by sampler config).
        signature_lookup = None
        if is_main_process():
            signature_lookup = {
                normalize_rule_signature(text): key
                for key, text in current_guidance.experiences.items()
            }
        best_candidate: RuleCandidatePayload | None = None
        for cand_idx, cand in enumerate(candidates, start=1):
            op = str(cand.get("op") or "upsert").strip().lower()
            if op not in {"upsert", "update", "merge", "remove"}:
                continue

            cand_text = str(cand.get("text") or "").strip()
            cand_sig = str(cand.get("signature") or "").strip()
            if not cand_sig and cand_text:
                cand_sig = normalize_rule_signature(cand_text)
            if op == "remove":
                cand_sig = str(cand.get("target_signature") or "").strip()
            candidate_id = str(cand.get("candidate_id") or "").strip() or None
            target_signature = str(cand.get("target_signature") or "").strip()
            target_signatures_raw = cand.get("target_signatures") or []
            if not isinstance(target_signatures_raw, Sequence) or isinstance(
                target_signatures_raw, (str, bytes)
            ):
                target_signatures = []
            else:
                target_signatures = [
                    str(item).strip() for item in target_signatures_raw if item
                ]
            if op in {"upsert", "update", "merge"} and not cand_text:
                continue

            candidate_guidance = None
            candidate_operation = None
            skip_candidate = False
            if is_main_process():
                assert mission_guidance_repo is not None
                reflection_id = f"rule_search_preview_{iteration}_{cand_idx}"
                assert signature_lookup is not None
                target_key = None
                merged_from: tuple[str, ...] | None = None

                if op in {"update", "remove"}:
                    target_key = signature_lookup.get(target_signature)
                    if target_key is None:
                        logger.debug(
                            "rule_search skip %s: missing target signature '%s'",
                            op,
                            target_signature,
                        )
                        skip_candidate = True
                    elif target_key.startswith("S"):
                        logger.debug(
                            "rule_search skip %s: scaffold target '%s'", op, target_key
                        )
                        skip_candidate = True
                    elif op == "remove" and target_key == "G0":
                        logger.debug(
                            "rule_search skip remove: non-removable target '%s'",
                            target_key,
                        )
                        skip_candidate = True

                if not skip_candidate and op == "merge":
                    resolved = []
                    for sig in target_signatures:
                        key = signature_lookup.get(sig)
                        if key is None or key.startswith("S") or key == "G0":
                            resolved = []
                            break
                        if key not in resolved:
                            resolved.append(key)
                    if len(resolved) < 2:
                        logger.debug(
                            "rule_search skip merge: invalid targets %s",
                            target_signatures,
                        )
                        skip_candidate = True
                    else:
                        target_key = resolved[0]
                        merged_from = tuple(resolved[1:])

                if not skip_candidate:
                    candidate_operation = ExperienceOperation(
                        op=cast(ExperienceOperationKind, op),
                        key=target_key,
                        text=cand_text or None,
                        rationale=str(cand.get("rationale") or "").strip() or None,
                        evidence=tuple(reflect_keys),
                        merged_from=merged_from,
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

            skip_candidate = bool(
                broadcast_int(1 if skip_candidate and is_main_process() else 0, src=0)
            )
            if skip_candidate:
                continue

            candidate_guidance = broadcast_object(
                candidate_guidance if is_main_process() else None, src=0
            )

            cand_train_tickets = None
            if is_main_process():
                cand_train_tickets, _ = _filter_overlength_tickets(
                    train_pool_tickets,
                    sampler=train_sampler,
                    guidance=candidate_guidance,
                    domain=domain,
                    label=f"candidate iter{iteration}#{cand_idx}",
                )
                if not cand_train_tickets:
                    logger.warning(
                        "rule_search candidate skipped: all tickets exceed max_prompt_tokens"
                    )
                    skip_candidate = True
            skip_candidate = bool(
                broadcast_int(1 if skip_candidate and is_main_process() else 0, src=0)
            )
            if skip_candidate:
                continue
            cand_train_tickets = cast(
                list[GroupTicket] | None,
                broadcast_object(
                    cand_train_tickets if is_main_process() else None, src=0
                ),
            )
            if cand_train_tickets is None:
                cand_train_tickets = []

            cand_samples, dropped_cand = _distributed_rollout_verdicts(
                tickets=cand_train_tickets,
                sampler=train_sampler,
                guidance=candidate_guidance,
                mission=mission,
                domain=domain,
                per_rank_batch_size=config.runner.per_rank_rollout_batch_size,
            )

            passed = False
            gate_stats: GateStats | None = None
            cand_metrics = None
            cand_stats_by_ticket: dict[str, TicketRolloutStats] = {}
            if is_main_process():
                # Build candidate stats and gate decision (rank 0 only).
                if dropped_cand:
                    logger.info(
                        "rule_search candidate rollout dropped tickets: dropped=%d kept=%d",
                        len(dropped_cand),
                        len(cand_samples),
                    )
                for ticket in train_pool_tickets:
                    if ticket.key not in cand_samples:
                        continue
                    verdicts = cand_samples.get(ticket.key, [])
                    cand_stats_by_ticket[ticket.key] = build_ticket_stats(
                        ticket_key=ticket.key,
                        gt_label=ticket.label,
                        verdicts=verdicts,
                    )

                common_keys = set(base_stats_by_ticket) & set(cand_stats_by_ticket)
                if not common_keys:
                    logger.warning(
                        "rule_search candidate skipped: no common tickets after drops"
                    )
                    passed = False
                    fp_delta = None
                    acc_delta = None
                    fp_improved = None
                    acc_improved = None
                    fp_increase_ok = None
                else:
                    base_stats_for_gate = {
                        key: base_stats_by_ticket[key] for key in common_keys
                    }
                    cand_stats_for_gate = {
                        key: cand_stats_by_ticket[key] for key in common_keys
                    }
                    gate_stats, passed = build_gate_stats(
                        base_stats=base_stats_for_gate,
                        new_stats=cand_stats_for_gate,
                        rer_threshold=config.rule_search.gate.min_relative_error_reduction,
                        bootstrap_iterations=config.rule_search.gate.bootstrap.iterations,
                        bootstrap_min_prob=config.rule_search.gate.bootstrap.min_prob,
                        bootstrap_seed=config.rule_search.gate.bootstrap.seed + iteration,
                        max_changed_fraction=config.rule_search.gate.max_changed_fraction,
                    )
                    cand_metrics = compute_rule_search_metrics(
                        list(cand_stats_for_gate.values())
                    )
                    base_metrics_for_gate = compute_rule_search_metrics(
                        list(base_stats_for_gate.values())
                    )
                    if op in {"update", "merge", "remove"}:
                        fp_delta = cand_metrics.fp_rate - base_metrics_for_gate.fp_rate
                        acc_delta = cand_metrics.acc - base_metrics_for_gate.acc
                        fp_improved = fp_delta < -1e-12
                        acc_improved = acc_delta > 1e-12
                        fp_increase_ok = (
                            fp_delta
                            <= float(config.rule_search.gate.max_fp_rate_increase)
                            + 1e-12
                        )
                        lifecycle_passed = (
                            fp_improved and acc_improved and fp_increase_ok
                        )
                        passed = passed and lifecycle_passed
                    else:
                        fp_delta = None
                        acc_delta = None
                        fp_improved = None
                        acc_improved = None
                        fp_increase_ok = None

            passed = bool(broadcast_int(1 if passed and is_main_process() else 0, src=0))

            eval_metrics = None
            eval_acc_drop = None
            if passed and eval_tickets and base_eval_metrics is not None:
                cand_eval_payloads, dropped_eval = _distributed_rollout_payloads(
                    tickets=eval_tickets,
                    sampler=eval_sampler,
                    guidance=candidate_guidance,
                    mission=mission,
                    domain=domain,
                    per_rank_batch_size=config.runner.per_rank_rollout_batch_size,
                )
                if is_main_process():
                    if dropped_eval:
                        logger.info(
                            "rule_search eval rollout dropped tickets: dropped=%d kept=%d",
                            len(dropped_eval),
                            len(cand_eval_payloads),
                        )
                    cand_eval_stats: dict[str, TicketRolloutStats] = {}
                    for ticket in eval_tickets:
                        if ticket.key not in cand_eval_payloads:
                            continue
                        verdicts = _extract_verdict_samples(
                            cand_eval_payloads.get(ticket.key, [])
                        )
                        cand_eval_stats[ticket.key] = build_ticket_stats(
                            ticket_key=ticket.key,
                            gt_label=ticket.label,
                            verdicts=verdicts,
                        )
                    eval_metrics = compute_rule_search_metrics(list(cand_eval_stats.values()))
                    eval_acc_drop = float(base_eval_metrics.acc) - float(
                        eval_metrics.acc
                    )

            if not is_main_process():
                continue

            decision = "promoted" if passed else "rejected"
            regressions = _candidate_regressions(
                base_stats_by_ticket,
                cand_stats_by_ticket,
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
                    "candidate_id": candidate_id,
                    "op": op,
                    "target_signature": target_signature or None,
                    "target_signatures": target_signatures or None,
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
                    "candidate_id": candidate_id,
                    "op": op,
                    "target_signature": target_signature or None,
                    "target_signatures": target_signatures or None,
                    "text": cand_text or None,
                    "rationale": str(cand.get("rationale") or "").strip() or None,
                    "source": cand.get("source"),
                    "train_n": len(train_pool_tickets),
                    "base_acc": base_metrics.acc if base_metrics else None,
                    "base_false_release_rate": _false_release_rate(base_metrics),
                    "base_false_block_rate": _false_block_rate(base_metrics),
                    "cand_acc": cand_metrics.acc if cand_metrics else None,
                    "cand_false_release_rate": _false_release_rate(cand_metrics),
                    "cand_false_block_rate": _false_block_rate(cand_metrics),
                    "eval_n": len(eval_tickets),
                    "eval_base_acc": (
                        base_eval_metrics.acc if base_eval_metrics else None
                    ),
                    "eval_base_false_release_rate": _false_release_rate(base_eval_metrics),
                    "eval_base_false_block_rate": _false_block_rate(base_eval_metrics),
                    "eval_cand_acc": eval_metrics.acc if eval_metrics else None,
                    "eval_cand_false_release_rate": _false_release_rate(eval_metrics),
                    "eval_cand_false_block_rate": _false_block_rate(eval_metrics),
                    "decision": decision,
                },
            )

            if passed and gate_stats is not None:
                if (
                    best_candidate is None
                    or gate_stats.relative_error_reduction
                    > _safe_float(best_candidate.get("relative_error_reduction", -1.0), -1.0)
                ):
                    best_candidate = cast(
                        RuleCandidatePayload,
                        {
                            "candidate_index": cand_idx,
                            "signature": cand_sig,
                            "text": cand_text or None,
                            "rationale": str(cand.get("rationale") or "").strip() or None,
                            "gate": gate_stats,
                            "cand_metrics": cand_metrics,
                            "eval_metrics": eval_metrics,
                            "eval_acc_drop": eval_acc_drop,
                            "candidate_id": candidate_id,
                            "op": op,
                            "source": cand.get("source"),
                            "target_signature": target_signature or None,
                            "target_signatures": target_signatures,
                            "operation": candidate_operation,
                        },
                    )

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
                op = best_candidate.get("operation")
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
                gate: object = best_candidate.get("gate")
                cand_metrics = best_candidate.get("cand_metrics")
                assert isinstance(
                    cand_metrics, EvalMetrics
                ), "cand_metrics must be EvalMetrics"
                _append_jsonl(
                    benchmarks_path,
                    {
                        "timestamp": time.time(),
                        "iteration": iteration,
                        "mission": mission,
                        "accepted_key": accepted_key,
                        "op": best_candidate.get("op"),
                        "target_signature": best_candidate.get("target_signature"),
                        "target_signatures": best_candidate.get("target_signatures"),
                        "candidate_id": best_candidate.get("candidate_id"),
                        "signature": best_candidate["signature"],
                        "text": best_candidate["text"],
                        "rationale": best_candidate["rationale"],
                        "train_n": len(train_pool_tickets),
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
                        "eval_n": len(eval_tickets),
                        "eval_base_acc": (
                            base_eval_metrics.acc if base_eval_metrics else None
                        ),
                        "eval_after_acc": (
                            (best_eval_metrics.acc if best_eval_metrics else None)
                            if (best_eval_metrics := cast(EvalMetrics | None, best_candidate.get("eval_metrics"))) is not None
                            else None
                        ),
                        "eval_acc_drop": best_candidate.get("eval_acc_drop"),
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
                    "rule_search iteration %d: promoted %s (key=%s, RER=%.3f)",
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
            early_stop_triggered = True
            break

        distill_cfg = config.stage_b_distillation
        distill_requested = bool(distill_cfg and distill_cfg.enabled and distill_cfg.distill_size)
        if distill_requested:
            distill_guidance = None
            if is_main_process():
                assert mission_guidance_repo is not None
                distill_guidance = mission_guidance_repo.get(mission)
            distill_guidance = broadcast_object(
                distill_guidance if is_main_process() else None, src=0
            )
            if is_main_process():
                logger.info(
                    "rule_search distill enabled (early_stop=%s); exporting ChatML for post-training",
                    early_stop_triggered,
                )
            _run_rule_search_distill(
                config=config,
                engine=engine,
                tokenizer=tokenizer,
                mission=mission,
                mission_tickets=mission_tickets,
                mission_dir=mission_dir,
                domain=domain,
                guidance=distill_guidance,
            )


def _run_rule_search_distill(
    *,
    config: StageBConfig,
    engine: GenerationEngine,
    tokenizer: PreTrainedTokenizerBase,
    mission: str,
    mission_tickets: Sequence[GroupTicket],
    mission_dir: Path,
    domain: str,
    guidance: MissionGuidance | None,
) -> None:
    distill_cfg = config.stage_b_distillation
    if distill_cfg is None or not distill_cfg.enabled:
        return
    if not mission_tickets:
        return
    distill_size = distill_cfg.distill_size
    if distill_size is None or distill_size <= 0:
        if is_main_process():
            logger.info(
                "rule_search distill skipped: stage_b_distillation.distill_size not set"
            )
        return
    if guidance is None:
        if is_main_process():
            logger.warning("rule_search distill skipped: guidance unavailable")
        return

    train_sampler_cfg = config.rule_search.train_sampler if config.rule_search else None
    if train_sampler_cfg is None:
        if is_main_process():
            logger.warning("rule_search distill skipped: train_sampler missing")
        return

    requested_temp = (
        float(distill_cfg.distill_temperature)
        if distill_cfg.distill_temperature is not None
        else None
    )
    target_temp = (
        requested_temp
        if requested_temp is not None
        else min(decode.temperature for decode in train_sampler_cfg.grid)
    )
    low_temp_grid = tuple(
        decode for decode in train_sampler_cfg.grid if decode.temperature == target_temp
    )
    if not low_temp_grid:
        closest = min(
            train_sampler_cfg.grid,
            key=lambda decode: abs(decode.temperature - target_temp),
        )
        if requested_temp is not None:
            low_temp_grid = (replace(closest, temperature=target_temp),)
        else:
            low_temp_grid = (closest,)
            target_temp = closest.temperature

    distill_sampler_cfg = SamplerConfig(
        grid=low_temp_grid,
        samples_per_decode=1,
        max_prompt_tokens=train_sampler_cfg.max_prompt_tokens,
    )
    distill_sampler = RolloutSampler(
        engine=engine, config=distill_sampler_cfg
    )

    distill_seed = (
        int(distill_cfg.distill_seed)
        if distill_cfg.distill_seed is not None
        else int(config.seed)
    )
    rng = random.Random(distill_seed)
    pool = list(mission_tickets)
    if distill_size >= len(pool):
        rng.shuffle(pool)
        distill_tickets = pool
    else:
        distill_tickets = rng.sample(pool, distill_size)

    distill_payloads, dropped_distill = _distributed_rollout_payloads(
        tickets=distill_tickets,
        sampler=distill_sampler,
        guidance=guidance,
        mission=mission,
        domain=domain,
        per_rank_batch_size=config.runner.per_rank_rollout_batch_size,
    )

    if not is_main_process():
        return

    if dropped_distill:
        logger.info(
            "rule_search distill dropped tickets: dropped=%d kept=%d",
            len(dropped_distill),
            len(distill_payloads),
        )

    distill_path = (
        Path(distill_cfg.log_chatml_path)
        if distill_cfg.log_chatml_path
        else (mission_dir / "distill_chatml.jsonl")
    )
    distill_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with distill_path.open("w", encoding="utf-8") as fh:
        for ticket in distill_tickets:
            if ticket.key not in distill_payloads:
                skipped += 1
                continue
            trajectories = distill_payloads.get(ticket.key, [])
            parsed = None
            for candidate in trajectories:
                if bool(getattr(candidate, "format_ok", False)) and getattr(
                    candidate, "verdict", None
                ) is not None:
                    parsed = candidate
                    break
            if parsed is None:
                skipped += 1
                continue
            verdict_text = (
                "通过" if getattr(parsed, "verdict", None) == "pass" else "不通过"
            )
            reason_text = str(getattr(parsed, "reason", "") or "")
            assistant_content = f"Verdict: {verdict_text}\nReason: {reason_text}"
            messages = build_messages(ticket, guidance, domain=domain)
            messages.append({"role": "assistant", "content": assistant_content})
            payload = {
                "ticket_key": ticket.key,
                "group_id": ticket.group_id,
                "mission": ticket.mission,
                "label": ticket.label,
                "messages": messages,
            }
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.write("\n")
            written += 1
    logger.info(
        "rule_search distill written: %d samples (skipped=%d, requested=%d, temp=%.3f) -> %s",
        written,
        skipped,
        len(distill_tickets),
        target_temp,
        distill_path,
    )


def _append_jsonl(path: Path, payload: UnstructuredMapping) -> None:
    payload = require_mapping(payload, context="stage_b.jsonl_payload")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


def _filter_overlength_tickets(
    tickets: Sequence[GroupTicket],
    *,
    sampler: RolloutSampler,
    guidance: MissionGuidance,
    domain: str,
    label: str,
) -> tuple[list[GroupTicket], tuple[str, ...]]:
    """Drop tickets whose prompts exceed sampler max_prompt_tokens."""

    max_prompt_tokens = sampler.config.max_prompt_tokens
    if max_prompt_tokens is None:
        return list(tickets), ()

    prompts: list[str] = []
    for ticket in tickets:
        try:
            prompts.append(sampler._build_prompt(ticket, guidance, domain=domain))
        except Exception:  # noqa: BLE001
            # Keep ticket if prompt build fails here; rollout will surface errors.
            prompts.append("")

    lengths = sampler._count_prompt_tokens(prompts)
    kept: list[GroupTicket] = []
    dropped: list[str] = []
    for ticket, length in zip(tickets, lengths):
        if length is None or length <= max_prompt_tokens:
            kept.append(ticket)
        else:
            dropped.append(ticket.key)

    if is_main_process() and dropped:
        logger.info(
            "rule_search %s prefilter dropped overlength tickets: dropped=%d kept=%d budget=%d",
            label,
            len(dropped),
            len(kept),
            max_prompt_tokens,
        )

    return kept, tuple(dropped)


def _load_jsonl(path: Path) -> list[UnstructuredMapping]:
    if not path.exists():
        return []
    items: list[UnstructuredMapping] = []
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
                items.append(require_mapping(payload, context="stage_b.jsonl_row"))
    return items


def _chunked(seq: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _shard_bounds(total: int, *, world_size: int, rank: int) -> tuple[int, int]:
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


def _hard_case_rows(
    stats_by_ticket: Mapping[str, object],
    *,
    mission: str,
    iteration: int,
    sampler: str,
    limit: int,
    reason_samples_by_ticket: Mapping[str, Sequence[str | None]] | None = None,
) -> list[HardCaseRow]:
    rows: list[HardCaseRow] = []
    for entry in stats_by_ticket.values():
        majority = getattr(entry, "majority_pred", None)
        gt_label = getattr(entry, "gt_label", None)
        if majority is None or majority == gt_label:
            continue
        ticket_key = getattr(entry, "ticket_key", None)
        verdict_samples = list(
            cast(Sequence[object], getattr(entry, "verdict_samples", ()))
        )
        reason_samples: list[str | None] = []
        majority_reason = None
        reason_counts: list[ReasonCountRow] = []
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

        row: HardCaseRow = {
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
        rows.append(row)

    rows.sort(
        key=lambda row: (
            _safe_float(row.get("hard_wrong")),
            _safe_float(row.get("difficulty")),
        ),
        reverse=True,
    )
    if limit > 0:
        rows = rows[:limit]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _candidate_regressions(
    base_stats: Mapping[str, TicketRolloutStats],
    new_stats: Mapping[str, TicketRolloutStats],
    *,
    limit: int,
) -> list[CandidateRegressionRow]:
    rows: list[CandidateRegressionRow] = []
    keys = sorted(set(base_stats.keys()) & set(new_stats.keys()))
    for key in keys:
        base = base_stats[key]
        new = new_stats[key]
        base_correct = bool(getattr(base, "majority_correct", False))
        new_correct = bool(getattr(new, "majority_correct", False))
        if not base_correct or new_correct:
            continue
        row: CandidateRegressionRow = {
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
        rows.append(row)

    rows.sort(
        key=lambda row: (
            _safe_float(row.get("cand_agreement")),
            _safe_float(row.get("base_agreement")),
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


def run_all(
    config: StageBConfig, log_level: str = "logging", *, jump_reflection: bool = False
) -> None:
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

    logger.info("Stage-B starting (rule_search only)")

    init_distributed()
    enable_tf32()
    world_size = get_world_size()
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

    excluded_ticket_keys = _load_excluded_ticket_keys(config)
    if excluded_ticket_keys:
        before = len(tickets)
        tickets = [ticket for ticket in tickets if ticket.key not in excluded_ticket_keys]
        dropped = before - len(tickets)
        if is_main_process():
            logger.info(
                "Ticket filter applied: excluded_keys=%d dropped=%d kept=%d",
                len(excluded_ticket_keys),
                dropped,
                len(tickets),
            )
            if dropped:
                shown = 0
                examples: list[str] = []
                for key in sorted(excluded_ticket_keys):
                    examples.append(key)
                    shown += 1
                    if shown >= 8:
                        break
                logger.info("Ticket filter excluded (sample): %s", ", ".join(examples))

    # Group tickets by mission
    tickets_by_mission: dict[str, list[GroupTicket]] = defaultdict(list)
    for ticket in tickets:
        tickets_by_mission[ticket.mission].append(ticket)

    logger.info(
        f"Discovered {len(tickets_by_mission)} mission(s): {list(tickets_by_mission.keys())}"
    )

    if len(tickets_by_mission) != 1:
        raise RuntimeError(
            f"Stage-B rule_search expects exactly one mission per run_name; got {len(tickets_by_mission)} missions"
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
    engine = _load_generation_engine(config)
    tokenizer = _require_engine_tokenizer(engine)

    mission = mission_name
    domain = resolve_domain_for_mission(config, mission)
    _run_rule_search_mission(
        config=config,
        engine=engine,
        tokenizer=tokenizer,
        mission=mission,
        mission_tickets=training_by_mission[mission],
        mission_dir=run_dir,
        domain=domain,
        jump_reflection=jump_reflection,
    )
    logger.info("Completed Stage-B rule_search run for mission %s", mission)


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
    parser.add_argument(
        "--jump-reflection",
        action="store_true",
        help="Skip proposer/reflection and only run baseline rollouts + dumps (for manual guidance edits).",
    )
    args = parser.parse_args()

    stage_b_config = load_stage_b_config(args.config)
    effective_level = "debug" if args.debug else args.log_level
    effective_jump_reflection = bool(args.jump_reflection) or bool(
        getattr(stage_b_config, "jump_reflection", False)
    )
    if args.step == "all":
        run_all(
            stage_b_config,
            log_level=effective_level,
            jump_reflection=effective_jump_reflection,
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported step: {args.step}")


if __name__ == "__main__":
    main()
