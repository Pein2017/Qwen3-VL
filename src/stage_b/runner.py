#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entrypoint for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)

from ..utils import configure_logging, get_logger
from .config import StageBConfig, load_stage_b_config
from .ingest import ingest_stage_a
from .io.export import serialize_selection, serialize_trajectory
from .io.group_report import build_group_report
from .io.guidance import GuidanceRepository
from .reflection import ReflectionEngine
from .rollout import RolloutSampler
from .sampling.prompts import build_messages
from .types import (
    DeterministicSignals,
    ExperienceRecord,
    GroupTicket,
    TrajectoryWithSignals,
)
from .utils.seed import seed_everything

logger = get_logger("stage_b.runner")


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
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": config.model.device_map,
    }
    if variant == "vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model.model_name_or_path,
            **model_kwargs,
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
        mission_dir: Directory for this mission ({root}/{run_name}/{mission})

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
    for filename in (
        "trajectories.jsonl",
        "selections.jsonl",
        "manual_review_queue.jsonl",
        "failure_malformed.jsonl",
        "reflection.jsonl",
    ):
        path = mission_dir / filename
        if path.exists():
            path.unlink()
        path.write_text("", encoding="utf-8")

    cache_dir = mission_dir / "reflection_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def _shuffle_indices(count: int, *, epoch: int, base_seed: int) -> List[int]:
    indices = list(range(count))
    seed_value = base_seed + epoch
    random.Random(seed_value).shuffle(indices)
    return indices


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


def _chunked(seq: List, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


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
        mission_dir: Directory for this mission ({root}/{run_name}/{mission})
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
        "Stage-B starting with reflection-first pipeline (single mission, no critic)"
    )

    seed_everything(config.seed)

    run_dir = config.output.root / config.output.run_name
    if run_dir.exists():
        logger.info(
            "Cleaning existing run directory to avoid stale artifacts: %s", run_dir
        )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Ingest all tickets to discover missions
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

    training_by_mission = tickets_by_mission

    # Log holdout status (deferred for future implementation)
    logger.info("Holdout evaluation: disabled (deferred for future implementation)")

    logger.info(f"Loading model {config.model.model_name_or_path}")
    model, tokenizer, processor = _load_model(config)
    sampler = RolloutSampler(model=model, tokenizer=tokenizer, config=config.sampler)

    # Process each mission separately
    total_selections = 0
    processed_missions = 0
    for mission, mission_tickets in training_by_mission.items():
        if not mission_tickets:
            logger.warning(f"Skipping mission {mission} because no tickets available")
            continue

        logger.info(f"Processing mission: {mission} ({len(mission_tickets)} tickets)")

        processed_missions += 1

        # Setup mission-specific directory structure
        mission_dir = run_dir / mission
        _reset_mission_artifacts(mission_dir)
        distill_cfg = config.stage_b_distillation
        distill_enabled = bool(distill_cfg.enabled) if distill_cfg else False
        default_distill_path = mission_dir / "distill_chatml.jsonl"
        distill_path = (
            Path(distill_cfg.log_chatml_path)
            if distill_enabled and distill_cfg and distill_cfg.log_chatml_path
            else default_distill_path
        )

        # Setup mission-specific guidance
        mission_guidance_repo = _setup_mission_guidance(
            startup_path=config.guidance.path,
            mission_dir=mission_dir,
            mission=mission,
            retention=config.guidance.retention,
            reset=config.guidance.reset_on_rerun,
        )

        # Prepare mission-specific output paths
        trajectories_path, selections_path = _prepare_mission_output_paths(mission_dir)

        # Reflection log goes under {root}/{run_name}/{mission}/
        reflection_log_path = mission_dir / "reflection.jsonl"

        reflection_engine = ReflectionEngine(
            model=model,
            tokenizer=tokenizer,
            config=config.reflection,
            guidance_repo=mission_guidance_repo,
            reflection_log=reflection_log_path,
        )

        mission_selection_count = 0
        total_groups = len(mission_tickets)
        # Initialize loop-scoped flags to satisfy static analysis and clarify defaults
        guidance_updated_epoch: bool = False
        distill_records_epoch: List[Dict[str, object]] = []
        epoch = 0
        for epoch in range(1, config.runner.epochs + 1):
            guidance_updated_epoch = False
            distill_records_epoch = []
            logger.info(
                f"Starting Stage-B epoch {epoch}/{config.runner.epochs} for mission {mission}"
            )
            ordered_indices = _shuffle_indices(
                total_groups, epoch=epoch, base_seed=config.seed
            )

            epoch_progress = tqdm(
                total=len(ordered_indices),
                desc=f"Stage-B epoch {epoch}/{config.runner.epochs} [{mission}]",
                unit="group",
                mininterval=0.1,  # Update at least every 0.1 seconds
                maxinterval=1.0,  # Force update every 1 second max
            )

            reflection_cycle = 0
            pending_records: List[ExperienceRecord] = []
            last_reflection_id: Optional[str] = None
            last_applied_rule_keys: List[
                str
            ] = []  # Track rules applied in last reflection

            manual_review_path = mission_dir / "manual_review_queue.jsonl"
            failure_path = mission_dir / "failure_malformed.jsonl"

            epoch_tickets = [mission_tickets[i] for i in ordered_indices]

            for batch in _chunked(epoch_tickets, config.runner.rollout_batch_size):
                guidance_map = mission_guidance_repo.load()

                logger.debug(
                    "Rollout batch len=%d (configured=%d), grid=%d, samples_per_decode=%d",
                    len(batch),
                    config.runner.rollout_batch_size,
                    len(config.sampler.grid),
                    config.sampler.samples_per_decode,
                )

                parsed_map = sampler.generate_for_batch(batch, guidance_map)

                for ticket in batch:
                    guidance = guidance_map[ticket.mission]

                    logger.info(
                        f"Sampling group {ticket.group_id} (mission={ticket.mission}) at guidance.step={guidance.step}"
                    )

                    parsed_candidates = parsed_map.get(ticket.group_id, [])
                    if not parsed_candidates:
                        logger.warning(
                            "No parsed candidates for %s; queuing manual review",
                            ticket.group_id,
                        )
                        failure_entry = {
                            "group_id": ticket.group_id,
                            "mission": ticket.mission,
                            "reason": "no_candidates",
                            "raw_text": None,
                        }
                        _append_jsonl(failure_path, failure_entry)
                        manual_entry = {
                            "group_id": ticket.group_id,
                            "mission": ticket.mission,
                            "label": ticket.label,
                            "model_verdict": None,
                            "reason": "no_candidates",
                        }
                        _append_jsonl(manual_review_path, manual_entry)
                        continue

                    wrapped_candidates: List[TrajectoryWithSignals] = []
                    for cand in parsed_candidates:
                        reason_text = cand.reason or ""
                        verdict_val = cand.verdict
                        contains_review = any(
                            term in reason_text for term in ("需复核", "需人工复核")
                        )
                        format_ok = (
                            cand.format_ok
                            and verdict_val is not None
                            and bool(reason_text.strip())
                            and not (verdict_val == "pass" and contains_review)
                        )
                        if not format_ok:
                            failure_entry = {
                                "group_id": ticket.group_id,
                                "mission": ticket.mission,
                                "reason": "format_error",
                                "raw_text": cand.base.response_text,
                            }
                            _append_jsonl(failure_path, failure_entry)
                            manual_entry = {
                                "group_id": ticket.group_id,
                                "mission": ticket.mission,
                                "label": ticket.label,
                                "model_verdict": None,
                                "reason": "format_error",
                            }
                            _append_jsonl(manual_review_path, manual_entry)
                            continue

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

                    if not wrapped_candidates:
                        # nothing usable for this group
                        manual_entry = {
                            "group_id": ticket.group_id,
                            "mission": ticket.mission,
                            "label": ticket.label,
                            "model_verdict": None,
                            "reason": "no_valid_candidates",
                        }
                        _append_jsonl(manual_review_path, manual_entry)
                        continue

                    # Select final verdict
                    from src.stage_b.scoring.selection import select_for_group

                    try:
                        selection = select_for_group(
                            ticket,
                            wrapped_candidates,
                            guidance_step=guidance.step,
                            reflection_cycle=reflection_cycle,
                            reflection_change=last_reflection_id,
                            config=config.selection,
                            manual_review=config.manual_review,
                        )
                    except Exception as exc:  # noqa: BLE001
                        failure_entry = {
                            "group_id": ticket.group_id,
                            "mission": ticket.mission,
                            "reason": f"selection_error: {exc}",
                        }
                        _append_jsonl(failure_path, failure_entry)
                        manual_entry = {
                            "group_id": ticket.group_id,
                            "mission": ticket.mission,
                            "label": ticket.label,
                            "model_verdict": None,
                            "reason": "selection_error",
                        }
                        _append_jsonl(manual_review_path, manual_entry)
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
                        _append_jsonl(trajectories_path, trajectory_payload)

                    selection_payload = serialize_selection(selection)
                    selection_payload["epoch"] = epoch
                    _append_jsonl(selections_path, selection_payload)
                    mission_selection_count += 1

                    # Buffer selected candidate for potential distillation logging
                    if distill_enabled:
                        distill_records_epoch.append(
                            {
                                "ticket": ticket,
                                "verdict": selection.verdict,
                                "reason": selection.reason,
                            }
                        )

                    # Two-line protocol: always forward usable candidates to reflection
                    record = reflection_engine.build_record(
                        ticket,
                        wrapped_candidates,
                        selection.selected_candidate,
                        guidance.step,
                    )
                    pending_records.append(record)

                    # Trigger reflection when batch is full
                    if len(pending_records) >= config.reflection.batch_size:
                        bundle = reflection_engine.build_bundle(
                            pending_records,
                            reflection_cycle=reflection_cycle,
                        )
                        outcome = reflection_engine.reflect(
                            bundle,
                            epoch=epoch,
                            log=True,
                        )
                        if outcome.applied:
                            guidance_updated_epoch = True
                        if outcome.proposal.uncertainty_note == "no_evidence_for_label":
                            # No evidence supporting the label - route to manual review
                            for rec in pending_records:
                                _append_jsonl(
                                    manual_review_path,
                                    {
                                        "group_id": rec.ticket.group_id,
                                        "mission": rec.ticket.mission,
                                        "label": rec.ticket.label,
                                        "model_verdict": None,
                                        "reason": "no_evidence_for_label",
                                    },
                                )
                            logger.info(
                                f"Reflection found no evidence for {len(pending_records)} tickets; routed to manual review"
                            )
                        elif not outcome.applied and not outcome.operations:
                            for rec in pending_records:
                                has_support = any(
                                    cand.signals and cand.signals.label_match is True
                                    for cand in rec.candidates
                                )
                                if not has_support:
                                    _append_jsonl(
                                        manual_review_path,
                                        {
                                            "group_id": rec.ticket.group_id,
                                            "mission": rec.ticket.mission,
                                            "label": rec.ticket.label,
                                            "model_verdict": None,
                                            "reason": "no_support_after_reflection",
                                        },
                                    )
                        # If winning candidate still mismatches label, enqueue manual review
                        for rec in pending_records:
                            if rec.winning_candidate is None:
                                continue
                            win_idx = rec.winning_candidate
                            win_cand = next(
                                (
                                    c
                                    for c in rec.candidates
                                    if c.candidate_index == win_idx
                                ),
                                None,
                            )
                            if (
                                win_cand
                                and win_cand.signals
                                and win_cand.signals.label_match is False
                            ):
                                _append_jsonl(
                                    manual_review_path,
                                    {
                                        "group_id": rec.ticket.group_id,
                                        "mission": rec.ticket.mission,
                                        "label": rec.ticket.label,
                                        "model_verdict": win_cand.verdict,
                                        "reason": "label_mismatch_after_reflection",
                                    },
                                )
                                # Update miss_count for rules that led to this failed prediction
                                if last_applied_rule_keys:
                                    mission_guidance_repo.increment_miss_count(
                                        mission, last_applied_rule_keys
                                    )
                                    logger.debug(
                                        f"Updated miss_count for rules {last_applied_rule_keys} due to prediction failure on {rec.ticket.group_id}"
                                    )
                                    guidance_updated_epoch = True

                        # Extract and track applied rules for miss_count updates
                        if outcome.applied:
                            applied_keys = []
                            for op in outcome.operations:
                                # Track both upsert (add/update) operations
                                if op.op == "upsert" and op.key:
                                    applied_keys.append(op.key)
                                elif op.op == "upsert" and op.text:
                                    # For new rules without explicit key, use text as identifier
                                    applied_keys.append(
                                        f"_text_{hash(op.text) % (10**8)}"
                                    )
                            last_applied_rule_keys = applied_keys
                            logger.debug(
                                f"Reflection applied {len(applied_keys)} rule(s): {applied_keys}"
                            )

                        last_reflection_id = (
                            outcome.reflection_id if outcome.applied else None
                        )
                        pending_records.clear()
                        reflection_cycle += 1

                    epoch_progress.update(1)
                    epoch_progress.set_postfix(
                        {"group": ticket.group_id, "verdict": selection.verdict}
                    )

            epoch_progress.close()

            if pending_records:
                bundle = reflection_engine.build_bundle(
                    pending_records,
                    reflection_cycle=reflection_cycle,
                )
                outcome = reflection_engine.reflect(
                    bundle,
                    epoch=epoch,
                    log=True,
                )
                if outcome.applied:
                    guidance_updated_epoch = True
                if outcome.proposal.uncertainty_note == "no_evidence_for_label":
                    # No evidence supporting the label - route to manual review
                    for rec in pending_records:
                        _append_jsonl(
                            mission_dir / "manual_review_queue.jsonl",
                            {
                                "group_id": rec.ticket.group_id,
                                "mission": rec.ticket.mission,
                                "label": rec.ticket.label,
                                "model_verdict": None,
                                "reason": "no_evidence_for_label",
                            },
                        )
                    logger.info(
                        f"Reflection found no evidence for {len(pending_records)} tickets; routed to manual review"
                    )
                elif not outcome.applied and not outcome.operations:
                    for rec in pending_records:
                        has_support = any(
                            cand.signals and cand.signals.label_match is True
                            for cand in rec.candidates
                        )
                        if not has_support:
                            _append_jsonl(
                                mission_dir / "manual_review_queue.jsonl",
                                {
                                    "group_id": rec.ticket.group_id,
                                    "mission": rec.ticket.mission,
                                    "label": rec.ticket.label,
                                    "model_verdict": None,
                                    "reason": "no_support_after_reflection",
                                },
                            )
                # If winning candidate still mismatches label, enqueue manual review
                for rec in pending_records:
                    if rec.winning_candidate is None:
                        continue
                    win_idx = rec.winning_candidate
                    win_cand = next(
                        (c for c in rec.candidates if c.candidate_index == win_idx),
                        None,
                    )
                    if (
                        win_cand
                        and win_cand.signals
                        and win_cand.signals.label_match is False
                    ):
                        _append_jsonl(
                            mission_dir / "manual_review_queue.jsonl",
                            {
                                "group_id": rec.ticket.group_id,
                                "mission": rec.ticket.mission,
                                "label": rec.ticket.label,
                                "model_verdict": win_cand.verdict,
                                "reason": "label_mismatch_after_reflection",
                            },
                        )
                        # Update miss_count for rules that led to this failed prediction
                        if last_applied_rule_keys:
                            mission_guidance_repo.increment_miss_count(
                                mission, last_applied_rule_keys
                            )
                            logger.debug(
                                f"Updated miss_count for rules {last_applied_rule_keys} due to prediction failure on {rec.ticket.group_id}"
                            )
                            guidance_updated_epoch = True

                # Extract and track applied rules from end-of-epoch reflection
                if outcome.applied:
                    applied_keys = []
                    for op in outcome.operations:
                        # Track both upsert (add/update) operations
                        if op.op == "upsert" and op.key:
                            applied_keys.append(op.key)
                        elif op.op == "upsert" and op.text:
                            # For new rules without explicit key, use text as identifier
                            applied_keys.append(f"_text_{hash(op.text) % (10**8)}")
                    last_applied_rule_keys = applied_keys
                    logger.debug(
                        f"End-of-epoch reflection applied {len(applied_keys)} rule(s): {applied_keys}"
                    )

                last_reflection_id = outcome.reflection_id if outcome.applied else None
                pending_records.clear()
                reflection_cycle += 1

        # Epoch-end cleanup: remove low-confidence rules
        if config.guidance_lifecycle and config.guidance_lifecycle.enable_auto_cleanup:
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

        if config.output.group_report:
            try:
                report_path = build_group_report(mission_dir, config.stage_a_paths)
                logger.info("Wrote grouped report to %s", report_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to build group report for %s: %s", mission, exc)

        # If guidance did not change this epoch and distillation is enabled, emit distill chatml and stop further epochs
        if distill_enabled and not guidance_updated_epoch:
            guidance_map = mission_guidance_repo.load()
            if mission not in guidance_map:
                raise RuntimeError(f"Mission {mission} guidance missing at convergence")
            guidance = guidance_map[mission]
            distill_path.parent.mkdir(parents=True, exist_ok=True)
            with distill_path.open("w", encoding="utf-8") as fh:
                for record in distill_records_epoch:
                    ticket: GroupTicket = record["ticket"]  # type: ignore[assignment]
                    verdict: str = record["verdict"]  # type: ignore[assignment]
                    reason: Optional[str] = record["reason"]  # type: ignore[assignment]
                    messages = build_messages(ticket, guidance)
                    verdict_text = "通过" if verdict == "pass" else "不通过"
                    reason_text = reason or ""
                    assistant_content = (
                        f"Verdict: {verdict_text}\nReason: {reason_text}"
                    )
                    messages.append({"role": "assistant", "content": assistant_content})
                    payload = {
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
