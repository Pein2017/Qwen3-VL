#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entrypoint for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

from .config import StageBConfig, load_stage_b_config
from .eval import evaluate_holdout
from .ingest import ingest_stage_a
from .io import GuidanceRepository, serialize_selection, serialize_trajectory
from .reflection import ReflectionEngine
from .rollout import RolloutSampler
from .selection import select_for_group
from .signals import attach_signals
from .types import ExperienceRecord, GroupTicket
from .utils.seed import seed_everything

logger = logging.getLogger("stage_b.runner")


def _configure_logging(log_level: str) -> None:
    normalized = log_level.strip().lower()
    level_map = {
        "debug": logging.DEBUG,
        "logging": logging.INFO,
        "warning": logging.WARNING,
    }
    if normalized not in level_map:
        raise ValueError(
            f"Unsupported log level '{log_level}'. Choose from: {', '.join(level_map)}"
        )

    level = level_map[normalized]
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    # Explicitly set level for all stage_b loggers to ensure debug messages are shown
    # when log_level is set to DEBUG. This is necessary because child loggers may not inherit
    # the root logger's level if they were created before basicConfig was called.
    stage_b_loggers = [
        "stage_b.runner",
        "stage_b.reflection",
        "stage_b.reflection.engine",
        "stage_b.rollout",
        "stage_b.ingest",
        "stage_b.ingest.stage_a",
        "stage_b.judge",
        "stage_b.io",
        "stage_b.io.guidance",
        "stage_b.io.export",
    ]
    for logger_name in stage_b_loggers:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.setLevel(level)
        # Ensure handlers inherit the level
        for handler in logger_instance.handlers:
            handler.setLevel(level)


def _dtype_from_str(name: str):
    lowered = name.lower()
    if not hasattr(torch, lowered):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return getattr(torch, lowered)


def _load_model(config: StageBConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name_or_path, padding_side="left"
    )
    dtype = _dtype_from_str(config.model.torch_dtype)
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": config.model.device_map,
    }
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model.model_name_or_path,
        **model_kwargs,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def _prepare_mission_output_paths(mission_dir: Path) -> tuple[Path, Path, Path]:
    """Prepare output paths for a specific mission.

    Args:
        mission_dir: Directory for this mission ({root}/{run_name}/{mission})

    Returns:
        Tuple of (trajectories_path, selections_path, parquet_path)
    """
    mission_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = mission_dir / "trajectories.jsonl"
    selections_path = mission_dir / "selections.jsonl"
    parquet_path = mission_dir / "selections.parquet"

    trajectories_path.write_text("", encoding="utf-8")
    selections_path.write_text("", encoding="utf-8")

    return trajectories_path, selections_path, parquet_path


def _shuffle_indices(count: int, *, epoch: int, base_seed: int) -> List[int]:
    indices = list(range(count))
    seed_value = base_seed + epoch
    random.Random(seed_value).shuffle(indices)
    return indices


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


def _setup_mission_guidance(
    startup_path: Path, mission_dir: Path, mission: str, retention: int
) -> GuidanceRepository:
    """Load initial guidance from startup path and copy to mission directory.

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

    # Load startup guidance and extract mission-specific guidance
    if startup_path.exists():
        with startup_path.open("r", encoding="utf-8") as fh:
            startup_guidance = json.load(fh)
        # Extract mission-specific guidance if it exists
        mission_guidance = {mission: startup_guidance.get(mission, {})}
    else:
        mission_guidance = {}

    # Write mission-specific guidance to mission directory
    with mission_guidance_path.open("w", encoding="utf-8") as fh:
        json.dump(mission_guidance, fh, ensure_ascii=False, indent=2)
    logger.info(
        "Initialized mission guidance for %s at %s",
        mission,
        mission_guidance_path,
    )

    return GuidanceRepository(
        mission_guidance_path,
        retention=retention,
    )


def _split_holdout(
    mission: str, tickets: List[GroupTicket], holdout_size: int, seed: int
) -> tuple[List[GroupTicket], List[GroupTicket]]:
    if not tickets:
        return [], []

    max_holdout = min(holdout_size, max(len(tickets) - 1, 0))
    if max_holdout <= 0:
        return [], list(tickets)

    rng = random.Random(f"{seed}:{mission}")
    indices = list(range(len(tickets)))
    rng.shuffle(indices)
    holdout_indices = set(indices[:max_holdout])

    holdout = [tickets[idx] for idx in range(len(tickets)) if idx in holdout_indices]
    training = [
        tickets[idx] for idx in range(len(tickets)) if idx not in holdout_indices
    ]

    return holdout, training


def run_all(config: StageBConfig, log_level: str = "logging") -> None:
    _configure_logging(log_level)

    seed_everything(config.seed)

    run_dir = config.output.root / config.output.run_name

    # First, ingest all tickets to discover missions
    # We need a temporary guidance repo for ingestion
    temp_guidance_repo = GuidanceRepository(
        config.guidance.path,
        retention=config.guidance.retention,
    )
    logger.info("Ingesting Stage-A outputs")
    tickets = list(ingest_stage_a(config.stage_a_paths, temp_guidance_repo))
    if not tickets:
        raise RuntimeError("No Stage-A records ingested; aborting Stage-B run")

    # Group tickets by mission
    tickets_by_mission: Dict[str, List[GroupTicket]] = defaultdict(list)
    for ticket in tickets:
        tickets_by_mission[ticket.mission].append(ticket)

    logger.info(
        "Discovered %d mission(s): %s",
        len(tickets_by_mission),
        list(tickets_by_mission.keys()),
    )

    holdout_by_mission: Dict[str, List[GroupTicket]] = {}
    training_by_mission: Dict[str, List[GroupTicket]] = {}
    for mission, mission_tickets in tickets_by_mission.items():
        holdout, training = _split_holdout(
            mission,
            list(mission_tickets),
            config.evaluation.holdout_size,
            config.seed,
        )
        holdout_by_mission[mission] = holdout
        training_by_mission[mission] = training
        logger.info(
            "Mission %s split into %d training and %d holdout tickets",
            mission,
            len(training),
            len(holdout),
        )
        if not training:
            logger.warning(
                "Mission %s has no training tickets after holdout split; skipping",
                mission,
            )

    logger.info("Loading model %s", config.model.model_name_or_path)
    model, tokenizer = _load_model(config)
    sampler = RolloutSampler(model=model, tokenizer=tokenizer, config=config.sampler)

    # Process each mission separately
    total_selections = 0
    processed_missions = 0
    for mission, mission_tickets in training_by_mission.items():
        mission_holdout = holdout_by_mission.get(mission, [])
        if not mission_tickets:
            logger.warning(
                "Skipping mission %s because no training tickets remain after holdout split",
                mission,
            )
            continue

        logger.info(
            "Processing mission: %s (%d training, %d holdout)",
            mission,
            len(mission_tickets),
            len(mission_holdout),
        )

        processed_missions += 1

        # Setup mission-specific directory structure
        mission_dir = run_dir / mission

        # Setup mission-specific guidance
        mission_guidance_repo = _setup_mission_guidance(
            startup_path=config.guidance.path,
            mission_dir=mission_dir,
            mission=mission,
            retention=config.guidance.retention,
        )

        # Prepare mission-specific output paths
        trajectories_path, selections_path, parquet_path = (
            _prepare_mission_output_paths(mission_dir)
        )

        # Reflection log goes under {root}/{run_name}/{mission}/
        reflection_log_path = mission_dir / "reflection.jsonl"

        reflection_engine = ReflectionEngine(
            model=model,
            tokenizer=tokenizer,
            config=config.reflection,
            guidance_repo=mission_guidance_repo,
            reflection_log=reflection_log_path,
        )

        parquet_buffer: List[dict] = []

        def _evaluate_mission_holdout() -> dict[str, float] | None:
            if not mission_holdout:
                return None
            return evaluate_holdout(
                mission_holdout,
                sampler=sampler,
                guidance_repo=mission_guidance_repo,
                signals_config=config.signals,
                selection_config=config.selection,
            )

        total_groups = len(mission_tickets)
        for epoch in range(1, config.runner.epochs + 1):
            logger.info(
                "Starting Stage-B epoch %d/%d for mission %s",
                epoch,
                config.runner.epochs,
                mission,
            )
            ordered_indices = _shuffle_indices(
                total_groups, epoch=epoch, base_seed=config.seed
            )

            epoch_progress = tqdm(
                total=len(ordered_indices),
                desc=f"Stage-B epoch {epoch}/{config.runner.epochs} [{mission}]",
                unit="group",
            )

            reflection_cycle = 0
            pending_records: List[ExperienceRecord] = []
            last_reflection_id: str | None = None

            for position, index in enumerate(ordered_indices, start=1):
                ticket = mission_tickets[index]
                guidance_map = mission_guidance_repo.load()
                guidance = guidance_map[ticket.mission]

                parsed_map = sampler.generate_for_batch([ticket], guidance_map)
                parsed_candidates = parsed_map.get(ticket.group_id, [])
                if not parsed_candidates:
                    raise RuntimeError(
                        f"Sampler produced no valid candidates for group {ticket.group_id}"
                    )

                scored_candidates = attach_signals(
                    ticket, parsed_candidates, config.signals
                )

                for candidate in scored_candidates:
                    trajectory_payload = serialize_trajectory(
                        candidate,
                        reflection_cycle=reflection_cycle,
                        guidance_step=guidance.step,
                    )
                    trajectory_payload["epoch"] = epoch
                    _append_jsonl(trajectories_path, trajectory_payload)

                # Import summarizer
                from src.stage_b.reflection.summarizer import SampleSummarizer
                
                summarizer = SampleSummarizer()
                # Use scoring.select_for_group which supports summarizer
                from src.stage_b.scoring.selection import select_for_group as select_for_group_with_summarizer
                selection = select_for_group_with_summarizer(
                    ticket,
                    scored_candidates,
                    guidance_step=guidance.step,
                    reflection_cycle=reflection_cycle,
                    reflection_change=last_reflection_id,
                    config=config.selection,
                    summarizer=summarizer,
                )

                selection_payload = serialize_selection(selection)
                selection_payload["epoch"] = epoch
                _append_jsonl(selections_path, selection_payload)
                parquet_buffer.append(selection_payload)

                record = reflection_engine.build_record(
                    ticket,
                    scored_candidates,
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

                    pre_metrics = _evaluate_mission_holdout()
                    pre_uplift_value = (
                        pre_metrics.get("label_match_rate")
                        if pre_metrics is not None
                        else None
                    )

                    outcome = reflection_engine.reflect(
                        bundle,
                        epoch=epoch,
                        pre_uplift=pre_uplift_value,
                        log=False,
                    )

                    post_metrics = None
                    post_uplift_value = None
                    if (
                        mission_holdout
                        and outcome.eligible
                        and outcome.operations
                    ):
                        base_guidance_map = mission_guidance_repo.load()
                        preview_guidance = mission_guidance_repo.preview_reflection(
                            mission,
                            proposal=outcome.proposal,
                            reflection_id=outcome.reflection_id,
                            source_group_ids=outcome.proposal.evidence_group_ids,
                            operations=outcome.operations,
                        )
                        preview_map = dict(base_guidance_map)
                        preview_map[mission] = preview_guidance

                        post_metrics = evaluate_holdout(
                            mission_holdout,
                            sampler=sampler,
                            guidance_repo=mission_guidance_repo,
                            signals_config=config.signals,
                            selection_config=config.selection,
                            guidance_override=preview_map,
                        )
                        post_uplift_value = post_metrics.get(
                            "label_match_rate",
                            pre_uplift_value if pre_uplift_value is not None else 0.0,
                        )

                    outcome = reflection_engine.finalize_outcome(
                        outcome,
                        epoch=epoch,
                        pre_uplift=pre_uplift_value,
                        post_uplift=post_uplift_value,
                    )

                    if pre_metrics is not None:
                        metrics_source = post_metrics or pre_metrics
                        sample_size = int(
                            metrics_source.get("sample_size", len(mission_holdout))
                        )
                        post_value_for_log = (
                            post_uplift_value
                            if post_uplift_value is not None
                            else outcome.post_uplift
                        )
                        pre_value_for_log = (
                            pre_uplift_value
                            if pre_uplift_value is not None
                            else outcome.pre_uplift
                        )
                        logger.info(
                            "Holdout uplift mission=%s cycle=%d pre=%.4f post=%.4f delta=%.4f sample=%d applied=%s",
                            mission,
                            reflection_cycle,
                            pre_value_for_log,
                            post_value_for_log,
                            post_value_for_log - pre_value_for_log,
                            sample_size,
                            outcome.applied,
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
                pre_metrics = _evaluate_mission_holdout()
                pre_uplift_value = (
                    pre_metrics.get("label_match_rate")
                    if pre_metrics is not None
                    else None
                )
                outcome = reflection_engine.reflect(
                    bundle,
                    epoch=epoch,
                    pre_uplift=pre_uplift_value,
                    log=False,
                )

                post_metrics = None
                post_uplift_value = None
                if (
                    mission_holdout
                    and outcome.eligible
                    and outcome.operations
                ):
                    base_guidance_map = mission_guidance_repo.load()
                    preview_guidance = mission_guidance_repo.preview_reflection(
                        mission,
                        proposal=outcome.proposal,
                        reflection_id=outcome.reflection_id,
                        source_group_ids=outcome.proposal.evidence_group_ids,
                        operations=outcome.operations,
                    )
                    preview_map = dict(base_guidance_map)
                    preview_map[mission] = preview_guidance

                    post_metrics = evaluate_holdout(
                        mission_holdout,
                        sampler=sampler,
                        guidance_repo=mission_guidance_repo,
                        signals_config=config.signals,
                        selection_config=config.selection,
                        guidance_override=preview_map,
                    )
                    post_uplift_value = post_metrics.get(
                        "label_match_rate",
                        pre_uplift_value if pre_uplift_value is not None else 0.0,
                    )

                outcome = reflection_engine.finalize_outcome(
                    outcome,
                    epoch=epoch,
                    pre_uplift=pre_uplift_value,
                    post_uplift=post_uplift_value,
                )

                if pre_metrics is not None:
                    metrics_source = post_metrics or pre_metrics
                    sample_size = int(
                        metrics_source.get("sample_size", len(mission_holdout))
                    )
                    post_value_for_log = (
                        post_uplift_value
                        if post_uplift_value is not None
                        else outcome.post_uplift
                    )
                    pre_value_for_log = (
                        pre_uplift_value
                        if pre_uplift_value is not None
                        else outcome.pre_uplift
                    )
                    logger.info(
                        "Holdout uplift mission=%s cycle=%d pre=%.4f post=%.4f delta=%.4f sample=%d applied=%s",
                        mission,
                        reflection_cycle,
                        pre_value_for_log,
                        post_value_for_log,
                        post_value_for_log - pre_value_for_log,
                        sample_size,
                        outcome.applied,
                    )

                last_reflection_id = outcome.reflection_id if outcome.applied else None
                pending_records.clear()
                reflection_cycle += 1

        # Write parquet file for this mission
        if parquet_buffer:
            import pandas as pd

            frame = pd.json_normalize(parquet_buffer)
            frame.to_parquet(parquet_path, index=False)
            total_selections += len(parquet_buffer)

        logger.info(
            "Completed mission %s: %d selections",
            mission,
            len(parquet_buffer),
        )

    logger.info(
        "Completed Stage-B pipeline across %d epoch(s) and %d mission(s): %d total selections",
        config.runner.epochs,
        processed_missions,
        total_selections,
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
        help="Logging level for the pipeline",
    )
    args = parser.parse_args()

    stage_b_config = load_stage_b_config(args.config)
    if args.step == "all":
        run_all(stage_b_config, log_level=args.log_level)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported step: {args.step}")


if __name__ == "__main__":
    main()
