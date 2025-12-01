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
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from .config import StageBConfig, load_stage_b_config
from .critic import CriticEngine
from .ingest import ingest_stage_a
from .io.guidance import GuidanceRepository
from .io.export import serialize_selection, serialize_trajectory
from .reflection import ReflectionEngine
from .rollout import RolloutSampler
from .sampling.prompts import _render_summaries
from .signals import attach_signals
from .types import (
    DeterministicSignals,
    ExperienceRecord,
    GroupTicket,
    TrajectoryWithSignals,
)
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
        "stage_b.scoring",
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
    processor = AutoProcessor.from_pretrained(
        config.model.model_name_or_path, trust_remote_code=True
    )
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


def _shuffle_indices(count: int, *, epoch: int, base_seed: int) -> List[int]:
    indices = list(range(count))
    seed_value = base_seed + epoch
    random.Random(seed_value).shuffle(indices)
    return indices


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


def _merge_signals_with_critic(
    signals: DeterministicSignals, critic_output
) -> DeterministicSignals:
    """Propagate critic uncertainty into deterministic signals without new rules."""

    needs_review = signals.needs_manual_review
    if critic_output is not None:
        needs_review = needs_review or bool(
            critic_output.needs_recheck
            or critic_output.evidence_sufficiency is False
            or critic_output.recommended_action == "人工复核"
        )

    return DeterministicSignals(
        label_match=signals.label_match,
        self_consistency=signals.self_consistency,
        confidence=signals.confidence,
        conflict_flag=signals.conflict_flag,
        needs_manual_review=needs_review,
    )


def _setup_mission_guidance(
    startup_path: Path, mission_dir: Path, mission: str, retention: int
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

    # Load startup/global guidance and extract mission section; fail fast if missing
    startup_repo = GuidanceRepository(startup_path, retention=retention)
    startup_map = startup_repo.load()
    if mission not in startup_map:
        raise RuntimeError(
            f"Mission {mission} not found in global guidance file: {startup_path}"
        )
    seed_section = startup_map[mission].to_payload()

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
    _configure_logging(log_level)

    seed_everything(config.seed)

    run_dir = config.output.root / config.output.run_name

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

    training_by_mission = tickets_by_mission

    # Log holdout status (deferred for future implementation)
    logger.info("Holdout evaluation: disabled (deferred for future implementation)")

    logger.info(f"Loading model {config.model.model_name_or_path}")
    model, tokenizer, processor = _load_model(config)
    sampler = RolloutSampler(model=model, tokenizer=tokenizer, config=config.sampler)

    # Initialize CriticEngine if enabled (shares model with sampler and reflection)
    critic_engine = None
    if config.critic.enabled:
        logger.info("Initializing CriticEngine with shared model")
        critic_engine = CriticEngine(
            config=config.critic,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
        )

    # Process each mission separately
    total_selections = 0
    processed_missions = 0
    for mission, mission_tickets in training_by_mission.items():
        if not mission_tickets:
            logger.warning(
                f"Skipping mission {mission} because no tickets available"
            )
            continue

        logger.info(
            f"Processing mission: {mission} ({len(mission_tickets)} tickets)"
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
        for epoch in range(1, config.runner.epochs + 1):
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
            )

            reflection_cycle = 0
            pending_records: List[ExperienceRecord] = []
            last_reflection_id: Optional[str] = None

            for position, index in enumerate(ordered_indices, start=1):
                ticket = mission_tickets[index]
                guidance_map = mission_guidance_repo.load()
                guidance = guidance_map[ticket.mission]

                logger.info(
                    f"Sampling group {ticket.group_id} (mission={ticket.mission}) at guidance.step={guidance.step}"
                )


                parsed_map = sampler.generate_for_batch([ticket], guidance_map)
                parsed_candidates = parsed_map.get(ticket.group_id, [])
                if not parsed_candidates:
                    raise RuntimeError(
                        f"Sampler produced no valid candidates for group {ticket.group_id}"
                    )

                scored_candidates = attach_signals(
                    ticket, parsed_candidates, config.signals
                )

                # Evaluate candidates with CriticEngine if enabled
                if critic_engine is not None:
                    # Extract parsed and signals (guaranteed non-None from attach_signals)
                    parsed_candidates_list = []
                    signals_list_for_critic = []
                    for c in scored_candidates:
                        if c.parsed is not None and c.signals is not None:
                            parsed_candidates_list.append(c.parsed)
                            signals_list_for_critic.append(c.signals)
                    
                    stage_a_summary = _render_summaries(ticket.summaries.as_dict())

                    critic_outputs = critic_engine.evaluate(
                        group_id=ticket.group_id,
                        mission=ticket.mission,
                        candidates=parsed_candidates_list,
                        signals=signals_list_for_critic,
                        stage_a_summary=stage_a_summary,
                    )
                    # Attach critic outputs to candidates
                    enriched_candidates = []
                    for candidate, critic_output in zip(
                        scored_candidates, critic_outputs
                    ):
                        merged_signals = _merge_signals_with_critic(
                            candidate.signals, critic_output  # type: ignore[arg-type]
                        )
                        enriched_candidates.append(
                            TrajectoryWithSignals(
                                parsed=candidate.parsed,
                                signals=merged_signals,
                                critic=critic_output,
                            )
                        )
                    scored_candidates = enriched_candidates

                # Select final verdict
                from src.stage_b.scoring.selection import select_for_group

                selection = select_for_group(
                    ticket,
                    scored_candidates,
                    guidance_step=guidance.step,
                    reflection_cycle=reflection_cycle,
                    reflection_change=last_reflection_id,
                    config=config.selection,
                    manual_review=config.manual_review,
                )

                # Write trajectories with critic outputs (already attached above)
                for candidate in scored_candidates:
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

                if not selection.manual_review_recommended:
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
                    outcome = reflection_engine.reflect(
                        bundle,
                        epoch=epoch,
                        log=True,
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

                last_reflection_id = outcome.reflection_id if outcome.applied else None
                pending_records.clear()
                reflection_cycle += 1

        total_selections += mission_selection_count

        logger.info(
            f"Completed mission {mission}: {mission_selection_count} selections"
        )

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
