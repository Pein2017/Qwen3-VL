"""SFT runner - pure YAML config-driven, no CLI arguments for hyperparameters"""

import argparse
import copy
from dataclasses import asdict
import logging
import math
from multiprocessing import Manager
import os
import re
from typing import Any, MutableMapping, Optional, TypeVar, Union, cast
from collections.abc import Iterable

T = TypeVar("T")

from swift.llm.train.rlhf import SwiftRLHF
from swift.llm.train.sft import SwiftSft
from swift.trainers import TrainerFactory
from swift.utils import get_dist_setting
import torch
from transformers.trainer_utils import SaveStrategy

from .callbacks.fusion_epoch import FusionEpochCallback
from .config import ConfigLoader, SaveDelayConfig
from .config.grpo import validate_grpo_config
from .data_collators.dataset_metrics import build_dataset_metrics_collator
from .datasets import BaseCaptionDataset
from .datasets.augmentation.curriculum import AugmentationCurriculumScheduler
from .datasets.fusion import FusionConfig
from .metrics.dataset_metrics import DatasetMetricsMixin
from .rlhf.grpo import register_grpo_rewards
from .trainers import with_final_checkpoint
from .utils import configure_logging, get_logger


def resolve_trainer_cls(train_args: Any) -> type:
    trainer_variant = getattr(train_args, "trainer_variant", None)
    if (
        getattr(train_args, "rlhf_type", None) == "gkd"
        and trainer_variant == "gkd_monitor"
    ):
        from .trainers import GKDTrainerWithMetrics

        trainer_cls: type = GKDTrainerWithMetrics
    else:
        trainer_cls = TrainerFactory.get_trainer_cls(train_args)

    # Skip forced final checkpoint in debug-style runs where the user requested no saves.
    save_strategy = getattr(train_args, "save_strategy", None)
    try:
        save_strategy_enum = (
            SaveStrategy(str(save_strategy))
            if save_strategy is not None
            else SaveStrategy.NO
        )
    except Exception:
        # Accept loose string values like "none"/"NO"/"No".
        save_strategy_enum = (
            SaveStrategy.NO
            if str(save_strategy).lower() in ("no", "none")
            else SaveStrategy.STEPS
        )

    save_last_epoch = bool(getattr(train_args, "save_last_epoch", True))
    if save_last_epoch and save_strategy_enum != SaveStrategy.NO:
        return with_final_checkpoint(trainer_cls)
    return trainer_cls


# Use the model's native chat_template (JSON/Jinja) shipped with the tokenizer

logger = get_logger(__name__)

# Ensure custom GRPO rewards are registered before trainer initialization.
register_grpo_rewards()


def parse_args():
    """Parse minimal runtime arguments.

    All training configuration comes from YAML files.
    CLI only accepts runtime settings like config path and debug mode.
    """
    parser = argparse.ArgumentParser(
        description="SFT training with YAML configuration - zero CLI hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with config
  python -m src.sft --config configs/qwen3vl_lora.yaml
  
  # With inheritance from base config
  python -m src.sft --config configs/qwen3vl_lora.yaml --base_config configs/base.yaml
  
  # Debug mode
  python -m src.sft --config configs/debug.yaml --debug
        """,
    )

    # Required: config file path
    _ = parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (required)",
    )

    # Optional: base config for inheritance
    _ = parser.add_argument(
        "--base_config",
        type=str,
        default=None,
        help="Path to base YAML config for inheritance (optional)",
    )

    # Runtime: debug mode
    _ = parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and print full config",
    )

    # Runtime: verbose mode (all ranks log)
    _ = parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable logging from all ranks in distributed training",
    )

    return parser.parse_args()


def main():
    """Main training entry point - pure config-driven."""
    args = parse_args()

    # Configure logging (src namespace only)
    configure_logging(
        debug=bool(args.debug),
        verbose=bool(args.verbose or args.debug),
        level=logging.INFO,
    )

    # Suppress extremely verbose torch autograd debug logs even when global debug is on.
    # Those logs (torch.autograd.graph) dump every backward node and overwhelm output.
    try:
        torch_autograd_log = logging.getLogger("torch.autograd.graph")
        torch_autograd_log.setLevel(logging.INFO)
        torch_autograd_log.propagate = False
    except Exception:
        pass

    logger.info("=" * 70)
    logger.info("  MS-Swift Training with YAML Configuration")
    logger.info("=" * 70)
    logger.info(f"Config file: {args.config}")
    if args.base_config:
        logger.info(f"Base config: {args.base_config}")
    logger.info("=" * 70)

    # Load configuration from YAML
    logger.info("Loading configuration...")
    train_args, training_config = ConfigLoader.load_training_config(
        args.config, args.base_config
    )
    custom_config = training_config.custom
    validate_grpo_config(training_config)

    # Packing is removed; fail fast if configs still request it.
    if getattr(train_args, "packing", False):
        raise ValueError(
            "training.packing is removed; use standard padded batching (set packing=false and remove packing knobs)."
        )
    forbidden_packing_keys = {
        "custom.packing_group_key": getattr(custom_config, "packing_group_key", None),
        "custom.packing_length_override": getattr(custom_config, "packing_length_override", None),
        "custom.cached_lengths": getattr(custom_config, "cached_lengths", None),
    }
    for key, val in forbidden_packing_keys.items():
        if val:
            raise ValueError(
                f"{key} is no longer supported because packing was removed; delete this config entry."
            )
    # Append run_name to output_dir and logging_dir to form final paths
    try:
        run_name = getattr(train_args, "run_name", None)
        training_args = getattr(train_args, "training_args", None)

        # Resolve and update output_dir
        base_output_dir: Any = getattr(train_args, "output_dir", None)
        if base_output_dir is None and training_args is not None:
            base_output_dir = getattr(training_args, "output_dir", None)
        if run_name and base_output_dir:
            base_output_dir_norm = os.path.normpath(str(base_output_dir))
            if os.path.basename(base_output_dir_norm) != str(run_name):
                final_output_dir = os.path.join(str(base_output_dir), str(run_name))
                try:
                    setattr(train_args, "output_dir", final_output_dir)
                except Exception:
                    pass
                if training_args is not None:
                    try:
                        setattr(training_args, "output_dir", final_output_dir)
                    except Exception:
                        pass

        # Resolve and update logging_dir (tensorboard dir)
        base_logging_dir: Any = getattr(train_args, "logging_dir", None)
        if base_logging_dir is None and training_args is not None:
            base_logging_dir = getattr(training_args, "logging_dir", None)
        if run_name and base_logging_dir:
            base_logging_dir_norm = os.path.normpath(str(base_logging_dir))
            if os.path.basename(base_logging_dir_norm) != str(run_name):
                final_logging_dir = os.path.join(str(base_logging_dir), str(run_name))
                try:
                    setattr(train_args, "logging_dir", final_logging_dir)
                except Exception:
                    pass
                if training_args is not None:
                    try:
                        setattr(training_args, "logging_dir", final_logging_dir)
                    except Exception:
                        pass
    except Exception:
        # Non-fatal: fall back to directories as provided by YAML
        pass

    # Debug mode: print full configuration
    if args.debug:
        logger.debug("TrainArguments:")
        for key, value in vars(train_args).items():
            if not key.startswith("_"):
                logger.debug(f"  {key}: {value}")
        logger.debug("Training configuration sections:")
        logger.debug(f"  model={training_config.model}")
        logger.debug(f"  data={training_config.data}")
        logger.debug(f"  template={training_config.template}")
        logger.debug(f"  training={training_config.training}")
        logger.debug(f"  tuner={training_config.tuner}")
        logger.debug(f"  rlhf={training_config.rlhf}")
        logger.debug(f"  deepspeed={training_config.deepspeed}")
        logger.debug(f"  prompts={training_config.prompts}")
        logger.debug("Custom dataset config:")
        for key, value in asdict(custom_config).items():
            logger.debug(f"  {key}: {value}")
        logger.debug("=" * 70)

    # Auto-configure ROOT_IMAGE_DIR from JSONL directory
    train_jsonl_raw = custom_config.train_jsonl or custom_config.extra.get("jsonl")
    if not train_jsonl_raw:
        raise ValueError("Config must specify 'custom.train_jsonl' or 'custom.jsonl'")
    train_jsonl = str(train_jsonl_raw)

    if os.environ.get("ROOT_IMAGE_DIR") in (None, ""):
        root_dir = os.path.abspath(os.path.dirname(train_jsonl))
        os.environ["ROOT_IMAGE_DIR"] = root_dir
        logger.info(f"Set ROOT_IMAGE_DIR={root_dir}")

    # Initialize SwiftSft with TrainArguments object directly
    logger.info("Initializing ms-swift pipeline...")
    rlhf_type = getattr(train_args, "rlhf_type", None)
    pipeline_cls = SwiftRLHF if rlhf_type else SwiftSft
    sft = pipeline_cls(train_args)
    logger.info(f"Model: {train_args.model}")
    logger.info(f"Training type: {train_args.train_type}")
    if rlhf_type:
        logger.info(f"RLHF mode: {rlhf_type}")
    if train_args.train_type == "lora":
        logger.info(
            f"LoRA rank: {train_args.lora_rank}, alpha: {train_args.lora_alpha}"
        )

    # Early validation: ensure teacher/student vocabulary compatibility in GKD mode
    if rlhf_type == "gkd":
        teacher_model: Any = getattr(sft, "teacher_model", None)
        if teacher_model is None:
            raise ValueError(
                "GKD mode requires a teacher_model. Set rlhf.teacher_model in the YAML and ensure it loads."
            )
        student_config = getattr(sft.model, "config", None)
        student_vocab = getattr(student_config, "vocab_size", None) if student_config is not None else None
        
        teacher_config = getattr(teacher_model, "config", None)
        teacher_vocab = getattr(teacher_config, "vocab_size", None) if teacher_config is not None else None
        if isinstance(student_vocab, int) and isinstance(teacher_vocab, int):
            if student_vocab != teacher_vocab:
                raise ValueError(
                    "Teacher/student tokenizer vocabulary size mismatch detected: "
                    + f"expected {student_vocab}, got {teacher_vocab}. "
                    + "Use a teacher checkpoint with a matching tokenizer/vocabulary (e.g., the same Qwen3‑VL family)."
                )

    # NOTE: Do NOT override processor normalization/rescale.
    # Qwen3-VL expects its native image preprocessing. We already pass do_resize=False at encode time.

    # Configure augmentation via YAML builder (applies only to training)
    augmenter = None
    bypass_prob = float(custom_config.bypass_prob)
    aug_cfg = custom_config.augmentation
    curriculum_cfg = None
    if isinstance(aug_cfg, dict) and aug_cfg.get("enabled", True):
        try:
            # Ensure ops are registered by importing ops module
            from .datasets.augmentation import ops as _register_ops  # noqa: F401

            # Use _register_ops to avoid "unused import" warning if linter ignores noqa
            _ = _register_ops
            from .datasets.augmentation.builder import build_compose_from_config

            augmenter = build_compose_from_config(aug_cfg)
            bypass_prob = float(aug_cfg.get("bypass_prob", custom_config.bypass_prob))
            curriculum_cfg = aug_cfg.get("curriculum")
            logger.info(
                f"Augmentation pipeline built (bypass_prob={bypass_prob:.2f}, training only)"
            )
        except Exception as e:
            raise ValueError(f"Failed to build augmentation pipeline from YAML: {e}")

    curriculum_state = None
    curriculum_scheduler = None
    if curriculum_cfg is None:
        curriculum_cfg = custom_config.augmentation_curriculum
    if curriculum_cfg:
        if augmenter is None:
            raise ValueError(
                "augmentation curriculum requires a built augmentation pipeline"
            )
        try:
            scheduler = AugmentationCurriculumScheduler.from_config(
                base_bypass=bypass_prob,
                op_meta=getattr(augmenter, "_augmentation_meta", []),
                curriculum_raw=curriculum_cfg,
            )
        except Exception as exc:
            raise ValueError(f"Failed to build augmentation curriculum: {exc}") from exc
        curriculum_scheduler = scheduler
        # Note: initial_state will be computed after dataset is loaded and total_steps is calculated

    # Sample limits for quick smoke tests
    shared_sample_limit = custom_config.sample_limit
    train_sample_limit = (
        custom_config.train_sample_limit
        if custom_config.train_sample_limit is not None
        else shared_sample_limit
    )
    val_sample_limit = (
        custom_config.val_sample_limit
        if custom_config.val_sample_limit is not None
        else shared_sample_limit
    )
    if train_sample_limit is not None:
        logger.info(f"Train sample limit: {train_sample_limit}")
    if val_sample_limit is not None:
        logger.info(f"Val sample limit: {val_sample_limit}")


    # Build training dataset
    logger.info(f"Loading training dataset: {str(train_jsonl)}")
    # Require minimal explicit keys; others have sane defaults
    if not custom_config.user_prompt or not custom_config.emit_norm:
        raise ValueError("custom.user_prompt and custom.emit_norm must be provided")

    # Extract mode control parameters
    use_summary = bool(custom_config.use_summary)
    default_mode = "summary" if use_summary else "dense"

    # Load fusion config early to detect per-dataset modes
    fusion_config_obj: FusionConfig | None = None
    if custom_config.fusion_config:
        fusion_config_obj = FusionConfig.from_file(custom_config.fusion_config)
        if any(
            getattr(spec, "key", "").lower() in {"bbu", "rru"}
            for spec in fusion_config_obj.targets
        ):
            if not custom_config.assistant_prefix_format:
                raise ValueError(
                    "custom.assistant_prefix_format must be provided for BBU/RRU fusion training."
                )

    def _requires_summary(spec: object) -> bool:
        mode_val = getattr(spec, "mode", None)
        if mode_val not in {"dense", "summary", None}:
            name = str(getattr(spec, "name", spec))
            raise ValueError(
                f"Invalid fusion dataset mode '{mode_val}' for {name}; "
                + "expected 'dense' or 'summary'."
            )
        resolved_mode = mode_val if mode_val is not None else default_mode
        return resolved_mode == "summary"

    needs_summary_mode = use_summary or (
        fusion_config_obj is not None
        and any(
            _requires_summary(spec)
            for spec in (*fusion_config_obj.targets, *fusion_config_obj.sources)
        )
    )

    # Prepare system prompts for both modes (keep dense prompt even if summary is default)
    system_prompt_dense = getattr(sft.template, "system", None)
    try:
        if system_prompt_dense is None or use_summary:
            from .config.prompts import build_dense_system_prompt

            system_prompt_dense = build_dense_system_prompt(custom_config.json_format)
    except Exception:
        pass

    system_prompt_summary = custom_config.system_prompt_summary
    if needs_summary_mode:
        if system_prompt_summary is None and use_summary:
            system_prompt_summary = getattr(sft.template, "system", None)
        if system_prompt_summary is None:
            try:
                from .prompts.stage_a_summary import build_stage_a_system_prompt

                system_prompt_summary = build_stage_a_system_prompt()
                logger.info("Loaded Stage-A summary system prompt for training")
            except ImportError as exc:
                raise ValueError(
                    "Summary mode is enabled but no summary system prompt was provided."
                ) from exc
    else:
        system_prompt_summary = None

    if use_summary:
        logger.info(
            "Default summary mode enabled (custom.use_summary=true); fusion datasets may override per-dataset."
        )
    else:
        logger.info(
            "Default dense mode (custom.use_summary=false); set per-dataset mode in fusion config to enable summary samples."
        )

    dataset_seed = 42
    dataset: Union["FusionCaptionDataset", "BaseCaptionDataset"]
    fusion_callback: FusionEpochCallback | None = None
    if fusion_config_obj:
        from .datasets.unified_fusion_dataset import FusionCaptionDataset

        dataset = FusionCaptionDataset(
            fusion_config=fusion_config_obj,
            base_template=sft.template,
            user_prompt=custom_config.user_prompt,
            emit_norm=custom_config.emit_norm,
            json_format=custom_config.json_format,
            assistant_prefix_format=custom_config.assistant_prefix_format,
            augmenter=augmenter,
            bypass_prob=bypass_prob,
            curriculum_state=curriculum_state,
            use_summary=use_summary,
            system_prompt_dense=system_prompt_dense,
            system_prompt_summary=system_prompt_summary,
            seed=dataset_seed,
            sample_limit=train_sample_limit,
            split="train",
        )
        # Fusion datasets rebuild their schedule each epoch
        fusion_callback = FusionEpochCallback(dataset)
    else:
        dataset = BaseCaptionDataset.from_jsonl(
            train_jsonl,
            template=sft.template,
            user_prompt=custom_config.user_prompt,
            emit_norm=custom_config.emit_norm,
            json_format=custom_config.json_format,
            augmenter=augmenter,
            bypass_prob=bypass_prob,
            curriculum_state=curriculum_state,
            sample_limit=train_sample_limit,
            use_summary=use_summary,
            system_prompt_dense=system_prompt_dense,
            system_prompt_summary=system_prompt_summary,
            seed=dataset_seed,
            assistant_prefix_format=None,
        )
    logger.info(f"Training dataset size: {len(dataset)}")

    # Optional: CHORD-style SFT mixing for GRPO.
    # This provides a supervised fallback gradient even when GRPO advantages collapse (e.g., zero-variance rollouts).
    chord_cfg = custom_config.grpo.chord
    chord_enabled = chord_cfg.enabled
    chord_dataset = None

    if chord_enabled:
        sft_per_device_bs = cast(int, chord_cfg.sft_per_device_train_batch_size)
        mu_warmup_steps = cast(int, chord_cfg.mu_warmup_steps)
        mu_decay_steps = cast(int, chord_cfg.mu_decay_steps)
        mu_peak = cast(float, chord_cfg.mu_peak)
        mu_valley = cast(float, chord_cfg.mu_valley)
        enable_phi = bool(chord_cfg.enable_phi_function)

        # Avoid sharing the same dataset instance between GRPO and CHORD dataloaders:
        # FusionCaptionDataset maintains per-epoch RNG/shuffling state; a second instance reduces coupling.
        if fusion_config_obj:
            from .datasets.unified_fusion_dataset import FusionCaptionDataset

            chord_dataset = FusionCaptionDataset(
                fusion_config=fusion_config_obj,
                base_template=sft.template,
                user_prompt=custom_config.user_prompt,
                emit_norm=custom_config.emit_norm,
                json_format=custom_config.json_format,
                assistant_prefix_format=custom_config.assistant_prefix_format,
                augmenter=augmenter,
                bypass_prob=bypass_prob,
                curriculum_state=curriculum_state,
                use_summary=use_summary,
                system_prompt_dense=system_prompt_dense,
                system_prompt_summary=system_prompt_summary,
                seed=dataset_seed,
                sample_limit=train_sample_limit,
                split="train",
            )
        else:
            chord_dataset = BaseCaptionDataset.from_jsonl(
                train_jsonl,
                template=sft.template,
                user_prompt=custom_config.user_prompt,
                emit_norm=custom_config.emit_norm,
                json_format=custom_config.json_format,
                augmenter=augmenter,
                bypass_prob=bypass_prob,
                curriculum_state=curriculum_state,
                sample_limit=train_sample_limit,
                use_summary=use_summary,
                system_prompt_dense=system_prompt_dense,
                system_prompt_summary=system_prompt_summary,
                seed=dataset_seed,
                assistant_prefix_format=None,
            )

        logger.info(
            "CHORD enabled for GRPO: sft_per_device_bs=%s mu_warmup_steps=%s mu_decay_steps=%s mu_peak=%s mu_valley=%s phi=%s",
            sft_per_device_bs,
            mu_warmup_steps,
            mu_decay_steps,
            mu_peak,
            mu_valley,
            enable_phi,
        )

        # Populate ms-swift CHORD arguments on both outer args and nested training_args (GRPOConfig).
        for args_obj in (train_args, getattr(train_args, "training_args", None)):
            if args_obj is None:
                continue
            try:
                setattr(args_obj, "chord_sft_per_device_train_batch_size", sft_per_device_bs)
                setattr(args_obj, "chord_mu_warmup_steps", mu_warmup_steps)
                setattr(args_obj, "chord_mu_decay_steps", mu_decay_steps)
                setattr(args_obj, "chord_mu_peak", mu_peak)
                setattr(args_obj, "chord_mu_valley", mu_valley)
                setattr(args_obj, "chord_enable_phi_function", enable_phi)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to attach CHORD settings to training arguments; ms-swift interface may have changed."
                ) from exc

        # DeepSpeed guardrail: ms-swift CHORD mixing performs an extra forward pass per step.
        # DeepSpeed ZeRO-2 (stage 1/2) may assert on duplicated gradient reductions when parameters
        # receive multiple gradient contributions in a single backward. Prefer ZeRO-3 for CHORD.
        def _infer_deepspeed_zero_stage(ds_value: object) -> int | None:
            if ds_value is None:
                return None
            if isinstance(ds_value, str):
                name = ds_value.strip()
                known = {
                    "zero0": 0,
                    "zero1": 1,
                    "zero2": 2,
                    "zero2_offload": 2,
                    "zero3": 3,
                    "zero3_offload": 3,
                }
                if name in known:
                    return known[name]
                if name.endswith(".json") and os.path.exists(name):
                    try:
                        import json

                        with open(name, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        stage = cfg.get("zero_optimization", {}).get("stage")
                        return int(stage) if stage is not None else None
                    except Exception:
                        return None
                return None
            if isinstance(ds_value, dict):
                stage = ds_value.get("zero_optimization", {}).get("stage")
                try:
                    return int(stage) if stage is not None else None
                except Exception:
                    return None
            return None

        ds_cfg = getattr(train_args, "deepspeed", None)
        ds_stage = _infer_deepspeed_zero_stage(ds_cfg)
        if ds_stage is not None and ds_stage in (1, 2):
            raise ValueError(
                "CHORD is enabled (custom.grpo.chord.enabled=true) but DeepSpeed ZeRO stage is "
                f"{ds_stage}. This configuration is known to trigger DeepSpeed assertions like "
                "\"Gradient computed twice for this partition\". "
                "Set deepspeed.config to zero3/zero3_offload (recommended) or disable CHORD."
            )

    # Calculate total_steps and initialize curriculum_state if needed
    if curriculum_scheduler is not None:
        requires_total_steps = bool(getattr(curriculum_scheduler, "_requires_total_steps", False))
        if requires_total_steps:
            # Calculate total_steps from dataset length, epochs, batch size, etc.
            num_train_epochs = getattr(train_args, "num_train_epochs", None)
            per_device_train_batch_size = getattr(
                train_args, "per_device_train_batch_size", None
            )
            gradient_accumulation_steps = getattr(
                train_args, "gradient_accumulation_steps", None
            )
            max_steps = getattr(train_args, "max_steps", None)

            if max_steps is not None and int(str(max_steps)) > 0:
                total_steps = int(str(max_steps))
            elif (
                num_train_epochs is not None
                and per_device_train_batch_size is not None
                and gradient_accumulation_steps is not None
            ):
                _, _, world_size, _ = get_dist_setting()
                if world_size <= 0:
                    world_size = 1
                len_dataset = len(dataset)
                total_train_batch_size = (
                    int(str(per_device_train_batch_size))
                    * int(str(gradient_accumulation_steps))
                    * world_size
                )
                num_update_steps_per_epoch = max(
                    len_dataset // total_train_batch_size, 1
                )
                total_steps = math.ceil(float(str(num_train_epochs)) * num_update_steps_per_epoch)
            else:
                raise ValueError(
                    "Cannot calculate total_steps for curriculum scheduler. "
                    + "Need either max_steps or (num_train_epochs, per_device_train_batch_size, gradient_accumulation_steps)"
                )
            curriculum_scheduler.set_total_steps(total_steps)
            logger.info(f"Curriculum scheduler: set total_steps={total_steps}")

        # Now get initial state and create curriculum_state
        initial_state = curriculum_scheduler.get_state(0)
        manager = Manager()
        curriculum_state = cast(
            MutableMapping[str, object],
            manager.dict(
            {
                "step": 0,
                "bypass_prob": initial_state["bypass_prob"],
                "ops": copy.deepcopy(initial_state["ops"]),
            }
            ),
        )
        # Update dataset's curriculum_state
        set_curriculum = getattr(dataset, "set_curriculum_state", None)
        if callable(set_curriculum):
            set_curriculum(curriculum_state)
        else:
            preprocessor = getattr(dataset, "preprocessor", None)
            if preprocessor is not None:
                setattr(preprocessor, "curriculum_state", curriculum_state)

    # Optional: multimodal health check (only in --debug mode)
    if args.debug:
        try:
            sample = dataset[0]
            img_grid = sample.get("image_grid_thw")
            pv = sample.get("pixel_values")
            input_ids = sample.get("input_ids")
            logger.debug(f"HealthCheck: keys={list(sample.keys())}")
            if img_grid is None or pv is None:
                raise ValueError(
                    "Encoded sample missing image_grid_thw/pixel_values. Check image paths and template preprocessing."
                )
            # Print basic shapes
            try:
                grid_shape = tuple(getattr(img_grid, "shape", []))
            except Exception:
                grid_shape = None
            try:
                pv_shape = tuple(getattr(pv, "shape", []))
            except Exception:
                pv_shape = None
            logger.debug(
                f"image_grid_thw shape: {grid_shape}; pixel_values shape: {pv_shape}"
            )

            # Token count sanity vs grid tokens
            image_token_id = getattr(dataset.template, "image_token_id", None)
            merge = getattr(
                getattr(dataset.template, "processor", None), "image_processor", None
            )
            merge_size = getattr(merge, "merge_size", 1)
            expected = None
            if hasattr(img_grid, "prod"):
                try:
                    expected = int(
                        img_grid.prod(dim=-1).sum().item() // (merge_size**2)
                    )
                except Exception:
                    expected = None
            if (
                isinstance(image_token_id, int)
                and isinstance(input_ids, list)
                and expected is not None
            ):
                actual = sum(1 for t in input_ids if t == image_token_id)
                logger.debug(f"image tokens: expected≈{expected}, actual={actual}")
                if actual == 0 or abs(actual - expected) > max(8, expected // 10):
                    logger.warning(
                        "Image token mismatch. Investigate chat_template and image processing."
                    )
        except Exception as e:
            logger.warning(f"HealthCheck failed: {e}")

    # Optional: dump conversation text-only (no tokens, no images) and full tokens
    dump_conv = bool(custom_config.dump_conversation_text or args.debug)
    if dump_conv and len(dataset) > 0:
        try:
            dataset_typed: Union["FusionCaptionDataset", "BaseCaptionDataset"] = dataset
            template = dataset_typed.template
            template.set_mode("pt")
            try:
                sample_encoded = dataset_typed[0]
            finally:
                template.set_mode("train")

            input_ids = (
                sample_encoded.get("input_ids")
                if isinstance(sample_encoded, dict)
                else None
            )
            if input_ids is None:
                raise ValueError(
                    "Sample does not contain input_ids for dumping conversation text."
                )
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()

            raw_text = template.tokenizer.decode(input_ids, skip_special_tokens=False)

            def _compress_image_pad(text: str) -> str:
                def repl(match: re.Match[str]) -> str:
                    count = match.group(0).count("<|image_pad|>")
                    return f"<|image_pad|>*{count}"

                return re.sub(r"(?:<\|image_pad\|>)+", repl, text)

            raw_text = _compress_image_pad(raw_text)

            assistant_gt = None
            try:
                # Re-assign to typed variable to help linter
                dataset_for_dump: Union["FusionCaptionDataset", "BaseCaptionDataset"] = dataset
                record_clone = copy.deepcopy(dataset_for_dump.base_records[0])
                builder = dataset_for_dump._create_builder(dataset_for_dump.mode)
                merged = builder.build_many([record_clone])
                assistant_turn = next(
                    (
                        turn
                        for turn in merged.get("messages", [])
                        if turn.get("role") == "assistant"
                    ),
                    None,
                )
                if assistant_turn:
                    contents = assistant_turn.get("content") or []
                    for item in contents:
                        if isinstance(item, dict) and item.get("type") == "text":
                            assistant_gt = item.get("text")
                            break
            except Exception as inner_e:
                logger.warning(f"Failed to extract assistant GT: {inner_e}")

            logger.debug("Conversation (raw):\n" + raw_text)
            if assistant_gt:
                logger.debug("Assistant GT:\n" + assistant_gt)

            dump_path = custom_config.dump_conversation_path or "conversation_text.txt"
            if not os.path.isabs(dump_path):
                base_output_dir = getattr(train_args, "output_dir", None)
                base_dir = (
                    base_output_dir if isinstance(base_output_dir, str) else os.getcwd()
                )
                dump_path = os.path.join(base_dir, dump_path)
            dump_dir = os.path.dirname(dump_path) or "."
            os.makedirs(dump_dir, exist_ok=True)
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write(raw_text)
                if not raw_text.endswith("\n"):
                    f.write("\n")
                if assistant_gt:
                    if not raw_text.endswith("\n"):
                        f.write("\n")
                    f.write("\n--- Assistant GT ---\n")
                    f.write(assistant_gt)
                    if not assistant_gt.endswith("\n"):
                        f.write("\n")
            logger.info(f"Conversation text saved to: {dump_path}")
        except Exception as e:
            logger.warning(f"Failed to dump conversation text: {e}")

    # Build validation dataset if provided
    eval_dataset = None
    val_jsonl = custom_config.val_jsonl
    if fusion_config_obj:
        target_val_path = None
        has_source_val = any(
            src.val_jsonl is not None for src in fusion_config_obj.sources
        )
        if val_jsonl:
            logger.info(
                "Fusion mode: ignoring custom.val_jsonl override and using per-target "
                f"validation splits; sources_with_val={has_source_val}"
            )
        else:
            logger.info(
                "Loading fusion validation dataset using per-target val_jsonl from config; "
                f"sources_with_val={has_source_val}"
            )

        from .datasets.unified_fusion_dataset import FusionCaptionDataset

        eval_dataset = FusionCaptionDataset(
            fusion_config=fusion_config_obj,
            base_template=sft.template,
            user_prompt=custom_config.user_prompt,
            emit_norm=custom_config.emit_norm,
            json_format=custom_config.json_format,
            assistant_prefix_format=custom_config.assistant_prefix_format,
            augmenter=None,  # No augmentation for validation
            bypass_prob=0.0,
            curriculum_state=None,
            use_summary=use_summary,
            system_prompt_dense=system_prompt_dense,
            system_prompt_summary=system_prompt_summary,
            seed=dataset_seed,
            shuffle=False,
            sample_limit=val_sample_limit,
            split="eval",
            target_eval_jsonl=target_val_path,
            include_source_eval=False,
        )
        logger.info(f"Validation dataset size: {len(eval_dataset)}")
    else:
        eval_path_fallback = val_jsonl
        if eval_path_fallback:
            logger.info(f"Loading validation dataset: {eval_path_fallback}")
            eval_dataset = BaseCaptionDataset.from_jsonl(
                eval_path_fallback,
                template=sft.template,
                user_prompt=custom_config.user_prompt,
                emit_norm=custom_config.emit_norm,
                json_format=custom_config.json_format,
                augmenter=None,  # No augmentation for validation
                bypass_prob=0.0,  # Explicit: no bypass for validation
                sample_limit=val_sample_limit,
                use_summary=use_summary,
                system_prompt_dense=system_prompt_dense,
                system_prompt_summary=system_prompt_summary,
                seed=dataset_seed,
                assistant_prefix_format=None,
            )
            logger.info(f"Validation dataset size: {len(eval_dataset)}")

    # Domain labels for per-dataset telemetry (padding-only path)
    dataset_domains: Optional[dict[str, str]] = None
    if fusion_config_obj:
        domains: dict[str, str] = {}
        for tgt in fusion_config_obj.targets:
            domains[tgt.name] = "target"
        for src in fusion_config_obj.sources:
            domains[src.name] = "source"
        dataset_domains = domains

    # Sample printing disabled to avoid dumping labels/ids

    # CRITICAL: Apply tuner (LoRA/adapters) before creating trainer
    logger.info("Preparing model with tuner...")
    sft.model = sft.prepare_model(
        train_args, sft.model, template=sft.template, train_dataset=dataset
    )
    logger.info(f"Model after tuner: {type(sft.model).__name__}")

    # Setup trainer
    logger.info("Setting up trainer...")
    base_collator = sft._get_data_collator()
    token_type_cfg = getattr(custom_config, "token_type_metrics", None)
    data_collator = build_dataset_metrics_collator(
        sft.template, base_collator, token_type_cfg=token_type_cfg
    )
    trainer_cls = resolve_trainer_cls(train_args)
    if not issubclass(trainer_cls, DatasetMetricsMixin):
        trainer_cls = type(
            f"{trainer_cls.__name__}WithDatasetMetrics",
            (DatasetMetricsMixin, trainer_cls),
            {},
        )

    # Add SaveDelayCallback if save_delay_steps is configured
    callbacks = sft.callbacks.copy() if sft.callbacks else []
    if curriculum_scheduler is not None and curriculum_state is not None:
        from .callbacks.augmentation_curriculum import (
            AugmentationCurriculumCallback,
        )

        callbacks.append(
            AugmentationCurriculumCallback(
                scheduler=curriculum_scheduler,
                curriculum_state=curriculum_state,
            )
        )
    if fusion_callback is not None:
        callbacks.append(fusion_callback)
    save_delay_cfg = getattr(train_args, "save_delay_config", None)
    from .callbacks import SaveDelayCallback

    if isinstance(save_delay_cfg, SaveDelayConfig) and save_delay_cfg.active:
        callbacks.append(SaveDelayCallback(config=save_delay_cfg))
        delay_info = (
            f"step {save_delay_cfg.steps}"
            if save_delay_cfg.steps is not None
            else f"epoch {save_delay_cfg.epochs}"
        )
        logger.info(
            f"SaveDelayCallback enabled: checkpoint saves blocked until {delay_info}"
        )
    else:
        save_delay_steps = getattr(train_args, "save_delay_steps", None)
        save_delay_epochs = getattr(train_args, "save_delay_epochs", None)
        if save_delay_steps is not None and save_delay_steps > 0:
            callbacks.append(SaveDelayCallback(save_delay_steps=save_delay_steps))
            logger.info(
                f"SaveDelayCallback enabled: no checkpoints until step {save_delay_steps}"
            )
        elif save_delay_epochs is not None and float(save_delay_epochs) > 0:
            callbacks.append(
                SaveDelayCallback(save_delay_epochs=float(save_delay_epochs))
            )
            logger.info(
                f"SaveDelayCallback enabled: no checkpoints until epoch {float(save_delay_epochs):.2f}"
            )

    trainer_kwargs = (
        sft._get_trainer_kwargs() if hasattr(sft, "_get_trainer_kwargs") else {}
    )
    # NOTE: Do NOT pass chord_sft_dataset via GRPOTrainer constructor.
    # ms-swift's GRPOTrainer calls _prepare_algorithm_params() before SwiftMixin/Trainer
    # initialization, and CHORD dataloader creation requires trainer.accelerator.
    # Initializing CHORD post-construction avoids AttributeError: no attribute 'accelerator'.
    # trainer_cls is dynamically determined and accepts standard trainer parameters
    # Type checker can't infer the exact constructor signature, so we ignore these errors
    trainer_ctor = cast(Any, trainer_cls)
    trainer = trainer_ctor(
        model=sft.model,  # type: ignore[arg-type]
        args=train_args.training_args,  # type: ignore[arg-type]
        data_collator=data_collator,  # type: ignore[arg-type]
        train_dataset=dataset,  # type: ignore[arg-type]
        eval_dataset=eval_dataset,  # type: ignore[arg-type]
        callbacks=callbacks,  # type: ignore[arg-type]
        template=sft.template,  # type: ignore[arg-type]
        **trainer_kwargs,
    )

    # Post-init CHORD wiring (GRPO-only). This must happen after trainer.accelerator exists.
    if chord_enabled and chord_dataset is not None:
        try:
            from swift.trainers.rlhf_trainer.utils import make_chord_sft_dataset

            chord_iter = make_chord_sft_dataset(trainer, chord_dataset)
            if chord_iter is None:
                raise RuntimeError(
                    "CHORD was enabled but CHORD iterator initialization returned None."
                )
            setattr(trainer, "chord_sft_iterator", chord_iter)
            logger.info(
                "CHORD SFT iterator prepared (post-init): sft_per_device_bs=%s",
                getattr(getattr(trainer, "args", None), "chord_sft_per_device_train_batch_size", None),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize CHORD SFT iterator post-init. "
                "This usually indicates a CHORD dataset/labels issue or an ms-swift incompatibility."
            ) from exc
    if dataset_domains is not None:
        try:
            setattr(trainer, "dataset_domains", dict(dataset_domains))
        except Exception:
            pass

    # Patch DeepSpeed __del__ to avoid noisy cleanup errors (safe no-op)
    try:
        import deepspeed  # type: ignore

        runtime_mod = getattr(deepspeed, "runtime", None)
        engine_mod = getattr(runtime_mod, "engine", None)
        ds_engine = getattr(engine_mod, "DeepSpeedEngine", None)
        if ds_engine is not None and hasattr(ds_engine, "__del__"):
            _orig_ds_del = ds_engine.__del__

            def _safe_ds_del(self):  # type: ignore[override]
                try:
                    _orig_ds_del(self)
                except Exception:
                    # Suppress non-fatal cleanup errors
                    pass

            ds_engine.__del__ = _safe_ds_del
    except Exception:
        pass

    # Start training
    logger.info("=" * 70)
    logger.info("  Starting Training")
    logger.info("=" * 70)
    logger.info(f"  Output directory: {train_args.output_dir}")
    logger.info(f"  Epochs: {train_args.num_train_epochs}")
    per_device_batch = getattr(train_args, "per_device_train_batch_size", None)
    grad_accum_steps = getattr(train_args, "gradient_accumulation_steps", None)
    if isinstance(per_device_batch, int) and isinstance(grad_accum_steps, int):
        logger.info(f"  Effective batch size: {per_device_batch * grad_accum_steps}")
    else:
        logger.info(
            "  Effective batch size: unavailable (missing batch or accumulation settings)"
        )
    logger.info("=" * 70)

    try:
        sft.train(trainer)
    except torch.cuda.OutOfMemoryError:
        debug_info = getattr(dataset, "last_sample_debug", None)
        logger.error(f"CUDA OOM encountered. Last sample debug: {debug_info}")
        raise
    finally:
        # Explicit cleanup to prevent DeepSpeed cleanup errors during GC
        # This addresses a known DeepSpeed issue where __del__ can fail
        # when accessing bf16_groups that are already partially destroyed
        try:
            # Check if DeepSpeed is enabled
            if (
                hasattr(trainer, "is_deepspeed_enabled")
                and trainer.is_deepspeed_enabled  # type: ignore[attr-defined]
            ):
                model_wrapped = getattr(trainer, "model_wrapped", None)
                # model_wrapped IS the DeepSpeed engine when DeepSpeed is enabled
                if model_wrapped is not None:
                    try:
                        # Patch the optimizer's destroy method to prevent IndexError
                        # This is safer than calling destroy() which can still fail
                        optimizer = getattr(model_wrapped, "optimizer", None)
                        if optimizer is not None and hasattr(optimizer, "destroy"):
                            original_destroy = optimizer.destroy

                            def safe_destroy():
                                try:
                                    original_destroy()
                                except (IndexError, AttributeError, RuntimeError):
                                    # Silently ignore errors during optimizer cleanup
                                    # These are harmless - training already completed
                                    pass

                            optimizer.destroy = safe_destroy

                        # Now safe to call engine destroy
                        if hasattr(model_wrapped, "destroy"):
                            model_wrapped.destroy()
                        logger.debug("DeepSpeed engine cleaned up successfully")
                    except Exception as cleanup_error:
                        # Ignore cleanup errors - they're harmless at this point
                        # Training already completed successfully
                        logger.debug(
                            f"DeepSpeed cleanup warning (non-fatal): {cleanup_error}"
                        )
        except Exception as e:
            # Non-fatal: training already completed successfully
            logger.debug(f"Cleanup warning (non-fatal): {e}")


if __name__ == "__main__":
    main()
