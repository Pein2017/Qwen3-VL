"""SFT runner - pure YAML config-driven, no CLI arguments for hyperparameters"""

import argparse
import copy
import logging
import os
import re
from dataclasses import asdict

from swift.llm.train.rlhf import SwiftRLHF
from swift.llm.train.sft import SwiftSft
from swift.trainers import TrainerFactory

from .config import ConfigLoader, SaveDelayConfig
from .datasets import DenseCaptionDataset
from .trainers import with_final_checkpoint
from .utils import enable_verbose_logging, get_logger, set_log_level


def resolve_trainer_cls(train_args):
    trainer_variant = getattr(train_args, "trainer_variant", None)
    if (
        getattr(train_args, "rlhf_type", None) == "gkd"
        and trainer_variant == "gkd_monitor"
    ):
        from .trainers import GKDTrainerWithMetrics

        trainer_cls = GKDTrainerWithMetrics
    else:
        trainer_cls = TrainerFactory.get_trainer_cls(train_args)

    save_last_epoch = getattr(train_args, "save_last_epoch", True)
    if save_last_epoch:
        return with_final_checkpoint(trainer_cls)
    return trainer_cls


# Use the model's native chat_template (JSON/Jinja) shipped with the tokenizer

logger = get_logger(__name__)


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
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (required)",
    )

    # Optional: base config for inheritance
    parser.add_argument(
        "--base_config",
        type=str,
        default=None,
        help="Path to base YAML config for inheritance (optional)",
    )

    # Runtime: debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and print full config",
    )

    # Runtime: verbose mode (all ranks log)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable logging from all ranks in distributed training",
    )

    return parser.parse_args()


def main():
    """Main training entry point - pure config-driven."""
    args = parse_args()

    # Configure logging based on runtime flags
    if args.verbose:
        enable_verbose_logging()

    if args.debug:
        set_log_level(logging.DEBUG)

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
    # Append run_name to output_dir and logging_dir to form final paths
    try:
        run_name = getattr(train_args, "run_name", None)
        training_args = getattr(train_args, "training_args", None)

        # Resolve and update output_dir
        base_output_dir = getattr(train_args, "output_dir", None)
        if base_output_dir is None and training_args is not None:
            base_output_dir = getattr(training_args, "output_dir", None)
        if run_name and base_output_dir:
            base_output_dir_norm = os.path.normpath(base_output_dir)
            if os.path.basename(base_output_dir_norm) != str(run_name):
                final_output_dir = os.path.join(base_output_dir, str(run_name))
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
        base_logging_dir = getattr(train_args, "logging_dir", None)
        if base_logging_dir is None and training_args is not None:
            base_logging_dir = getattr(training_args, "logging_dir", None)
        if run_name and base_logging_dir:
            base_logging_dir_norm = os.path.normpath(base_logging_dir)
            if os.path.basename(base_logging_dir_norm) != str(run_name):
                final_logging_dir = os.path.join(base_logging_dir, str(run_name))
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
    train_jsonl = custom_config.train_jsonl or custom_config.extra.get("jsonl")
    if not train_jsonl:
        raise ValueError("Config must specify 'custom.train_jsonl' or 'custom.jsonl'")

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
        teacher_model = getattr(sft, "teacher_model", None)
        if teacher_model is None:
            raise ValueError(
                "GKD mode requires a teacher_model. Set rlhf.teacher_model in the YAML and ensure it loads."
            )
        student_vocab = getattr(getattr(sft.model, "config", None), "vocab_size", None)
        teacher_vocab = getattr(
            getattr(teacher_model, "config", None), "vocab_size", None
        )
        if isinstance(student_vocab, int) and isinstance(teacher_vocab, int):
            if student_vocab != teacher_vocab:
                raise ValueError(
                    "Teacher/student tokenizer vocabulary size mismatch detected: "
                    f"expected {student_vocab}, got {teacher_vocab}. "
                    "Use a teacher checkpoint with a matching tokenizer/vocabulary (e.g., the same Qwen3‑VL family)."
                )

    # NOTE: Do NOT override processor normalization/rescale.
    # Qwen3-VL expects its native image preprocessing. We already pass do_resize=False at encode time.

    # Configure augmentation via YAML builder (applies only to training)
    augmenter = None
    bypass_prob = float(custom_config.bypass_prob)
    aug_cfg = custom_config.augmentation
    if isinstance(aug_cfg, dict) and aug_cfg.get("enabled", True):
        try:
            # Ensure ops are registered by importing ops module
            from .datasets.augmentation import ops as _register_ops  # noqa: F401
            from .datasets.augmentation.builder import build_compose_from_config

            augmenter = build_compose_from_config(aug_cfg)
            bypass_prob = float(aug_cfg.get("bypass_prob", custom_config.bypass_prob))
            logger.info(
                f"Augmentation pipeline built (bypass_prob={bypass_prob:.2f}, training only)"
            )
        except Exception as e:
            raise ValueError(f"Failed to build augmentation pipeline from YAML: {e}")

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
    if train_sample_limit:
        logger.info(f"Train sample limit: {train_sample_limit}")
    if val_sample_limit:
        logger.info(f"Val sample limit: {val_sample_limit}")

    # Build training dataset
    logger.info(f"Loading training dataset: {train_jsonl}")
    # Require minimal explicit keys; others have sane defaults
    if not custom_config.user_prompt or not custom_config.emit_norm:
        raise ValueError("custom.user_prompt and custom.emit_norm must be provided")

    # Extract mode control parameters
    use_summary = bool(custom_config.use_summary)

    # Prepare system prompts for the selected mode
    # The system prompt is set on the template by ConfigLoader.resolve_prompts
    system_prompt_dense = getattr(sft.template, "system", None)
    system_prompt_summary = custom_config.system_prompt_summary

    if use_summary:
        logger.info("Summary mode enabled (custom.use_summary=true)")
        if system_prompt_summary is None:
            system_prompt_summary = getattr(sft.template, "system", None)
        if system_prompt_summary is None:
            try:
                from .config.prompts import SYSTEM_PROMPT_SUMMARY

                system_prompt_summary = SYSTEM_PROMPT_SUMMARY
                logger.info("Loaded default SYSTEM_PROMPT_SUMMARY")
            except ImportError as exc:
                raise ValueError(
                    "custom.use_summary is true but no summary system prompt was provided."
                ) from exc
        system_prompt_dense = None
    else:
        logger.info("Dense mode only (custom.use_summary=false)")

    dataset_seed = 42

    dataset = DenseCaptionDataset.from_jsonl(
        train_jsonl,
        template=sft.template,
        user_prompt=custom_config.user_prompt,
        emit_norm=custom_config.emit_norm,
        augmenter=augmenter,
        bypass_prob=bypass_prob,
        sample_limit=train_sample_limit,
        use_summary=use_summary,
        system_prompt_dense=system_prompt_dense,
        system_prompt_summary=system_prompt_summary,
        seed=dataset_seed,
        toon_mode=custom_config.toon_mode,
    )
    logger.info(f"Training dataset size: {len(dataset)}")

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
            template = dataset.template
            template.set_mode("pt")
            try:
                sample_encoded = dataset[0]
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
                def repl(match: re.Match) -> str:
                    count = match.group(0).count("<|image_pad|>")
                    return f"<|image_pad|>*{count}"

                return re.sub(r"(?:<\|image_pad\|>)+", repl, text)

            raw_text = _compress_image_pad(raw_text)

            assistant_gt = None
            try:
                record_clone = copy.deepcopy(dataset.base_records[0])
                builder = dataset._create_builder(dataset.mode)
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
    if val_jsonl:
        logger.info(f"Loading validation dataset: {val_jsonl}")
        eval_dataset = DenseCaptionDataset.from_jsonl(
            val_jsonl,
            template=sft.template,
            user_prompt=custom_config.user_prompt,
            emit_norm=custom_config.emit_norm,
            augmenter=None,  # No augmentation for validation
            bypass_prob=0.0,  # Explicit: no bypass for validation
            sample_limit=val_sample_limit,
            use_summary=use_summary,
            system_prompt_dense=system_prompt_dense,
            system_prompt_summary=system_prompt_summary,
            seed=dataset_seed,
            toon_mode=custom_config.toon_mode,
        )
        logger.info(f"Validation dataset size: {len(eval_dataset)}")

    # Sample printing disabled to avoid dumping labels/ids

    # CRITICAL: Apply tuner (LoRA/adapters) before creating trainer
    logger.info("Preparing model with tuner...")
    sft.model = sft.prepare_model(
        train_args, sft.model, template=sft.template, train_dataset=dataset
    )
    logger.info(f"Model after tuner: {type(sft.model).__name__}")

    # Setup trainer
    logger.info("Setting up trainer...")
    data_collator = sft._get_data_collator()
    trainer_cls = resolve_trainer_cls(train_args)

    # Add SaveDelayCallback if save_delay_steps is configured
    callbacks = sft.callbacks.copy() if sft.callbacks else []
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
    trainer = trainer_cls(
        model=sft.model,
        args=train_args.training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        template=sft.template,
        **trainer_kwargs,
    )

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

    sft.train(trainer)


if __name__ == "__main__":
    main()
