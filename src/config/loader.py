"""Pure YAML config loader - directly instantiates ms-swift objects"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from swift.llm.argument import RLHFArguments, TrainArguments
from swift.utils import get_dist_setting

from .prompts import (
    SYSTEM_PROMPT_SUMMARY,
    USER_PROMPT_JSON,
    USER_PROMPT_SUMMARY,
    build_dense_system_prompt,
)
from .schema import PromptOverrides, SaveDelayConfig, TrainingConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load YAML config and directly instantiate ms-swift dataclasses.

    No CLI argument parsing - direct object construction from YAML.
    All hyperparameters must be explicitly defined in YAML.
    """

    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """Load YAML file into dictionary.

        Args:
            config_path: Path to YAML config file

        Returns:
            Configuration dictionary
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    @staticmethod
    def _normalize_to_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return [str(value)]

    @staticmethod
    def _coerce_bool(value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value in (0, 1, 0.0, 1.0):
                return bool(value)
            raise ValueError(f"{field_name} must be boolean (0 or 1), got {value!r}.")
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
            raise ValueError(
                f"{field_name} string value '{value}' is not a recognized boolean representation."
            )
        raise TypeError(f"{field_name} must be a boolean value, got {type(value)!r}.")

    @staticmethod
    def load_yaml_with_extends(
        config_path: str, _visited: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Load YAML and resolve inheritance via 'extends'/'inherit'.

        Supports a top-level key in the YAML:
          - extends: str | list[str]     # relative to the current file
          - inherit: str | list[str]     # alias of extends

        Bases are merged in order (earlier are lower precedence).
        The current file has the highest precedence.
        Cycles are detected and will raise a ValueError.
        """
        abs_path = str(Path(config_path).resolve())
        visited: Set[str] = set(_visited or set())
        if abs_path in visited:
            raise ValueError(f"Cyclic config inheritance detected at: {abs_path}")
        visited.add(abs_path)

        current_dir = Path(abs_path).parent
        config = ConfigLoader.load_yaml(abs_path) or {}

        # Gather base paths from supported keys
        extends_value = None
        if isinstance(config, dict):
            extends_value = config.pop("extends", None)
            if extends_value is None:
                extends_value = config.pop("inherit", None)

        base_paths = ConfigLoader._normalize_to_list(extends_value)

        # Merge all bases in order
        merged_base: Dict[str, Any] = {}
        for base_ref in base_paths:
            base_path = Path(base_ref)
            if not base_path.is_absolute():
                base_path = (current_dir / base_path).resolve()
            base_cfg = ConfigLoader.load_yaml_with_extends(str(base_path), visited)
            merged_base = ConfigLoader.merge_configs(merged_base, base_cfg)

        # Finally merge current file on top
        return ConfigLoader.merge_configs(merged_base, config)

    @staticmethod
    def merge_configs(base: Dict, override: Dict) -> Dict:
        """Deep merge two config dictionaries.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        merged = base.copy()
        for key, value in override.items():
            if (
                isinstance(value, dict)
                and key in merged
                and isinstance(merged[key], dict)
            ):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def resolve_prompts(config: Dict[str, Any]) -> PromptOverrides:
        prompts_config = config.get("prompts", {}) or {}
        if not isinstance(prompts_config, dict):
            raise TypeError("prompts section must be a mapping if provided")

        use_summary = False
        custom_section = config.get("custom")
        json_format_hint: Optional[str] = None
        if custom_section is not None:
            if not isinstance(custom_section, dict):
                raise TypeError(
                    "custom section must be a mapping when resolving prompts"
                )
            if "summary_ratio" in custom_section:
                raise ValueError(
                    "custom.summary_ratio has been removed; use custom.use_summary instead."
                )
            if "use_summary" in custom_section:
                use_summary = ConfigLoader._coerce_bool(
                    custom_section["use_summary"], "custom.use_summary"
                )
            json_format_hint_raw = custom_section.get("json_format")
            if json_format_hint_raw is not None:
                json_format_hint = str(json_format_hint_raw)

        if use_summary:
            default_system = SYSTEM_PROMPT_SUMMARY
            default_user = USER_PROMPT_SUMMARY
            output_variant = "summary"
        else:
            default_system = build_dense_system_prompt(json_format_hint)
            default_user = USER_PROMPT_JSON
            output_variant = "dense"

        system_prompt = prompts_config.get("system", default_system)
        user_prompt = prompts_config.get("user", default_user)

        return PromptOverrides(
            system=str(system_prompt) if system_prompt is not None else None,
            user=str(user_prompt) if user_prompt is not None else None,
            output_variant=output_variant,
        )

    @staticmethod
    def build_train_arguments(config: TrainingConfig) -> TrainArguments:
        """Directly instantiate TrainArguments from config.

        TrainArguments is a unified dataclass that inherits from:
        - Seq2SeqTrainingArguments (HuggingFace Transformers)
        - TunerArguments (LoRA, adapters, etc.)
        - DataArguments (dataset configuration)
        - ModelArguments (model loading)
        - QuantizeArguments (quantization)
        - TemplateArguments (prompt templates)
        - SwanlabArguments (logging)

        We merge all config sections and pass to TrainArguments constructor,
        which will use ms-swift's built-in defaults for any missing fields.

        Args:
            config: Configuration dictionary from YAML

        Returns:
            Fully initialized TrainArguments object
        """
        model_section = dict(config.model)
        quant_section = dict(config.quantization)
        data_section = dict(config.data)
        template_section = dict(config.template)
        tuner_section = dict(config.tuner)
        training_section = dict(config.training)
        rlhf_section_original = dict(config.rlhf)
        rlhf_section = dict(rlhf_section_original)
        llm_kd_weight_raw = rlhf_section.pop("llm_kd_weight", None)
        if llm_kd_weight_raw is None:
            llm_kd_weight = 1.0
        else:
            try:
                llm_kd_weight = float(llm_kd_weight_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("rlhf.llm_kd_weight must be a numeric value") from exc
            if not math.isfinite(llm_kd_weight):
                raise ValueError(
                    f"rlhf.llm_kd_weight must be finite, got {llm_kd_weight_raw!r}"
                )
            if llm_kd_weight < 0:
                raise ValueError(
                    f"rlhf.llm_kd_weight must be >= 0, got {llm_kd_weight_raw!r}"
                )

        raw_save_delay_steps = training_section.pop("save_delay_steps", None)
        raw_save_delay_epochs = training_section.pop("save_delay_epochs", None)
        save_last_epoch_raw = training_section.pop("save_last_epoch", None)
        if save_last_epoch_raw is None:
            save_last_epoch = True
        else:
            save_last_epoch = ConfigLoader._coerce_bool(
                save_last_epoch_raw, "training.save_last_epoch"
            )

        # Auto-calculate gradient_accumulation_steps from effective_batch_size
        effective_batch_size = training_section.pop("effective_batch_size", None)
        if effective_batch_size is not None:
            try:
                effective_batch_size = int(effective_batch_size)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "training.effective_batch_size must be an integer"
                ) from exc
            if effective_batch_size <= 0:
                raise ValueError(
                    f"training.effective_batch_size must be > 0, got {effective_batch_size}"
                )

            per_device_train_batch_size = training_section.get(
                "per_device_train_batch_size", 1
            )
            try:
                per_device_train_batch_size = int(per_device_train_batch_size)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "training.per_device_train_batch_size must be an integer"
                ) from exc
            if per_device_train_batch_size <= 0:
                raise ValueError(
                    f"training.per_device_train_batch_size must be > 0, got {per_device_train_batch_size}"
                )

            # Get world_size (number of GPUs) from environment
            _, _, world_size, _ = get_dist_setting()
            if world_size <= 0:
                world_size = 1

            # Calculate gradient_accumulation_steps
            # Formula: effective_batch_size = per_device_train_batch_size × world_size × gradient_accumulation_steps
            denominator = per_device_train_batch_size * world_size
            gradient_accumulation_steps = max(
                1, math.ceil(effective_batch_size / denominator)
            )
            training_section["gradient_accumulation_steps"] = (
                gradient_accumulation_steps
            )

            logger.info(
                f"Auto-calculated gradient_accumulation_steps={gradient_accumulation_steps} "
                f"from effective_batch_size={effective_batch_size}, "
                f"per_device_train_batch_size={per_device_train_batch_size}, "
                f"world_size={world_size}"
            )

        if config.global_max_length is not None:
            model_section.setdefault("max_model_len", config.global_max_length)
            template_section.setdefault("max_length", config.global_max_length)

        if "system" not in template_section and config.prompts.system:
            template_section["system"] = config.prompts.system

        teacher_model_path = rlhf_section_original.get("teacher_model")
        rlhf_type = rlhf_section_original.get("rlhf_type")
        llm_kd_active = rlhf_type == "gkd" and llm_kd_weight > 0
        kd_requested = llm_kd_active or config.custom.visual_kd.enabled
        if kd_requested and not teacher_model_path:
            raise ValueError(
                "rlhf.teacher_model must be provided when llm KD or visual KD is enabled. "
                "Set rlhf.llm_kd_weight to 0 and disable custom.visual_kd to run without a teacher."
            )

        args_dict: Dict[str, Any] = {}
        for section in (
            model_section,
            quant_section,
            data_section,
            template_section,
            tuner_section,
            training_section,
            rlhf_section,
        ):
            if section:
                args_dict.update(section)

        if config.deepspeed and config.deepspeed.enabled:
            args_dict["deepspeed"] = config.deepspeed.config

        save_delay_config = SaveDelayConfig.from_raw(
            raw_save_delay_steps, raw_save_delay_epochs
        )

        args_cls = RLHFArguments if args_dict.get("rlhf_type") else TrainArguments
        train_args = args_cls(**args_dict)

        try:
            setattr(train_args, "save_last_epoch", save_last_epoch)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "Unable to attach save_last_epoch to TrainArguments; ensure ms-swift exposes this attribute."
            ) from exc

        if config.custom.trainer_variant:
            try:
                setattr(train_args, "trainer_variant", config.custom.trainer_variant)
            except Exception as exc:  # pragma: no cover - explicit failure
                raise RuntimeError(
                    "Unable to attach trainer_variant to TrainArguments; update ms-swift if interface changed."
                ) from exc

        setattr(train_args, "save_delay_config", save_delay_config)
        if save_delay_config.steps is not None:
            setattr(train_args, "save_delay_steps", save_delay_config.steps)
        if save_delay_config.epochs is not None:
            setattr(train_args, "save_delay_epochs", save_delay_config.epochs)

        try:
            setattr(train_args, "visual_kd_config", config.custom.visual_kd)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach visual_kd_config to TrainArguments; ensure ms-swift exposes this attribute."
            ) from exc

        try:
            setattr(train_args, "llm_kd_weight", llm_kd_weight)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach llm_kd_weight to TrainArguments; ensure ms-swift exposes this attribute."
            ) from exc

        inner_args = getattr(train_args, "training_args", None)
        if inner_args is None:
            raise RuntimeError(
                "TrainArguments missing nested training_args; ms-swift interface may have changed."
            )

        try:
            setattr(inner_args, "save_last_epoch", save_last_epoch)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach save_last_epoch to inner training arguments; ensure ms-swift exposes this attribute."
            ) from exc

        try:
            setattr(inner_args, "visual_kd_config", config.custom.visual_kd)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach visual_kd_config to inner training arguments; ensure ms-swift exposes this attribute."
            ) from exc

        try:
            setattr(inner_args, "llm_kd_weight", llm_kd_weight)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach llm_kd_weight to inner training arguments; ensure ms-swift exposes this attribute."
            ) from exc

        return train_args

    @staticmethod
    def _materialize_training_config(
        raw_config: Dict[str, Any], prompts: PromptOverrides
    ) -> TrainingConfig:
        try:
            return TrainingConfig.from_mapping(raw_config, prompts)
        except TypeError as exc:
            raise ValueError(
                "configuration must define a 'custom' mapping with dataset parameters"
            ) from exc

    @staticmethod
    def load_training_config(
        config_path: str, base_config_path: Optional[str] = None
    ) -> tuple[TrainArguments, TrainingConfig]:
        config = ConfigLoader.load_yaml_with_extends(config_path)

        if base_config_path:
            base_config = ConfigLoader.load_yaml_with_extends(base_config_path)
            config = ConfigLoader.merge_configs(base_config, config)

        prompts = ConfigLoader.resolve_prompts(config)
        materialized = ConfigLoader._materialize_training_config(config, prompts)
        train_args = ConfigLoader.build_train_arguments(materialized)

        return train_args, materialized


__all__ = ["ConfigLoader"]
