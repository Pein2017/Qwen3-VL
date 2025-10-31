"""Pure YAML config loader - directly instantiates ms-swift objects"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from swift.llm.argument import RLHFArguments, TrainArguments

from .prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_SUMMARY, USER_PROMPT
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

        summary_ratio = 0.0
        custom_section = config.get("custom")
        if custom_section is not None:
            if not isinstance(custom_section, dict):
                raise TypeError("custom section must be a mapping when resolving prompts")
            if "summary_ratio" in custom_section:
                sr_raw = custom_section["summary_ratio"]
                try:
                    summary_ratio = float(sr_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "custom.summary_ratio must be numeric if provided"
                    ) from exc

        default_system = SYSTEM_PROMPT
        output_variant = "dense"
        if summary_ratio >= 1.0:
            default_system = SYSTEM_PROMPT_SUMMARY
            output_variant = "summary"

        system_prompt = prompts_config.get("system", default_system)
        user_prompt = prompts_config.get("user", USER_PROMPT)

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
        rlhf_section = dict(config.rlhf)

        raw_save_delay_steps = training_section.pop("save_delay_steps", None)
        raw_save_delay_epochs = training_section.pop("save_delay_epochs", None)

        if config.global_max_length is not None:
            model_section.setdefault("max_model_len", config.global_max_length)
            template_section.setdefault("max_length", config.global_max_length)

        if "system" not in template_section and config.prompts.system:
            template_section["system"] = config.prompts.system

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

        inner_args = getattr(train_args, "training_args", None)
        if inner_args is None:
            raise RuntimeError(
                "TrainArguments missing nested training_args; ms-swift interface may have changed."
            )

        try:
            setattr(inner_args, "visual_kd_config", config.custom.visual_kd)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach visual_kd_config to inner training arguments; ensure ms-swift exposes this attribute."
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
