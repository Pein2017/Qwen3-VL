"""Typed configuration schemas for training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping, MutableMapping
from typing import Any, Literal, cast

AllowedNorm = Literal["none", "norm100", "norm1000"]
AllowedVisualDistance = Literal["mse", "cosine"]
AllowedJsonFormat = Literal["standard"]

ALLOWED_JSON_FORMATS: set[str] = {"standard"}


def _normalize_json_format(value: object) -> AllowedJsonFormat:
    if not isinstance(value, str):
        raise TypeError("custom.json_format must be a string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in ALLOWED_JSON_FORMATS:
        raise ValueError("custom.json_format must be 'standard'")
    return cast(AllowedJsonFormat, normalized)


def _as_dict(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section must be a mapping, got {type(value)!r}")
    return value



@dataclass(frozen=True)
class PromptOverrides:
    system: str | None = None
    user: str | None = None
    output_variant: Literal["dense", "summary"] = "dense"


@dataclass(frozen=True)
class TokenTypeMetricsConfig:
    enabled: bool = False
    include: tuple[str, ...] = ("target", "lvis")
    exclude: tuple[str, ...] = ("coig_lang_chat",)

    def __post_init__(self) -> None:
        # Normalize to lowercase strings for stable comparisons
        inc = tuple(str(v).strip().lower() for v in self.include)
        exc = tuple(str(v).strip().lower() for v in self.exclude)
        object.__setattr__(self, "include", inc)
        object.__setattr__(self, "exclude", exc)

    @classmethod
    def from_mapping(cls, payload: object) -> "TokenTypeMetricsConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("custom.token_type_metrics must be a mapping when provided")

        enabled = bool(payload.get("enabled", False))
        include_raw = payload.get("include", cls.include)
        exclude_raw = payload.get("exclude", cls.exclude)

        def _to_tuple(value: object, field: str) -> tuple[str, ...]:
            if value is None:
                return ()
            if isinstance(value, (list, tuple)):
                return tuple(str(v).strip() for v in value)
            return (str(value).strip(),)

        include = _to_tuple(include_raw, "include")
        exclude = _to_tuple(exclude_raw, "exclude")

        return cls(enabled=enabled, include=include, exclude=exclude)


@dataclass(frozen=True)
class DeepSpeedConfig:
    enabled: bool
    config: Any

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any] | None
    ) -> DeepSpeedConfig | None:
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError("deepspeed section must be a mapping")
        if "enabled" not in payload:
            raise ValueError("deepspeed.enabled must be explicitly set")

        enabled = bool(payload["enabled"])

        if enabled:
            if "config" not in payload:
                raise ValueError(
                    "deepspeed.config must be provided when deepspeed.enabled is true"
                )
            config_value = payload["config"]
        else:
            config_value = payload.get("config")

        if enabled and (config_value is None or config_value == ""):
            raise ValueError(
                "deepspeed.config must be a non-empty value when deepspeed.enabled is true"
            )
        return cls(enabled=enabled, config=config_value)


@dataclass(frozen=True)
class SaveDelayConfig:
    steps: int | None = None
    epochs: float | None = None

    @classmethod
    def from_raw(cls, steps: Any, epochs: Any) -> "SaveDelayConfig":
        parsed_steps: int | None = None
        if steps is not None:
            try:
                value = int(steps)
            except (TypeError, ValueError) as exc:
                raise ValueError("save_delay_steps must be an integer") from exc
            if value > 0:
                parsed_steps = value

        parsed_epochs: float | None = None
        if epochs is not None:
            try:
                value = float(epochs)
            except (TypeError, ValueError) as exc:
                raise ValueError("save_delay_epochs must be numeric") from exc
            if value > 0:
                parsed_epochs = value

        return cls(steps=parsed_steps, epochs=parsed_epochs)

    @property
    def active(self) -> bool:
        return (self.steps or 0) > 0 or (self.epochs or 0.0) > 0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "SaveDelayConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("save_delay section must be a mapping")
        steps = payload.get("steps")
        epochs = payload.get("epochs")
        return cls.from_raw(steps, epochs)


@dataclass(frozen=True)
class VisualKDTargetConfig:
    enabled: bool = False
    weight: float = 0.0
    distance: AllowedVisualDistance = "mse"

    def __post_init__(self) -> None:
        if self.enabled and self.weight <= 0:
            raise ValueError("visual_kd.*.weight must be > 0 when enabled")
        if self.distance not in {"mse", "cosine"}:
            raise ValueError("visual_kd.*.distance must be one of {mse, cosine}")


@dataclass(frozen=True)
class VisualKDConfig:
    enabled: bool
    vit: VisualKDTargetConfig = field(default_factory=VisualKDTargetConfig)
    aligner: VisualKDTargetConfig = field(default_factory=VisualKDTargetConfig)
    deepstack: VisualKDTargetConfig = field(default_factory=VisualKDTargetConfig)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if not (self.vit.enabled or self.aligner.enabled or self.deepstack.enabled):
            raise ValueError(
                "custom.visual_kd must enable at least one of vit/aligner/deepstack when visual_kd.enabled is true"
            )

    @classmethod
    def disabled(cls) -> "VisualKDConfig":
        return cls(enabled=False)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "VisualKDConfig":
        if payload is None:
            return cls.disabled()
        if not isinstance(payload, Mapping):
            raise TypeError("custom.visual_kd must be a mapping when provided")

        enabled = bool(payload.get("enabled", False))
        if not enabled:
            return cls.disabled()

        def parse_target(
            name: str, raw: Mapping[str, Any] | None
        ) -> VisualKDTargetConfig:
            if raw is None:
                return VisualKDTargetConfig()
            if not isinstance(raw, Mapping):
                raise TypeError(
                    f"custom.visual_kd.{name} must be a mapping when provided"
                )

            target_enabled = bool(raw.get("enabled", False))
            raw_weight = raw.get("weight", 0.0)
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"custom.visual_kd.{name}.weight must be numeric"
                ) from exc

            raw_distance = raw.get("distance", "mse")
            if not isinstance(raw_distance, str):
                raise TypeError(f"custom.visual_kd.{name}.distance must be a string")
            distance = cast(AllowedVisualDistance, raw_distance.lower())

            if distance not in {"mse", "cosine"}:
                raise ValueError(
                    f"custom.visual_kd.{name}.distance must be one of {{mse, cosine}}"
                )

            return VisualKDTargetConfig(
                enabled=target_enabled,
                weight=weight,
                distance=distance,
            )

        vit_cfg = parse_target("vit", payload.get("vit"))
        aligner_cfg = parse_target("aligner", payload.get("aligner"))
        deepstack_cfg = parse_target("deepstack", payload.get("deepstack"))

        if not (vit_cfg.enabled or aligner_cfg.enabled or deepstack_cfg.enabled):
            raise ValueError(
                "custom.visual_kd.enabled is true but all per-target configs are disabled; enable at least one of vit/aligner/deepstack"
            )

        return cls(
            enabled=True,
            vit=vit_cfg,
            aligner=aligner_cfg,
            deepstack=deepstack_cfg,
        )


@dataclass(frozen=True)
class GrpoChordConfig:
    enabled: bool = False
    sft_per_device_train_batch_size: int | None = None
    mu_warmup_steps: int | None = None
    mu_decay_steps: int | None = None
    mu_peak: float | None = None
    mu_valley: float | None = None
    enable_phi_function: bool = False

    @classmethod
    def disabled(cls) -> "GrpoChordConfig":
        return cls(enabled=False)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "GrpoChordConfig":
        if payload is None:
            return cls.disabled()
        if not isinstance(payload, Mapping):
            raise TypeError("custom.grpo.chord must be a mapping when provided")

        enabled = bool(payload.get("enabled", False))
        if not enabled:
            return cls.disabled()

        def _require_int(name: str, *, min_value: int = 0) -> int:
            raw = payload.get(name)
            if raw is None:
                raise ValueError(f"custom.grpo.chord.{name} must be provided when enabled")
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"custom.grpo.chord.{name} must be an integer, got {raw!r}"
                ) from exc
            if value < min_value:
                raise ValueError(
                    f"custom.grpo.chord.{name} must be >= {min_value}, got {value}"
                )
            return value

        def _require_float(
            name: str, *, min_value: float = 0.0, max_value: float = 1.0
        ) -> float:
            raw = payload.get(name)
            if raw is None:
                raise ValueError(f"custom.grpo.chord.{name} must be provided when enabled")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"custom.grpo.chord.{name} must be a float, got {raw!r}"
                ) from exc
            if not (min_value <= value <= max_value):
                raise ValueError(
                    f"custom.grpo.chord.{name} must be in [{min_value}, {max_value}], got {value}"
                )
            return value

        return cls(
            enabled=True,
            sft_per_device_train_batch_size=_require_int(
                "sft_per_device_train_batch_size", min_value=1
            ),
            mu_warmup_steps=_require_int("mu_warmup_steps", min_value=0),
            mu_decay_steps=_require_int("mu_decay_steps", min_value=0),
            mu_peak=_require_float("mu_peak"),
            mu_valley=_require_float("mu_valley"),
            enable_phi_function=bool(payload.get("enable_phi_function", False)),
        )


@dataclass(frozen=True)
class GrpoConfig:
    chord: GrpoChordConfig = field(default_factory=GrpoChordConfig.disabled)
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "GrpoConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("custom.grpo must be a mapping when provided")
        data = dict(payload)
        chord = GrpoChordConfig.from_mapping(data.pop("chord", None))
        return cls(chord=chord, extra=dict(data))


@dataclass(frozen=True)
class CustomConfig:
    train_jsonl: str
    user_prompt: str
    emit_norm: AllowedNorm
    json_format: AllowedJsonFormat
    assistant_prefix_format: str | None = None
    use_summary: bool = False
    system_prompt_summary: str | None = None
    augmentation: Mapping[str, Any] | None = None
    augmentation_curriculum: Mapping[str, Any] | None = None
    bypass_prob: float = 0.0
    trainer_variant: str | None = None
    sample_limit: int | None = None
    train_sample_limit: int | None = None
    val_sample_limit: int | None = None
    dump_conversation_text: bool = False
    dump_conversation_path: str | None = None
    val_jsonl: str | None = None
    output_variant: Literal["dense", "summary"] = "dense"
    visual_kd: VisualKDConfig = field(default_factory=VisualKDConfig.disabled)
    token_type_metrics: TokenTypeMetricsConfig = field(default_factory=TokenTypeMetricsConfig)
    extra: Mapping[str, Any] = field(default_factory=dict)
    fusion_config: str | None = None
    grpo: GrpoConfig = field(default_factory=GrpoConfig)

    def __post_init__(self) -> None:
        if not self.train_jsonl:
            raise ValueError("custom.train_jsonl must be provided")
        if not self.user_prompt:
            raise ValueError("custom.user_prompt must be provided")
        if self.emit_norm not in {"none", "norm100", "norm1000"}:
            raise ValueError(
                "custom.emit_norm must be one of {'none', 'norm100', 'norm1000'}"
            )
        if not isinstance(self.use_summary, bool):
            raise TypeError("custom.use_summary must be a boolean value")
        if self.json_format not in ALLOWED_JSON_FORMATS:
            raise ValueError("custom.json_format must be 'standard'")
        if self.assistant_prefix_format is not None:
            fmt = self.assistant_prefix_format.strip()
            if not fmt:
                raise ValueError(
                    "custom.assistant_prefix_format must be a non-empty string"
                )
            if "\n" in fmt or "\r" in fmt:
                raise ValueError(
                    "custom.assistant_prefix_format must be a single-line string"
                )
            if "{domain}" not in fmt or "{task}" not in fmt:
                raise ValueError(
                    "custom.assistant_prefix_format must include '{domain}' and '{task}' placeholders"
                )

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any] | None, *, prompts: PromptOverrides
    ) -> "CustomConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("custom section must be a mapping")

        data: MutableMapping[str, Any] = dict(payload)
        if "use_legacy_fusion" in data:
            raise ValueError(
                "custom.use_legacy_fusion has been removed; unified fusion loader is the only supported path."
            )
        if "grpo_chord" in data:
            raise ValueError(
                "custom.grpo_chord has been removed; use custom.grpo.chord instead."
            )

        train_jsonl = data.pop("train_jsonl", data.pop("jsonl", None))
        user_prompt = data.pop("user_prompt", prompts.user)
        emit_norm = data.pop("emit_norm", None)

        if isinstance(user_prompt, str) and user_prompt.endswith(".txt"):
            path = Path(user_prompt)
            if path.is_file():
                user_prompt = path.read_text(encoding="utf-8").strip("\n")

        if "summary_ratio" in data:
            raise ValueError(
                "custom.summary_ratio has been removed; use custom.use_summary instead."
            )

        def _parse_bool(value: object, field_name: str) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                if value in (0, 1, 0.0, 1.0):
                    return bool(value)
                raise ValueError(
                    f"{field_name} must be boolean (0 or 1), got {value!r}."
                )
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "y", "on"}:
                    return True
                if normalized in {"false", "0", "no", "n", "off"}:
                    return False
                raise ValueError(
                    f"{field_name} string value '{value}' is not a recognized boolean representation."
                )
            raise TypeError(
                f"{field_name} must be a boolean value, got {type(value)!r}."
            )

        use_summary_raw = data.pop("use_summary", None)
        use_summary = (
            False
            if use_summary_raw is None
            else _parse_bool(use_summary_raw, "custom.use_summary")
        )
        if "summary_label_grouping" in data:
            raise ValueError(
                "custom.summary_label_grouping has been removed; delete the field to continue."
            )
        system_prompt_summary = data.pop("system_prompt_summary", None)
        if "images_per_user_turn" in data:
            raise ValueError(
                "custom.images_per_user_turn is no longer supported; remove the field to use single-image turns."
            )
        augmentation = data.pop("augmentation", None)
        augmentation_curriculum = data.pop("augmentation_curriculum", None)
        bypass_prob = float(data.pop("bypass_prob", 0.0))
        trainer_variant = data.pop("trainer_variant", None)
        def _parse_sample_limit(value: Any, field_name: str) -> int | None:
            if value is None:
                return None
            if isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer when provided")
            if isinstance(value, (int, float)):
                if isinstance(value, float) and not value.is_integer():
                    raise ValueError(
                        f"{field_name} must be an integer, got {value!r}"
                    )
                int_value = int(value)
                return int_value if int_value > 0 else None
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.isdigit():
                    int_value = int(stripped)
                    return int_value if int_value > 0 else None
                raise ValueError(
                    f"{field_name} must be an integer or numeric string, got {value!r}"
                )
            raise TypeError(f"{field_name} must be an integer when provided")

        sample_limit = _parse_sample_limit(
            data.pop("sample_limit", None), "custom.sample_limit"
        )
        train_sample_limit = _parse_sample_limit(
            data.pop("train_sample_limit", None), "custom.train_sample_limit"
        )
        val_sample_limit = _parse_sample_limit(
            data.pop("val_sample_limit", None), "custom.val_sample_limit"
        )
        dump_conversation_text = bool(data.pop("dump_conversation_text", False))
        dump_conversation_path = data.pop("dump_conversation_path", None)
        val_jsonl = data.pop("val_jsonl", None)
        fusion_config = data.pop("fusion_config", None)
        grpo_raw = data.pop("grpo", None)
        grpo = GrpoConfig.from_mapping(grpo_raw)
        visual_kd_raw = data.pop("visual_kd", None)
        visual_kd = VisualKDConfig.from_mapping(visual_kd_raw)
        token_type_metrics_raw = data.pop("token_type_metrics", None)
        token_type_metrics = TokenTypeMetricsConfig.from_mapping(token_type_metrics_raw)
        hsm_raw = data.pop("hard_sample_mining", None)
        if hsm_raw is not None:
            raise ValueError(
                "custom.hard_sample_mining has been removed; delete this block to continue with standard SFT."
            )
        json_format_raw = data.pop("json_format", None)
        if json_format_raw is None:
            raise ValueError("custom.json_format must be provided")
        json_format = _normalize_json_format(json_format_raw)
        assistant_prefix_format_raw = data.pop("assistant_prefix_format", None)
        assistant_prefix_format = None
        if assistant_prefix_format_raw is not None:
            if not isinstance(assistant_prefix_format_raw, str):
                raise TypeError(
                    "custom.assistant_prefix_format must be a string if provided"
                )
            assistant_prefix_format = assistant_prefix_format_raw.strip()

        extra = dict(data)

        if emit_norm is None:
            raise ValueError("custom.emit_norm must be provided")
        if not isinstance(emit_norm, str):
            raise TypeError("custom.emit_norm must be a string")
        emit_norm_value = emit_norm.strip()
        if emit_norm_value not in {"none", "norm100", "norm1000"}:
            raise ValueError(
                "custom.emit_norm must be one of {'none', 'norm100', 'norm1000'}"
            )

        return cls(
            train_jsonl=str(train_jsonl) if train_jsonl is not None else "",
            user_prompt=str(user_prompt) if user_prompt is not None else "",
            emit_norm=cast("AllowedNorm", emit_norm_value),
            json_format=json_format,
            assistant_prefix_format=assistant_prefix_format,
            use_summary=use_summary,
            system_prompt_summary=system_prompt_summary,
            augmentation=augmentation
            if isinstance(augmentation, Mapping)
            else augmentation,
            augmentation_curriculum=augmentation_curriculum
            if isinstance(augmentation_curriculum, Mapping)
            else augmentation_curriculum,
            bypass_prob=bypass_prob,
            trainer_variant=str(trainer_variant)
            if trainer_variant is not None
            else None,
            sample_limit=sample_limit,
            train_sample_limit=train_sample_limit,
            val_sample_limit=val_sample_limit,
            dump_conversation_text=dump_conversation_text,
            dump_conversation_path=str(dump_conversation_path)
            if dump_conversation_path is not None
            else None,
            val_jsonl=str(val_jsonl) if val_jsonl is not None else None,
            fusion_config=str(fusion_config) if fusion_config is not None else None,
            output_variant=prompts.output_variant,
            visual_kd=visual_kd,
            token_type_metrics=token_type_metrics,
            extra=extra,
            grpo=grpo,
        )


@dataclass(frozen=True)
class TrainingConfig:
    template: Mapping[str, Any]
    custom: CustomConfig
    model: Mapping[str, Any] = field(default_factory=dict)
    quantization: Mapping[str, Any] = field(default_factory=dict)
    data: Mapping[str, Any] = field(default_factory=dict)
    tuner: Mapping[str, Any] = field(default_factory=dict)
    training: Mapping[str, Any] = field(default_factory=dict)
    rlhf: Mapping[str, Any] = field(default_factory=dict)
    prompts: PromptOverrides = field(default_factory=PromptOverrides)
    deepspeed: DeepSpeedConfig | None = None
    global_max_length: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], prompts: PromptOverrides
    ) -> "TrainingConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("config payload must be a mapping")

        data = dict(payload)

        model = dict(_as_dict(data.pop("model", None)))
        quantization = dict(_as_dict(data.pop("quantization", None)))
        template_raw = _as_dict(data.pop("template", None))
        template = dict(template_raw)
        data_section = dict(_as_dict(data.pop("data", None)))
        tuner = dict(_as_dict(data.pop("tuner", None)))
        training = dict(_as_dict(data.pop("training", None)))
        rlhf = dict(_as_dict(data.pop("rlhf", None)))
        custom_raw = data.pop("custom", None)
        deepspeed = DeepSpeedConfig.from_mapping(data.pop("deepspeed", None))
        global_max_length = data.pop("global_max_length", None)

        extra = dict(data)

        if global_max_length is not None:
            if not isinstance(global_max_length, int) or global_max_length <= 0:
                raise ValueError(
                    "global_max_length must be a positive integer when provided"
                )

        if not template:
            raise ValueError("template section must be provided in the config")

        if prompts.system and "system" not in template:
            template["system"] = prompts.system

        custom = CustomConfig.from_mapping(custom_raw, prompts=prompts)

        return cls(
            template=template,
            custom=custom,
            model=model,
            quantization=quantization,
            data=data_section,
            tuner=tuner,
            training=training,
            rlhf=rlhf,
            prompts=prompts,
            deepspeed=deepspeed,
            global_max_length=global_max_length,
            extra=extra,
        )
