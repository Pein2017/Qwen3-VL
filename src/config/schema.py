"""Typed configuration schemas for training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

AllowedNorm = Literal["none", "norm100", "norm1000"]
AllowedVisualDistance = Literal["mse", "cosine"]


def _as_dict(value: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section must be a mapping, got {type(value)!r}")
    return value


@dataclass(frozen=True)
class PromptOverrides:
    system: Optional[str] = None
    user: Optional[str] = None
    output_variant: Literal["dense", "summary"] = "dense"


@dataclass(frozen=True)
class DeepSpeedConfig:
    enabled: bool
    config: Any

    @classmethod
    def from_mapping(
        cls, payload: Optional[Mapping[str, Any]]
    ) -> Optional["DeepSpeedConfig"]:
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
    steps: Optional[int] = None
    epochs: Optional[float] = None

    @classmethod
    def from_raw(cls, steps: Any, epochs: Any) -> "SaveDelayConfig":
        parsed_steps: Optional[int] = None
        if steps is not None:
            try:
                value = int(steps)
            except (TypeError, ValueError) as exc:
                raise ValueError("save_delay_steps must be an integer") from exc
            if value > 0:
                parsed_steps = value

        parsed_epochs: Optional[float] = None
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
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "SaveDelayConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("save_delay section must be a mapping")
        steps = payload.get("steps")
        epochs = payload.get("epochs")
        return cls.from_raw(steps, epochs)


@dataclass(frozen=True)
class VisualKDConfig:
    enabled: bool
    weight: float = 0.0
    targets: Tuple[str, ...] = field(default_factory=tuple)
    distance: AllowedVisualDistance = "mse"

    def __post_init__(self) -> None:
        if self.enabled and self.weight <= 0:
            raise ValueError("visual_kd.weight must be > 0 when enabled")
        if self.enabled and not self.targets:
            raise ValueError("visual_kd.targets must be non-empty when enabled")
        if self.distance not in {"mse", "cosine"}:
            raise ValueError("visual_kd.distance must be one of {mse, cosine}")

    @classmethod
    def disabled(cls) -> "VisualKDConfig":
        return cls(enabled=False, weight=0.0, targets=tuple(), distance="mse")

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "VisualKDConfig":
        if payload is None:
            return cls.disabled()
        if not isinstance(payload, Mapping):
            raise TypeError("custom.visual_kd must be a mapping when provided")

        enabled = bool(payload.get("enabled", False))
        raw_weight = payload.get("weight", 0.0)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("custom.visual_kd.weight must be numeric") from exc

        raw_targets = payload.get("targets", [])
        if isinstance(raw_targets, str):
            targets_iter: Sequence[str] = [raw_targets]
        elif isinstance(raw_targets, Sequence):
            targets_iter = [str(t) for t in raw_targets]
        else:
            raise TypeError("custom.visual_kd.targets must be a sequence or string")

        dedup: list[str] = []
        seen: set[str] = set()
        for item in targets_iter:
            lowered = item.lower()
            if lowered not in {"merger", "deepstack"}:
                raise ValueError(
                    "custom.visual_kd.targets contains unsupported entries: " + lowered
                )
            if lowered not in seen:
                seen.add(lowered)
                dedup.append(lowered)

        raw_distance = payload.get("distance", "mse")
        if not isinstance(raw_distance, str):
            raise TypeError("custom.visual_kd.distance must be a string")
        distance = raw_distance.lower()

        if not enabled:
            return cls.disabled()

        return cls(enabled=True, weight=weight, targets=tuple(dedup), distance=distance)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CustomConfig:
    train_jsonl: str
    user_prompt: str
    emit_norm: AllowedNorm
    summary_ratio: Optional[float] = None
    system_prompt_summary: Optional[str] = None
    augmentation: Optional[Mapping[str, Any]] = None
    bypass_prob: float = 0.0
    trainer_variant: Optional[str] = None
    sample_limit: Optional[Any] = None
    train_sample_limit: Optional[Any] = None
    val_sample_limit: Optional[Any] = None
    dump_conversation_text: bool = False
    dump_conversation_path: Optional[str] = None
    val_jsonl: Optional[str] = None
    output_variant: Literal["dense", "summary"] = "dense"
    visual_kd: VisualKDConfig = field(default_factory=VisualKDConfig.disabled)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.train_jsonl:
            raise ValueError("custom.train_jsonl must be provided")
        if not self.user_prompt:
            raise ValueError("custom.user_prompt must be provided")
        if self.emit_norm not in {"none", "norm100", "norm1000"}:
            raise ValueError(
                "custom.emit_norm must be one of {'none', 'norm100', 'norm1000'}"
            )
        if self.summary_ratio is not None:
            try:
                ratio = float(self.summary_ratio)
            except (TypeError, ValueError) as exc:
                raise ValueError("custom.summary_ratio must be numeric") from exc
            if not 0.0 <= ratio <= 1.0:
                raise ValueError("custom.summary_ratio must be within [0, 1]")

    @classmethod
    def from_mapping(
        cls, payload: Optional[Mapping[str, Any]], *, prompts: PromptOverrides
    ) -> "CustomConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("custom section must be a mapping")

        data: MutableMapping[str, Any] = dict(payload)

        train_jsonl = data.pop("train_jsonl", data.pop("jsonl", None))
        user_prompt = data.pop("user_prompt", prompts.user)
        emit_norm = data.pop("emit_norm", None)

        if isinstance(user_prompt, str) and user_prompt.endswith(".txt"):
            path = Path(user_prompt)
            if path.is_file():
                user_prompt = path.read_text(encoding="utf-8").strip("\n")

        summary_ratio = data.pop("summary_ratio", None)
        system_prompt_summary = data.pop("system_prompt_summary", None)
        if "images_per_user_turn" in data:
            raise ValueError(
                "custom.images_per_user_turn is no longer supported; remove the field to use single-image turns."
            )
        augmentation = data.pop("augmentation", None)
        bypass_prob = float(data.pop("bypass_prob", 0.0))
        trainer_variant = data.pop("trainer_variant", None)
        sample_limit = data.pop("sample_limit", None)
        train_sample_limit = data.pop("train_sample_limit", None)
        val_sample_limit = data.pop("val_sample_limit", None)
        dump_conversation_text = bool(data.pop("dump_conversation_text", False))
        dump_conversation_path = data.pop("dump_conversation_path", None)
        val_jsonl = data.pop("val_jsonl", None)
        visual_kd_raw = data.pop("visual_kd", None)
        visual_kd = VisualKDConfig.from_mapping(visual_kd_raw)

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
            summary_ratio=summary_ratio,
            system_prompt_summary=system_prompt_summary,
            augmentation=augmentation
            if isinstance(augmentation, Mapping)
            else augmentation,
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
            output_variant=prompts.output_variant,
            visual_kd=visual_kd,
            extra=extra,
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
    deepspeed: Optional[DeepSpeedConfig] = None
    global_max_length: Optional[int] = None
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
