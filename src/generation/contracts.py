#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured contracts for the centralized generation engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from src.utils.unstructured import require_mapping, require_mapping_sequence


@dataclass(frozen=True)
class StopOptions:
    """Stop-policy options for generation.

    stop: stop strings that should terminate decoded output.
    stop_token_ids: explicit token ids that should terminate generation.
    """

    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for idx, item in enumerate(self.stop):
            if not isinstance(item, str):
                raise TypeError(f"stop.stop[{idx}] must be a string")
        for idx, item in enumerate(self.stop_token_ids):
            if not isinstance(item, int):
                raise TypeError(f"stop.stop_token_ids[{idx}] must be an int")
            if item < 0:
                raise ValueError(f"stop.stop_token_ids[{idx}] must be >= 0")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, context: str) -> "StopOptions":
        raw = require_mapping(raw, context=context)
        stop_raw = raw.get("stop", ())
        stop_token_ids_raw = raw.get("stop_token_ids", ())
        if stop_raw is None:
            stop_raw = ()
        if stop_token_ids_raw is None:
            stop_token_ids_raw = ()
        if not isinstance(stop_raw, Sequence) or isinstance(stop_raw, (str, bytes)):
            raise TypeError(f"{context}.stop must be a sequence of strings")
        stop: list[str] = []
        for idx, item in enumerate(stop_raw):
            if not isinstance(item, str):
                raise TypeError(f"{context}.stop[{idx}] must be a string")
            if item:
                stop.append(item)
        if not isinstance(stop_token_ids_raw, Sequence) or isinstance(
            stop_token_ids_raw, (str, bytes)
        ):
            raise TypeError(f"{context}.stop_token_ids must be a sequence of ints")
        stop_token_ids: list[int] = []
        for idx, item in enumerate(stop_token_ids_raw):
            if not isinstance(item, int):
                raise TypeError(f"{context}.stop_token_ids[{idx}] must be an int")
            if item < 0:
                raise ValueError(f"{context}.stop_token_ids[{idx}] must be >= 0")
            stop_token_ids.append(int(item))
        return cls(stop=tuple(stop), stop_token_ids=tuple(stop_token_ids))


@dataclass(frozen=True)
class DecodeOptions:
    """Decode preferences for GenerationResult text slots."""

    skip_special_tokens: bool = True
    clean_up_tokenization_spaces: bool = False
    strip_whitespace: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.skip_special_tokens, bool):
            raise TypeError("decode.skip_special_tokens must be a boolean")
        if not isinstance(self.clean_up_tokenization_spaces, bool):
            raise TypeError("decode.clean_up_tokenization_spaces must be a boolean")
        if not isinstance(self.strip_whitespace, bool):
            raise TypeError("decode.strip_whitespace must be a boolean")


@dataclass(frozen=True)
class ChatTemplateOptions:
    """Options for chat-template rendering."""

    add_generation_prompt: bool = True
    tokenize: bool = False
    enable_thinking: bool = False
    continue_final_message: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.add_generation_prompt, bool):
            raise TypeError("chat_template.add_generation_prompt must be a boolean")
        if not isinstance(self.tokenize, bool):
            raise TypeError("chat_template.tokenize must be a boolean")
        if not isinstance(self.enable_thinking, bool):
            raise TypeError("chat_template.enable_thinking must be a boolean")
        if not isinstance(self.continue_final_message, bool):
            raise TypeError("chat_template.continue_final_message must be a boolean")


@dataclass(frozen=True)
class VlmPreprocessOptions:
    """Preprocessing options for VLM processors/tokenizers."""

    max_pixels: int | None = None
    min_pixels: int | None = None
    do_resize: bool | None = None
    padding_side: str = "left"
    truncation_side: str = "left"
    pad_token_fallback: bool = True

    def __post_init__(self) -> None:
        if self.max_pixels is not None and self.max_pixels <= 0:
            raise ValueError("vlm_preprocess.max_pixels must be > 0")
        if self.min_pixels is not None and self.min_pixels <= 0:
            raise ValueError("vlm_preprocess.min_pixels must be > 0")
        if self.do_resize is not None and not isinstance(self.do_resize, bool):
            raise TypeError("vlm_preprocess.do_resize must be a boolean")
        if not isinstance(self.padding_side, str):
            raise TypeError("vlm_preprocess.padding_side must be a string")
        if not isinstance(self.truncation_side, str):
            raise TypeError("vlm_preprocess.truncation_side must be a string")
        if not isinstance(self.pad_token_fallback, bool):
            raise TypeError("vlm_preprocess.pad_token_fallback must be a boolean")


@dataclass(frozen=True)
class ModelLoadConfig:
    """Model loading configuration for generation backends."""

    model_name_or_path: str
    torch_dtype: str | None = None
    device: str | None = None
    device_map: object | None = None
    attn_implementation: str | None = None
    trust_remote_code: bool = True
    variant: str | None = None  # "vlm" or "text"; None => auto-detect

    def __post_init__(self) -> None:
        if not isinstance(self.model_name_or_path, str) or not self.model_name_or_path:
            raise TypeError("model.model_name_or_path must be a non-empty string")
        if self.torch_dtype is not None and not isinstance(self.torch_dtype, str):
            raise TypeError("model.torch_dtype must be a string")
        if self.device is not None and not isinstance(self.device, str):
            raise TypeError("model.device must be a string")
        if self.attn_implementation is not None and not isinstance(
            self.attn_implementation, str
        ):
            raise TypeError("model.attn_implementation must be a string")
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("model.trust_remote_code must be a boolean")
        if self.variant is not None:
            variant = str(self.variant).lower()
            if variant not in {"vlm", "text"}:
                raise ValueError("model.variant must be 'vlm' or 'text'")


@dataclass(frozen=True)
class VllmEngineOptions:
    """vLLM-specific engine configuration."""

    tensor_parallel_size: int = 1
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    trust_remote_code: bool = True
    dtype: str | None = None
    enforce_eager: bool | None = None

    def __post_init__(self) -> None:
        if self.tensor_parallel_size <= 0:
            raise ValueError("vllm.tensor_parallel_size must be > 0")
        if self.gpu_memory_utilization is not None and not (
            0 < self.gpu_memory_utilization <= 1
        ):
            raise ValueError("vllm.gpu_memory_utilization must be in (0, 1]")
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise ValueError("vllm.max_model_len must be > 0")
        if self.dtype is not None and not isinstance(self.dtype, str):
            raise TypeError("vllm.dtype must be a string")
        if self.enforce_eager is not None and not isinstance(self.enforce_eager, bool):
            raise TypeError("vllm.enforce_eager must be a boolean")


@dataclass(frozen=True)
class GenerationOptions:
    """Generation-time options shared across backends."""

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    do_sample: bool | None = None
    no_repeat_ngram_size: int | None = None
    stop: StopOptions = field(default_factory=StopOptions)
    seed: int | None = None
    use_cache: bool = True
    eos_token_id: int | Sequence[int] | None = None
    pad_token_id: int | None = None
    decode: DecodeOptions = field(default_factory=DecodeOptions)
    extra_generation_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens <= 0:
            raise ValueError("generation.max_new_tokens must be > 0")
        if not isinstance(self.temperature, (int, float)):
            raise TypeError("generation.temperature must be a number")
        if not isinstance(self.top_p, (int, float)):
            raise TypeError("generation.top_p must be a number")
        if not isinstance(self.repetition_penalty, (int, float)):
            raise TypeError("generation.repetition_penalty must be a number")
        if self.do_sample is not None and not isinstance(self.do_sample, bool):
            raise TypeError("generation.do_sample must be a boolean")
        if self.no_repeat_ngram_size is not None:
            if not isinstance(self.no_repeat_ngram_size, int):
                raise TypeError("generation.no_repeat_ngram_size must be an int")
            if self.no_repeat_ngram_size < 0:
                raise ValueError("generation.no_repeat_ngram_size must be >= 0")
        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError("generation.seed must be an int")
        if not isinstance(self.use_cache, bool):
            raise TypeError("generation.use_cache must be a boolean")
        if self.pad_token_id is not None and not isinstance(self.pad_token_id, int):
            raise TypeError("generation.pad_token_id must be an int")
        if not isinstance(self.extra_generation_kwargs, Mapping):
            raise TypeError("generation.extra_generation_kwargs must be a mapping")

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, object], *, context: str
    ) -> "GenerationOptions":
        raw = require_mapping(raw, context=context)
        stop_raw = raw.get("stop")
        stop_token_ids_raw = raw.get("stop_token_ids")
        stop = StopOptions.from_mapping(
            {"stop": stop_raw, "stop_token_ids": stop_token_ids_raw},
            context=f"{context}.stop",
        )
        extra: dict[str, object] = {
            key: value
            for key, value in raw.items()
            if key
            not in {
                "max_new_tokens",
                "temperature",
                "top_p",
                "repetition_penalty",
                "do_sample",
                "no_repeat_ngram_size",
                "seed",
                "use_cache",
                "eos_token_id",
                "pad_token_id",
                "stop",
                "stop_token_ids",
                "decode",
            }
        }
        decode_raw = raw.get("decode")
        decode = DecodeOptions()
        if decode_raw is not None:
            decode_map = require_mapping(decode_raw, context=f"{context}.decode")
            decode = DecodeOptions(
                skip_special_tokens=bool(
                    decode_map.get("skip_special_tokens", decode.skip_special_tokens)
                ),
                clean_up_tokenization_spaces=bool(
                    decode_map.get(
                        "clean_up_tokenization_spaces",
                        decode.clean_up_tokenization_spaces,
                    )
                ),
                strip_whitespace=bool(
                    decode_map.get("strip_whitespace", decode.strip_whitespace)
                ),
            )
        return cls(
            max_new_tokens=int(raw.get("max_new_tokens", cls.max_new_tokens)),
            temperature=float(raw.get("temperature", cls.temperature)),
            top_p=float(raw.get("top_p", cls.top_p)),
            repetition_penalty=float(
                raw.get("repetition_penalty", cls.repetition_penalty)
            ),
            do_sample=raw.get("do_sample"),
            no_repeat_ngram_size=raw.get("no_repeat_ngram_size"),
            stop=stop,
            seed=raw.get("seed"),
            use_cache=bool(raw.get("use_cache", cls.use_cache)),
            eos_token_id=raw.get("eos_token_id"),
            pad_token_id=raw.get("pad_token_id"),
            decode=decode,
            extra_generation_kwargs=extra,
        )


@dataclass(frozen=True)
class TextGenerationRequest:
    """Text-only request using raw chat-template messages.

    messages are intentionally unstructured mappings expected by the
    tokenizer chat-template APIs. They are validated as a sequence of mappings
    to comply with the Schema Constitution's escape hatch rules.
    """

    messages: Sequence[Mapping[str, object]]

    def __post_init__(self) -> None:
        require_mapping_sequence(self.messages, context="text_request.messages")


@dataclass(frozen=True)
class VlmGenerationRequest:
    """VLM request using raw chat-template messages (must include one image)."""

    messages: Sequence[Mapping[str, object]]
    verify: bool = False

    def __post_init__(self) -> None:
        require_mapping_sequence(self.messages, context="vlm_request.messages")
        if not isinstance(self.verify, bool):
            raise TypeError("vlm_request.verify must be a boolean")


@dataclass(frozen=True)
class GenerationResult:
    """Decoded generation result."""

    text: str
    raw_text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("result.text must be a string")
        if not isinstance(self.raw_text, str):
            raise TypeError("result.raw_text must be a string")
        if self.prompt_tokens is not None and not isinstance(
            self.prompt_tokens, int
        ):
            raise TypeError("result.prompt_tokens must be an int")
        if self.completion_tokens is not None and not isinstance(
            self.completion_tokens, int
        ):
            raise TypeError("result.completion_tokens must be an int")
        if self.total_tokens is not None and not isinstance(self.total_tokens, int):
            raise TypeError("result.total_tokens must be an int")


__all__ = [
    "ChatTemplateOptions",
    "DecodeOptions",
    "GenerationOptions",
    "GenerationResult",
    "ModelLoadConfig",
    "StopOptions",
    "TextGenerationRequest",
    "VlmGenerationRequest",
    "VlmPreprocessOptions",
    "VllmEngineOptions",
]
