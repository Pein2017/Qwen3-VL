#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transformers-based generation backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen3VLForConditionalGeneration,
)

from src.utils import get_logger

from ..chat_template import render_chat_template
from ..contracts import (
    ChatTemplateOptions,
    GenerationOptions,
    GenerationResult,
    ModelLoadConfig,
    TextGenerationRequest,
    VlmGenerationRequest,
    VlmPreprocessOptions,
)
from ..preprocess import (
    build_image_kwargs,
    configure_vlm_processor,
    normalize_tokenizer,
)
from ..stop_policy import merge_eos_token_ids, normalize_stop_options, truncate_text_at_stops

logger = get_logger(__name__)


def _dtype_from_str(name: str) -> torch.dtype:
    lowered = name.lower()
    if not hasattr(torch, lowered):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return getattr(torch, lowered)


def _detect_variant(model_path: str) -> str:
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read model config at {model_path}: {exc}") from exc
    model_type = getattr(cfg, "model_type", "") or ""
    has_vision = getattr(cfg, "vision_config", None) is not None
    if "vl" in str(model_type).lower() or has_vision:
        return "vlm"
    return "text"


def _extract_model_inputs(payload: Mapping[str, object]) -> dict[str, torch.Tensor]:
    model_inputs: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        value = payload.get(key)
        if value is not None:
            model_inputs[key] = cast(torch.Tensor, value)
    return model_inputs


def _trim_trailing_eos_pad(
    token_ids: torch.Tensor,
    *,
    eos_id: int | None,
    pad_id: int | None,
) -> torch.Tensor:
    if token_ids.numel() == 0:
        return token_ids
    eos_set: set[int] = set()
    if eos_id is not None:
        eos_set.add(int(eos_id))
    if pad_id is not None:
        eos_set.add(int(pad_id))
    if not eos_set:
        return token_ids
    ids = token_ids
    while ids.numel() > 0 and int(ids[-1]) in eos_set:
        ids = ids[:-1]
    return ids


def _decode_with_fallback(
    tokenizer: PreTrainedTokenizerBase,
    processor: object | None,
    token_ids: torch.Tensor,
    *,
    skip_special_tokens: bool,
    clean_up_tokenization_spaces: bool,
) -> str:
    if processor is not None:
        try:
            decode = getattr(processor, "batch_decode", None)
            if callable(decode):
                return decode(
                    token_ids.unsqueeze(0),
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )[0]
        except Exception:
            pass
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )


def _collect_prompt_lengths(encoded: BatchEncoding) -> list[int]:
    attention_mask = encoded.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        return [int(val) for val in attention_mask.sum(dim=1).tolist()]
    input_ids = encoded.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        return [int((row != 0).sum().item()) for row in input_ids]
    return []


def _merge_hf_plugins(plugins: Sequence[object] | None, prompt_lengths: Sequence[int]):
    stopping = None
    logits = None
    if not plugins:
        return stopping, logits
    for plugin in plugins:
        stopper = getattr(plugin, "hf_stopping_criteria", None)
        if callable(stopper):
            candidate = stopper(prompt_lengths)
            if candidate is not None:
                if stopping is None:
                    stopping = candidate
                else:
                    stopping.extend(candidate)
        processor = getattr(plugin, "hf_logits_processor", None)
        if callable(processor):
            candidate = processor(prompt_lengths)
            if candidate is not None:
                if logits is None:
                    logits = candidate
                else:
                    logits.extend(candidate)
    return stopping, logits


@dataclass
class HfBackend:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    processor: object | None
    chat_template: ChatTemplateOptions
    supports_vlm: bool

    @property
    def name(self) -> str:
        return "hf"

    @property
    def device(self) -> torch.device | str:
        return self.model.device if hasattr(self.model, "device") else "cpu"

    def generate_text_batch(
        self,
        requests: Sequence[TextGenerationRequest],
        options: GenerationOptions,
        *,
        plugins: Sequence[object] | None = None,
    ) -> list[GenerationResult]:
        if not requests:
            return []
        prompts = [
            render_chat_template(self.tokenizer, req.messages, options=self.chat_template)
            for req in requests
        ]
        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_len = int(encoded["input_ids"].shape[1])
        prompt_lengths = _collect_prompt_lengths(encoded)
        inputs = {k: v.to(self.device) for k, v in encoded.items()}

        stop_token_ids, stop_strings = normalize_stop_options(
            options.stop, tokenizer=self.tokenizer
        )
        eos_ids = merge_eos_token_ids(
            options.eos_token_id or self.tokenizer.eos_token_id, stop_token_ids
        )
        pad_token_id = (
            options.pad_token_id
            if options.pad_token_id is not None
            else self.tokenizer.pad_token_id
        )
        if pad_token_id is None and eos_ids:
            pad_token_id = eos_ids[0]

        do_sample = (
            options.do_sample
            if options.do_sample is not None
            else options.temperature > 0
        )
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": options.max_new_tokens,
            "temperature": options.temperature if do_sample else None,
            "top_p": options.top_p,
            "repetition_penalty": options.repetition_penalty,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_ids or None,
            "use_cache": options.use_cache,
        }
        if options.no_repeat_ngram_size is not None:
            generate_kwargs["no_repeat_ngram_size"] = options.no_repeat_ngram_size
        if options.seed is not None:
            torch.manual_seed(int(options.seed))
        generate_kwargs.update(options.extra_generation_kwargs)

        stopping, logits = _merge_hf_plugins(plugins, prompt_lengths)
        if stopping is not None:
            generate_kwargs["stopping_criteria"] = stopping
        if logits is not None:
            generate_kwargs["logits_processor"] = logits

        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

        with torch.inference_mode():
            output = cast(Any, self.model).generate(**inputs, **generate_kwargs)

        sequences = output if isinstance(output, torch.Tensor) else output.sequences
        sequences = sequences.to("cpu")
        results: list[GenerationResult] = []
        for idx in range(sequences.size(0)):
            generated_ids_full = sequences[idx, input_len:]
            generated_ids = _trim_trailing_eos_pad(
                generated_ids_full,
                eos_id=self.tokenizer.eos_token_id,
                pad_id=pad_token_id,
            )
            raw_text = _decode_with_fallback(
                self.tokenizer,
                self.processor,
                generated_ids_full,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=options.decode.clean_up_tokenization_spaces,
            )
            clean_text = _decode_with_fallback(
                self.tokenizer,
                self.processor,
                generated_ids,
                skip_special_tokens=options.decode.skip_special_tokens,
                clean_up_tokenization_spaces=options.decode.clean_up_tokenization_spaces,
            )
            clean_text = truncate_text_at_stops(clean_text, stop_strings)
            if options.decode.strip_whitespace:
                raw_text = raw_text.strip()
                clean_text = clean_text.strip()
            if plugins:
                for plugin in plugins:
                    post = getattr(plugin, "postprocess_text", None)
                    if callable(post):
                        clean_text = post(clean_text)
            prompt_tokens = prompt_lengths[idx] if idx < len(prompt_lengths) else None
            completion_tokens = int(generated_ids_full.numel())
            total_tokens = (
                prompt_tokens + completion_tokens
                if prompt_tokens is not None
                else None
            )
            results.append(
                GenerationResult(
                    text=clean_text,
                    raw_text=raw_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            )
        return results

    def generate_vlm_batch(
        self,
        requests: Sequence[VlmGenerationRequest],
        options: GenerationOptions,
        *,
        plugins: Sequence[object] | None = None,
    ) -> list[GenerationResult]:
        if not self.supports_vlm or self.processor is None:
            raise RuntimeError("HF backend does not support VLM generation")
        if not requests:
            return []

        prompts: list[str] = []
        images: list[Any] = []
        for req in requests:
            image = _extract_single_image(req.messages)
            if image is None:
                raise ValueError("VLM request must include exactly one image")
            images.append(image)
            prompts.append(
                render_chat_template(
                    self.processor, req.messages, options=self.chat_template
                )
            )

        img_kwargs = build_image_kwargs(self.processor)
        encoded = cast(
            BatchEncoding,
            self.processor(  # type: ignore[operator]
                images=images,
                text=prompts,
                return_tensors="pt",
                padding=True,
                images_kwargs=img_kwargs,
            ),
        )
        input_len = int(encoded["input_ids"].shape[1])
        prompt_lengths = _collect_prompt_lengths(encoded)
        model_inputs = _extract_model_inputs(encoded)
        if "input_ids" not in model_inputs:
            raise ValueError("processor output missing input_ids")

        if any(req.verify for req in requests):
            _log_vlm_verify(
                requests=requests,
                images=images,
                processor=self.processor,
                model_inputs=model_inputs,
            )

        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        stop_token_ids, stop_strings = normalize_stop_options(
            options.stop, tokenizer=self.tokenizer
        )
        eos_ids = merge_eos_token_ids(
            options.eos_token_id or self.tokenizer.eos_token_id, stop_token_ids
        )
        pad_token_id = (
            options.pad_token_id
            if options.pad_token_id is not None
            else self.tokenizer.pad_token_id
        )
        if pad_token_id is None and eos_ids:
            pad_token_id = eos_ids[0]

        do_sample = (
            options.do_sample
            if options.do_sample is not None
            else options.temperature > 0
        )
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": options.max_new_tokens,
            "temperature": options.temperature if do_sample else None,
            "top_p": options.top_p,
            "repetition_penalty": options.repetition_penalty,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_ids or None,
            "use_cache": options.use_cache,
        }
        if options.no_repeat_ngram_size is not None:
            generate_kwargs["no_repeat_ngram_size"] = options.no_repeat_ngram_size
        if options.seed is not None:
            torch.manual_seed(int(options.seed))
        generate_kwargs.update(options.extra_generation_kwargs)

        stopping, logits = _merge_hf_plugins(plugins, prompt_lengths)
        if stopping is not None:
            generate_kwargs["stopping_criteria"] = stopping
        if logits is not None:
            generate_kwargs["logits_processor"] = logits

        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

        with torch.inference_mode():
            output = cast(Any, self.model).generate(**model_inputs, **generate_kwargs)

        sequences = output if isinstance(output, torch.Tensor) else output.sequences
        sequences = sequences.to("cpu")
        results: list[GenerationResult] = []
        for idx in range(sequences.size(0)):
            generated_ids_full = sequences[idx, input_len:]
            generated_ids = _trim_trailing_eos_pad(
                generated_ids_full,
                eos_id=self.tokenizer.eos_token_id,
                pad_id=pad_token_id,
            )
            raw_text = _decode_with_fallback(
                self.tokenizer,
                self.processor,
                generated_ids_full,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=options.decode.clean_up_tokenization_spaces,
            )
            clean_text = _decode_with_fallback(
                self.tokenizer,
                self.processor,
                generated_ids,
                skip_special_tokens=options.decode.skip_special_tokens,
                clean_up_tokenization_spaces=options.decode.clean_up_tokenization_spaces,
            )
            clean_text = truncate_text_at_stops(clean_text, stop_strings)
            if options.decode.strip_whitespace:
                raw_text = raw_text.strip()
                clean_text = clean_text.strip()
            if plugins:
                for plugin in plugins:
                    post = getattr(plugin, "postprocess_text", None)
                    if callable(post):
                        clean_text = post(clean_text)
            prompt_tokens = prompt_lengths[idx] if idx < len(prompt_lengths) else None
            completion_tokens = int(generated_ids_full.numel())
            total_tokens = (
                prompt_tokens + completion_tokens
                if prompt_tokens is not None
                else None
            )
            results.append(
                GenerationResult(
                    text=clean_text,
                    raw_text=raw_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            )
        return results


def _extract_single_image(messages: Sequence[Mapping[str, object]]):
    found = None
    for message in messages:
        content = message.get("content")
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") == "image":
                    if found is not None:
                        return None
                    found = part.get("image")
    return found


def _log_vlm_verify(
    *,
    requests: Sequence[VlmGenerationRequest],
    images: Sequence[Any],
    processor: object,
    model_inputs: Mapping[str, torch.Tensor],
) -> None:
    input_ids = model_inputs.get("input_ids")
    grid = model_inputs.get("image_grid_thw")
    if input_ids is None:
        return
    try:
        image_token_id = getattr(processor, "image_token_id", None) or getattr(
            getattr(processor, "tokenizer", None), "image_token_id", None
        )
    except Exception:
        image_token_id = None
    for idx, req in enumerate(requests):
        if not req.verify:
            continue
        try:
            import hashlib

            buf = images[idx].tobytes()
            sha = hashlib.sha256(buf).hexdigest()[:16]
        except Exception:
            sha = "unknown"
        expected_tokens = -1
        if grid is not None:
            try:
                merge = getattr(
                    getattr(processor, "image_processor", None), "merge_size", 2
                )
                expected_tokens = int((grid[idx].prod() // (merge * merge)).item())
            except Exception:
                expected_tokens = -1
        text_token_count = -1
        try:
            if image_token_id is not None:
                text_token_count = int((input_ids[idx] == image_token_id).sum().item())
        except Exception:
            text_token_count = -1
        logger.info(
            "[verify] idx=%d size=%sx%s sha256=%s grid_thw=%s expected_image_tokens=%s text_image_tokens=%s",
            idx,
            getattr(images[idx], "width", None),
            getattr(images[idx], "height", None),
            sha,
            tuple(grid[idx].tolist()) if grid is not None else None,
            expected_tokens,
            text_token_count,
        )


def load_hf_backend(
    config: ModelLoadConfig,
    *,
    chat_template: ChatTemplateOptions,
    preprocess: VlmPreprocessOptions,
) -> HfBackend:
    variant = config.variant or _detect_variant(config.model_name_or_path)
    dtype = (
        _dtype_from_str(config.torch_dtype)
        if config.torch_dtype is not None
        else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.attn_implementation is not None:
        attn_impl = config.attn_implementation
        if attn_impl.lower() in ("flash_attn", "flash_attention_2"):
            attn_impl = "flash_attention_2"
        model_kwargs["attn_implementation"] = attn_impl
    if config.device_map is not None:
        model_kwargs["device_map"] = config.device_map

    processor: object | None = None
    tokenizer: PreTrainedTokenizerBase
    if variant == "vlm":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_name_or_path,
            **model_kwargs,
        )
        processor = AutoProcessor.from_pretrained(
            config.model_name_or_path, trust_remote_code=config.trust_remote_code
        )
        tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            **model_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )

    if config.device_map is None and config.device is not None:
        cast(torch.nn.Module, model).to(config.device)
    model.eval()

    normalize_tokenizer(tokenizer, options=preprocess)
    if processor is not None:
        configure_vlm_processor(processor, options=preprocess)

    supports_vlm = processor is not None
    return HfBackend(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        chat_template=chat_template,
        supports_vlm=supports_vlm,
    )


__all__ = ["HfBackend", "load_hf_backend"]
