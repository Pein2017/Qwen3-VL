#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vLLM colocate backend for fast generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase

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
    VllmEngineOptions,
)
from ..preprocess import configure_vlm_processor, normalize_tokenizer
from ..stop_policy import normalize_stop_options, truncate_text_at_stops

logger = get_logger(__name__)


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


def _decode_tokens(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Sequence[int],
    *,
    skip_special_tokens: bool,
    clean_up_tokenization_spaces: bool,
) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )


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


def _build_mm_kwargs(options: VlmPreprocessOptions) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if options.min_pixels is not None:
        kwargs["min_pixels"] = int(options.min_pixels)
    if options.max_pixels is not None:
        kwargs["max_pixels"] = int(options.max_pixels)
    if options.do_resize is not None:
        kwargs["do_resize"] = bool(options.do_resize)
    return kwargs


@dataclass
class VllmBackend:
    llm: Any
    tokenizer: PreTrainedTokenizerBase
    processor: object | None
    chat_template: ChatTemplateOptions
    preprocess: VlmPreprocessOptions
    supports_vlm: bool

    @property
    def name(self) -> str:
        return "vllm"

    def generate_text_batch(
        self,
        requests: Sequence[TextGenerationRequest],
        options: GenerationOptions,
        *,
        plugins: Sequence[object] | None = None,
    ) -> list[GenerationResult]:
        if not requests:
            return []
        if plugins:
            for plugin in plugins:
                name = getattr(plugin, "name", plugin.__class__.__name__)
                uses_hf_stopping = bool(getattr(plugin, "uses_hf_stopping_criteria", False))
                uses_hf_logits = bool(getattr(plugin, "uses_hf_logits_processor", False))
                if uses_hf_stopping or uses_hf_logits:
                    raise RuntimeError(
                        f"Plugin '{name}' requires HF-only hooks; vLLM backend unsupported"
                    )
        prompts = [
            render_chat_template(self.tokenizer, req.messages, options=self.chat_template)
            for req in requests
        ]

        stop_token_ids, stop_strings = normalize_stop_options(
            options.stop, tokenizer=self.tokenizer
        )
        sampling = _build_sampling_params(options, stop_token_ids, stop_strings)
        outputs = self.llm.generate(prompts, sampling)
        results: list[GenerationResult] = []
        for output in outputs:
            completion = output.outputs[0] if output.outputs else None
            token_ids = completion.token_ids if completion is not None else []
            raw_text = _decode_tokens(
                self.tokenizer,
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=options.decode.clean_up_tokenization_spaces,
            )
            clean_text = _decode_tokens(
                self.tokenizer,
                token_ids,
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
            prompt_tokens = (
                len(output.prompt_token_ids) if output.prompt_token_ids else None
            )
            completion_tokens = len(token_ids)
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
        if not self.supports_vlm:
            raise RuntimeError("vLLM backend does not support VLM generation")
        if not requests:
            return []
        if plugins:
            for plugin in plugins:
                name = getattr(plugin, "name", plugin.__class__.__name__)
                uses_hf_stopping = bool(getattr(plugin, "uses_hf_stopping_criteria", False))
                uses_hf_logits = bool(getattr(plugin, "uses_hf_logits_processor", False))
                if uses_hf_stopping or uses_hf_logits:
                    raise RuntimeError(
                        f"Plugin '{name}' requires HF-only hooks; vLLM backend unsupported"
                    )

        prompts: list[dict[str, Any]] = []
        mm_kwargs = _build_mm_kwargs(self.preprocess)
        for req in requests:
            image = _extract_single_image(req.messages)
            if image is None:
                raise ValueError("VLM request must include exactly one image")
            prompt_text = render_chat_template(
                self.processor or self.tokenizer,
                req.messages,
                options=self.chat_template,
            )
            prompt_payload: dict[str, Any] = {
                "prompt": prompt_text,
                "multi_modal_data": {"image": image},
            }
            if mm_kwargs:
                prompt_payload["mm_processor_kwargs"] = mm_kwargs
            prompts.append(prompt_payload)

        stop_token_ids, stop_strings = normalize_stop_options(
            options.stop, tokenizer=self.tokenizer
        )
        sampling = _build_sampling_params(options, stop_token_ids, stop_strings)
        outputs = self.llm.generate(prompts, sampling)
        results: list[GenerationResult] = []
        for output in outputs:
            completion = output.outputs[0] if output.outputs else None
            token_ids = completion.token_ids if completion is not None else []
            raw_text = _decode_tokens(
                self.tokenizer,
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=options.decode.clean_up_tokenization_spaces,
            )
            clean_text = _decode_tokens(
                self.tokenizer,
                token_ids,
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
            prompt_tokens = (
                len(output.prompt_token_ids) if output.prompt_token_ids else None
            )
            completion_tokens = len(token_ids)
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


def _build_sampling_params(
    options: GenerationOptions,
    stop_token_ids: Sequence[int],
    stop_strings: Sequence[str],
):
    from vllm import SamplingParams

    do_sample = (
        options.do_sample if options.do_sample is not None else options.temperature > 0
    )
    params = SamplingParams(
        max_tokens=options.max_new_tokens,
        temperature=options.temperature if do_sample else 0.0,
        top_p=options.top_p,
        repetition_penalty=options.repetition_penalty,
        stop=list(stop_strings) if stop_strings else None,
        stop_token_ids=list(stop_token_ids) if stop_token_ids else None,
        seed=options.seed,
    )
    if options.no_repeat_ngram_size:
        logger.warning("vLLM backend ignores no_repeat_ngram_size")
    return params


def _check_vllm_vlm_support(llm: Any) -> bool:
    try:
        from vllm.multimodal import MULTIMODAL_REGISTRY

        model_config = llm.llm_engine.model_config
        return bool(MULTIMODAL_REGISTRY.supports_multimodal_inputs(model_config))
    except Exception:
        return False


def load_vllm_backend(
    config: ModelLoadConfig,
    *,
    vllm_options: VllmEngineOptions,
    chat_template: ChatTemplateOptions,
    preprocess: VlmPreprocessOptions,
) -> VllmBackend:
    from vllm import LLM

    variant = config.variant or _detect_variant(config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=config.trust_remote_code
    )
    normalize_tokenizer(tokenizer, options=preprocess)

    processor: object | None = None
    if variant == "vlm":
        processor = AutoProcessor.from_pretrained(
            config.model_name_or_path, trust_remote_code=config.trust_remote_code
        )
        configure_vlm_processor(processor, options=preprocess)

    llm_kwargs: dict[str, Any] = {
        "model": config.model_name_or_path,
        "trust_remote_code": vllm_options.trust_remote_code,
        "tensor_parallel_size": vllm_options.tensor_parallel_size,
    }
    if vllm_options.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = vllm_options.gpu_memory_utilization
    if vllm_options.max_model_len is not None:
        llm_kwargs["max_model_len"] = vllm_options.max_model_len
    if vllm_options.dtype is not None:
        llm_kwargs["dtype"] = vllm_options.dtype
    if vllm_options.enforce_eager is not None:
        llm_kwargs["enforce_eager"] = vllm_options.enforce_eager

    llm = LLM(**llm_kwargs)
    supports_vlm = variant == "vlm" and _check_vllm_vlm_support(llm)
    return VllmBackend(
        llm=llm,
        tokenizer=tokenizer,
        processor=processor,
        chat_template=chat_template,
        preprocess=preprocess,
        supports_vlm=supports_vlm,
    )


__all__ = ["VllmBackend", "load_vllm_backend"]
