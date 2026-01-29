#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Facade for centralized generation backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.utils import get_logger

from .backends.base import GenerationBackend
from .backends.hf_backend import load_hf_backend
from .backends.vllm_backend import load_vllm_backend
from .contracts import (
    ChatTemplateOptions,
    GenerationOptions,
    GenerationResult,
    ModelLoadConfig,
    TextGenerationRequest,
    VlmGenerationRequest,
    VlmPreprocessOptions,
    VllmEngineOptions,
)

logger = get_logger(__name__)


def _plugin_attr(plugin: object, name: str, default):
    return getattr(plugin, name, default)


def _validate_plugins(
    plugins: Sequence[object] | None, *, backend: str, request_type: str
) -> None:
    if not plugins:
        return
    for plugin in plugins:
        name = _plugin_attr(plugin, "name", plugin.__class__.__name__)
        backends = _plugin_attr(plugin, "backends", ("hf", "vllm"))
        request_types = _plugin_attr(plugin, "request_types", ("text", "vlm"))
        uses_hf_stopping = bool(
            _plugin_attr(plugin, "uses_hf_stopping_criteria", False)
        )
        uses_hf_logits = bool(_plugin_attr(plugin, "uses_hf_logits_processor", False))

        if backend not in backends:
            raise RuntimeError(
                f"Plugin '{name}' is not compatible with backend '{backend}'"
            )
        if request_type not in request_types:
            raise RuntimeError(
                f"Plugin '{name}' is not compatible with request type '{request_type}'"
            )
        if backend != "hf" and (uses_hf_stopping or uses_hf_logits):
            raise RuntimeError(
                f"Plugin '{name}' requires HF-only hooks; backend '{backend}' unsupported"
            )


@dataclass
class GenerationEngine:
    backend: GenerationBackend
    fallback_backend: GenerationBackend | None = None

    @property
    def tokenizer(self) -> object | None:
        return self.backend.tokenizer

    @property
    def processor(self) -> object | None:
        return self.backend.processor

    def generate_text_batch(
        self,
        requests: Sequence[TextGenerationRequest],
        options: GenerationOptions,
        *,
        plugins: Sequence[object] | None = None,
    ) -> list[GenerationResult]:
        _validate_plugins(plugins, backend=self.backend.name, request_type="text")
        return self.backend.generate_text_batch(requests, options, plugins=plugins)

    def generate_vlm_batch(
        self,
        requests: Sequence[VlmGenerationRequest],
        options: GenerationOptions,
        *,
        plugins: Sequence[object] | None = None,
    ) -> list[GenerationResult]:
        if not self.backend.supports_vlm:
            if self.fallback_backend is None:
                raise RuntimeError(
                    f"Backend '{self.backend.name}' does not support VLM requests"
                )
            logger.warning(
                "VLM unsupported on backend '%s'; falling back to '%s'",
                self.backend.name,
                self.fallback_backend.name,
            )
            _validate_plugins(
                plugins, backend=self.fallback_backend.name, request_type="vlm"
            )
            return self.fallback_backend.generate_vlm_batch(
                requests, options, plugins=plugins
            )
        _validate_plugins(plugins, backend=self.backend.name, request_type="vlm")
        return self.backend.generate_vlm_batch(requests, options, plugins=plugins)


def build_hf_engine(
    config: ModelLoadConfig,
    *,
    chat_template: ChatTemplateOptions | None = None,
    preprocess: VlmPreprocessOptions | None = None,
) -> GenerationEngine:
    chat_template = chat_template or ChatTemplateOptions()
    preprocess = preprocess or VlmPreprocessOptions()
    backend = load_hf_backend(
        config, chat_template=chat_template, preprocess=preprocess
    )
    return GenerationEngine(backend=backend)


def build_vllm_engine(
    config: ModelLoadConfig,
    *,
    vllm_options: VllmEngineOptions | None = None,
    chat_template: ChatTemplateOptions | None = None,
    preprocess: VlmPreprocessOptions | None = None,
    fallback_backend: GenerationBackend | None = None,
) -> GenerationEngine:
    chat_template = chat_template or ChatTemplateOptions()
    preprocess = preprocess or VlmPreprocessOptions()
    vllm_options = vllm_options or VllmEngineOptions()
    backend = load_vllm_backend(
        config,
        vllm_options=vllm_options,
        chat_template=chat_template,
        preprocess=preprocess,
    )
    return GenerationEngine(backend=backend, fallback_backend=fallback_backend)


__all__ = ["GenerationEngine", "build_hf_engine", "build_vllm_engine"]
