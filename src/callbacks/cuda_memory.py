"""CUDA memory tracing + cleanup helpers for long-running training.

This callback is intended for GRPO + vLLM colocate runs where CUDA memory can
creep upward after evaluation due to allocator reserved-memory growth and/or
cross-allocator fragmentation (PyTorch vs vLLM).
"""

from __future__ import annotations

import gc
import json
import os
from typing import Any, TypedDict

from typing_extensions import override

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from ..config.schema import CudaMemoryConfig
from ..utils import get_logger

logger = get_logger(__name__)


class _CudaStats(TypedDict):
    device: int
    free_bytes: int
    total_bytes: int
    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int


class _CudaTraceRecord(TypedDict):
    tag: str
    step: int
    rank: int
    device: int
    free_gib: float
    total_gib: float
    allocated_gib: float
    reserved_gib: float
    max_allocated_gib: float
    max_reserved_gib: float


class CudaMemoryCallback(TrainerCallback):
    """Profile and optionally cleanup CUDA memory at eval/save boundaries."""

    def __init__(self, config: CudaMemoryConfig) -> None:
        self.config = config
        self._trainer: Any | None = None
        self._last_pre_eval_step: int | None = None
        self._last_post_eval_step: int | None = None
        self._last_save_step: int | None = None
        self._trace_path: str | None = None

    def attach_trainer(self, trainer: Any) -> None:
        self._trainer = trainer

    @staticmethod
    def _bytes_to_gib(value: int) -> float:
        return float(value) / (1024.0**3)

    @staticmethod
    def _global_rank() -> int:
        rank_raw = os.environ.get("RANK")
        if rank_raw is None:
            return 0
        try:
            return int(rank_raw)
        except ValueError:  # pragma: no cover - defensive
            return 0

    def _should_log(self, state: TrainerState) -> bool:
        if not self.config.enabled or not self.config.profile:
            return False
        if not torch.cuda.is_available():
            return False
        if self.config.rank0_only and self._global_rank() != 0:
            return False
        return True

    @staticmethod
    def _collect_cuda_stats() -> _CudaStats:
        device = int(torch.cuda.current_device())
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated_bytes = torch.cuda.memory_allocated(device)
        reserved_bytes = torch.cuda.memory_reserved(device)
        max_allocated_bytes = torch.cuda.max_memory_allocated(device)
        max_reserved_bytes = torch.cuda.max_memory_reserved(device)
        return {
            "device": device,
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "allocated_bytes": int(allocated_bytes),
            "reserved_bytes": int(reserved_bytes),
            "max_allocated_bytes": int(max_allocated_bytes),
            "max_reserved_bytes": int(max_reserved_bytes),
        }

    def _log_cuda_stats(self, *, tag: str, state: TrainerState) -> None:
        if not self._should_log(state):
            return
        stats = self._collect_cuda_stats()

        payload: _CudaTraceRecord = {
            "tag": tag,
            "step": int(state.global_step),
            "rank": self._global_rank(),
            "device": stats["device"],
            "free_gib": round(self._bytes_to_gib(stats["free_bytes"]), 4),
            "total_gib": round(self._bytes_to_gib(stats["total_bytes"]), 4),
            "allocated_gib": round(self._bytes_to_gib(stats["allocated_bytes"]), 4),
            "reserved_gib": round(self._bytes_to_gib(stats["reserved_bytes"]), 4),
            "max_allocated_gib": round(
                self._bytes_to_gib(stats["max_allocated_bytes"]), 4
            ),
            "max_reserved_gib": round(
                self._bytes_to_gib(stats["max_reserved_bytes"]), 4
            ),
        }
        logger.debug("cuda_mem %s", payload)

        trace_path = self._resolve_trace_path()
        if trace_path is not None:
            try:
                with open(trace_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                # Best-effort logging only; do not break training.
                pass

    def _resolve_trace_path(self) -> str | None:
        if self._trace_path is not None:
            return self._trace_path
        trainer = self._trainer
        if trainer is None:
            return None
        output_dir = getattr(getattr(trainer, "args", None), "output_dir", None)
        if not output_dir:
            return None
        rank = self._global_rank()
        self._trace_path = os.path.join(
            str(output_dir), f"cuda_memory_trace.rank{rank}.jsonl"
        )
        return self._trace_path

    def _try_reset_vllm_prefix_cache(self) -> None:
        trainer = self._trainer
        if trainer is None:
            return

        # ms-swift GRPO colocate mode:
        # - trainer.engine is GRPOVllmEngine
        # - trainer.engine.engine is the underlying vLLM engine
        engine = getattr(trainer, "engine", None)
        inner_engine = getattr(engine, "engine", None) if engine is not None else None
        if inner_engine is not None and hasattr(inner_engine, "reset_prefix_cache"):
            try:
                inner_engine.reset_prefix_cache()
                return
            except Exception:
                return

        # TRL colocate mode:
        llm = getattr(trainer, "llm", None)
        if llm is not None and hasattr(llm, "reset_prefix_cache"):
            try:
                llm.reset_prefix_cache()
            except Exception:
                return

    def _cleanup_cuda(self, *, state: TrainerState, reason: str) -> None:
        if not self.config.enabled or not self.config.cleanup:
            return
        if not torch.cuda.is_available():
            return

        for attempt in range(self.config.max_retries):
            if self.config.force_sync:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            try:
                gc.collect()
            except Exception:
                pass

            if self.config.force_sync:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass

            before = self._collect_cuda_stats()
            unused_reserved = int(before["reserved_bytes"]) - int(
                before["allocated_bytes"]
            )
            if unused_reserved < 1024 * 1024 * 1024:  # < 1 GiB
                break

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            if self.config.ipc_collect:
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

            if self.config.force_sync:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass

            after = self._collect_cuda_stats()
            freed_reserved = int(before["reserved_bytes"]) - int(
                after["reserved_bytes"]
            )
            freed_allocated = int(before["allocated_bytes"]) - int(
                after["allocated_bytes"]
            )

            if self._should_log(state):
                logger.debug(
                    "cuda_mem_cleanup[%s] attempt=%d freed_reserved=%.2fGiB freed_allocated=%.2fGiB",
                    reason,
                    attempt + 1,
                    self._bytes_to_gib(freed_reserved),
                    self._bytes_to_gib(freed_allocated),
                )

            # If almost nothing was freed, further retries are unlikely to help.
            if freed_reserved < 512 * 1024 * 1024:  # < 0.5 GiB
                break

    @override
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        self._log_cuda_stats(tag="train_begin", state=state)

    @override
    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if not self.config.enabled:
            return
        # Log once at the beginning of each evaluation loop.
        if self._last_pre_eval_step == state.global_step:
            return
        self._last_pre_eval_step = state.global_step
        self._log_cuda_stats(tag="eval_begin", state=state)

    @override
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if not self.config.enabled:
            return
        if self._last_post_eval_step == state.global_step:
            return
        self._last_post_eval_step = state.global_step

        self._log_cuda_stats(tag="post_eval", state=state)

        # vLLM can retain prefix caches across evals. Resetting is cheap and keeps
        # memory behavior more predictable.
        self._try_reset_vllm_prefix_cache()

        # Release reserved (but unused) CUDA memory back to the driver to restore
        # headroom for other allocators (notably vLLM in colocate mode).
        self._cleanup_cuda(state=state, reason="post_eval")

        self._log_cuda_stats(tag="post_eval_cleanup", state=state)

    @override
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if not self.config.enabled:
            return
        if self._last_save_step == state.global_step:
            return
        self._last_save_step = state.global_step

        self._log_cuda_stats(tag="post_save", state=state)
        self._cleanup_cuda(state=state, reason="post_save")
        self._log_cuda_stats(tag="post_save_cleanup", state=state)
