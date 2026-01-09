"""Periodic rollout dumps for GRPO training/evaluation.

This is a lightweight debugging tool to help trace training dynamics:
- Train: dump a small number of the most recent (prompt, rollout, GT) triples.
- Eval: dump model outputs on a fixed subset of eval samples across time.

The intent is that reading these dumps gives an intuitive sense of *what* the model
is learning, beyond scalar metrics.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import random
from typing import Any, TypedDict

from typing_extensions import override

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from ..config.schema import GrpoDumpConfig
from ..utils import get_logger

logger = get_logger(__name__)


class _RolloutDumpRecord(TypedDict, total=False):
    split: str
    global_step: int
    epoch: float
    timestamp: str
    prompt: str
    completion: str
    solution: str
    prompt_id: str
    request_id: str
    eval_index: int
    generation_index: int


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return default


class RolloutDumpCallback(TrainerCallback):
    def __init__(self, config: GrpoDumpConfig) -> None:
        self.config = config
        self._trainer: Any | None = None
        self._resolved_dump_dir: str | None = None
        self._eval_indices: list[int] | None = None
        self._last_train_dump_step: int | None = None
        self._last_eval_dump_step: int | None = None

    def attach_trainer(self, trainer: Any) -> None:
        self._trainer = trainer

    def _is_rank0(self) -> bool:
        trainer = self._trainer
        if trainer is None:
            return True

        is_zero = getattr(trainer, "is_world_process_zero", None)
        if callable(is_zero):
            try:
                return bool(is_zero())
            except Exception:
                return True

        accelerator = getattr(trainer, "accelerator", None)
        if accelerator is not None:
            try:
                return bool(getattr(accelerator, "is_main_process", True))
            except Exception:
                return True

        rank_raw = os.environ.get("RANK")
        if rank_raw is None:
            return True
        try:
            return int(rank_raw) == 0
        except Exception:
            return True

    def _resolve_dump_dir(self) -> str | None:
        if self._resolved_dump_dir is not None:
            return self._resolved_dump_dir

        trainer = self._trainer
        base = self.config.dump_dir or "rollout_dumps"
        join_with_output_dir = True

        if trainer is not None and base in {
            "logging_dir",
            "$logging_dir",
            "${logging_dir}",
        }:
            resolved = getattr(getattr(trainer, "args", None), "logging_dir", None)
            if resolved:
                base = str(resolved)
                # logging_dir is already a fully-resolved runtime setting. Even if
                # it's a relative path, it should NOT be nested under output_dir.
                join_with_output_dir = False

        if trainer is not None:
            output_dir = getattr(getattr(trainer, "args", None), "output_dir", None)
            if join_with_output_dir and output_dir and not os.path.isabs(base):
                base = os.path.join(str(output_dir), base)

        try:
            os.makedirs(base, exist_ok=True)
        except Exception as exc:
            logger.warning("Failed to create dump_dir=%s: %s", base, exc)
            return None

        self._resolved_dump_dir = base
        return base

    def _write_jsonl(self, path: str, records: list[_RolloutDumpRecord]) -> None:
        if not records:
            return
        dump_dir = os.path.dirname(path) or "."
        os.makedirs(dump_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for rec in records:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _dump_train_recent(self, *, state: TrainerState) -> None:
        trainer = self._trainer
        if trainer is None:
            return

        logs = getattr(trainer, "_logs", None)
        if not isinstance(logs, dict):
            return

        prompts_raw = logs.get("prompt")
        completions_raw = logs.get("completion")
        solutions_raw = logs.get("solution")
        if prompts_raw is None or completions_raw is None:
            return

        prompts = list(prompts_raw)
        completions = list(completions_raw)
        solutions = list(solutions_raw) if solutions_raw is not None else []

        limit = int(self.config.dump_sample_size)
        limit = max(1, limit)
        max_len = min(len(prompts), len(completions))
        if solutions:
            max_len = min(max_len, len(solutions))

        if max_len <= 0:
            return

        take = min(limit, max_len)
        prompts = prompts[-take:]
        completions = completions[-take:]
        solutions = solutions[-take:] if solutions else [""] * take

        base_dir = self._resolve_dump_dir()
        if base_dir is None:
            return
        path = os.path.join(
            base_dir, "train", f"step_{int(state.global_step):08d}.jsonl"
        )

        records: list[_RolloutDumpRecord] = []
        timestamp = _now_iso()
        epoch = float(state.epoch) if state.epoch is not None else 0.0
        for prompt, completion, solution in zip(prompts, completions, solutions):
            records.append(
                {
                    "split": "train",
                    "global_step": int(state.global_step),
                    "epoch": epoch,
                    "timestamp": timestamp,
                    "prompt": str(prompt),
                    "completion": str(completion),
                    "solution": str(solution),
                }
            )

        try:
            self._write_jsonl(path, records)
            logger.info(
                "Rollout dump (train): step=%s samples=%s path=%s",
                state.global_step,
                len(records),
                path,
            )
        except Exception as exc:
            logger.warning("Failed to write train rollout dump: %s", exc)

    def _ensure_eval_indices(self) -> list[int] | None:
        if self._eval_indices is not None:
            return self._eval_indices

        trainer = self._trainer
        if trainer is None:
            return None

        eval_dataset = getattr(trainer, "eval_dataset", None)
        if eval_dataset is None:
            return None

        try:
            dataset_size = len(eval_dataset)
        except Exception:
            return None

        if dataset_size <= 0:
            return None

        sample_size = (
            dataset_size
            if self.config.eval_sample_size is None
            else int(self.config.eval_sample_size)
        )
        sample_size = max(1, min(sample_size, dataset_size))

        rng = random.Random(int(self.config.eval_seed))
        if sample_size >= dataset_size:
            indices = list(range(dataset_size))
        else:
            indices = rng.sample(range(dataset_size), sample_size)
            indices.sort()

        self._eval_indices = indices

        # Best-effort persistence for easier debugging / resume stability.
        base_dir = self._resolve_dump_dir()
        if base_dir is not None and self._is_rank0():
            path = os.path.join(base_dir, "eval", "indices.json")
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"seed": int(self.config.eval_seed), "indices": indices}
                        )
                        + "\n"
                    )
            except Exception:
                pass

        return indices

    def _decode_completion_text(self, trainer: Any, completion: object) -> str:
        if isinstance(completion, str):
            return completion

        token_ids: object | None = None
        if isinstance(completion, list):
            token_ids = completion
        elif isinstance(completion, dict):
            token_ids = completion.get("token_ids")

        if token_ids is not None:
            decoder = getattr(
                getattr(trainer, "processing_class", None), "decode", None
            )
            if callable(decoder):
                try:
                    return str(decoder(token_ids))
                except Exception:
                    pass
        return str(completion)

    def _render_prompt(self, trainer: Any, messages: object) -> str:
        apply_fn = getattr(trainer, "_apply_chat_template_to_messages_list", None)
        if callable(apply_fn):
            try:
                rendered = apply_fn([messages])
                if isinstance(rendered, list) and rendered:
                    return str(rendered[0])
            except Exception:
                pass
        return json.dumps(messages, ensure_ascii=False)

    def _dump_eval_fixed_subset(self, *, state: TrainerState) -> None:
        trainer = self._trainer
        if trainer is None:
            return

        indices = self._ensure_eval_indices()
        if not indices:
            return

        eval_dataset = getattr(trainer, "eval_dataset", None)
        if eval_dataset is None:
            return

        accelerator = getattr(trainer, "accelerator", None)
        world_size = _safe_int(getattr(accelerator, "num_processes", None), 1)
        process_index = _safe_int(getattr(accelerator, "process_index", None), 0)
        world_size = max(1, world_size)
        process_index = max(0, min(process_index, world_size - 1))

        local_indices = indices[process_index::world_size]
        if not local_indices:
            local_indices = []

        num_generations = _safe_int(getattr(trainer, "num_generations", None), 1)
        num_generations = max(1, num_generations)

        inputs: list[dict[str, Any]] = []
        meta: list[tuple[int, int, str]] = []

        for eval_index in local_indices:
            try:
                sample_raw = eval_dataset[int(eval_index)]
            except Exception:
                continue

            if not isinstance(sample_raw, dict):
                continue

            solution = sample_raw.get("solution")
            solution_text = str(solution) if solution is not None else ""

            for generation_index in range(num_generations):
                inputs.append(copy.deepcopy(sample_raw))
                meta.append((int(eval_index), int(generation_index), solution_text))

        if not inputs:
            return

        generate_fn = getattr(trainer, "_generate_completions", None)
        if not callable(generate_fn):
            return

        model = getattr(trainer, "model", None)
        was_training = bool(getattr(model, "training", False))
        if model is not None:
            try:
                model.eval()
            except Exception:
                pass

        try:
            with torch.no_grad():
                outputs = generate_fn(inputs)
        finally:
            if model is not None and was_training:
                try:
                    model.train()
                except Exception:
                    pass

        local_records: list[_RolloutDumpRecord] = []
        timestamp = _now_iso()
        epoch = float(state.epoch) if state.epoch is not None else 0.0
        for (eval_index, generation_index, solution_text), output in zip(meta, outputs):
            try:
                out_messages = (
                    output.get("messages") if isinstance(output, dict) else None
                )
                if not isinstance(out_messages, list) or not out_messages:
                    continue
                prompt_messages = out_messages[:-1]
                completion_obj = (
                    out_messages[-1].get("content")
                    if isinstance(out_messages[-1], dict)
                    else None
                )
                completion_text = self._decode_completion_text(trainer, completion_obj)

                prompt_text = self._render_prompt(trainer, prompt_messages)

                record: _RolloutDumpRecord = {
                    "split": "eval",
                    "global_step": int(state.global_step),
                    "epoch": epoch,
                    "timestamp": timestamp,
                    "eval_index": int(eval_index),
                    "generation_index": int(generation_index),
                    "prompt": prompt_text,
                    "completion": completion_text,
                    "solution": solution_text,
                }

                prompt_id = (
                    output.get("prompt_id") if isinstance(output, dict) else None
                )
                if isinstance(prompt_id, str):
                    record["prompt_id"] = prompt_id
                request_id = (
                    output.get("request_id") if isinstance(output, dict) else None
                )
                if isinstance(request_id, str):
                    record["request_id"] = request_id

                local_records.append(record)
            except Exception:
                continue

        if world_size > 1:
            try:
                from accelerate.utils import gather_object

                gathered = gather_object(local_records)
                all_records: list[_RolloutDumpRecord] = []
                for part in gathered:
                    if isinstance(part, list):
                        all_records.extend(part)
                local_records = all_records
            except Exception:
                # Best-effort only; rank0 may still dump its local subset.
                pass

        if not self._is_rank0():
            return

        base_dir = self._resolve_dump_dir()
        if base_dir is None:
            return

        local_records.sort(
            key=lambda rec: (
                int(rec.get("eval_index", 0)),
                int(rec.get("generation_index", 0)),
            )
        )

        path = os.path.join(
            base_dir, "eval", f"step_{int(state.global_step):08d}.jsonl"
        )
        try:
            self._write_jsonl(path, local_records)
            logger.info(
                "Rollout dump (eval): step=%s prompts=%s rollouts=%s path=%s",
                state.global_step,
                len(indices),
                len(local_records),
                path,
            )
        except Exception as exc:
            logger.warning("Failed to write eval rollout dump: %s", exc)

    @override
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        _ = args
        _ = control
        _ = kwargs
        if not self.config.enabled:
            return
        if state.global_step <= 0:
            return
        if self._last_train_dump_step == int(state.global_step):
            return
        if int(state.global_step) % int(self.config.dump_step or 1) != 0:
            return
        if not self._is_rank0():
            return
        self._last_train_dump_step = int(state.global_step)
        self._dump_train_recent(state=state)

    @override
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        _ = args
        _ = control
        _ = kwargs
        if not self.config.enabled:
            return
        if self._last_eval_dump_step == int(state.global_step):
            return
        self._last_eval_dump_step = int(state.global_step)
        self._dump_eval_fixed_subset(state=state)
