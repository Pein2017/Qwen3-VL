"""Custom GKD trainer wrapper that records KL and token accuracy metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from packaging import version
from swift.trainers.rlhf_trainer.gkd_trainer import GKDTrainer as _MsSwiftGKDTrainer

from ..config import VisualKDConfig

logger = transformers.utils.logging.get_logger(__name__)


class GKDTrainerWithMetrics(_MsSwiftGKDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure metrics dict exists even if parent implementation changes.
        if not hasattr(self, "_metrics"):
            self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        self._visual_hooks = []
        self._student_visual_cache: Dict[str, torch.Tensor] = {}
        self._teacher_visual_cache: Dict[str, torch.Tensor] = {}

        cfg_raw = getattr(self.args, "visual_kd_config", VisualKDConfig.disabled())
        if isinstance(cfg_raw, Mapping):
            cfg = VisualKDConfig.from_mapping(cfg_raw)
        elif isinstance(cfg_raw, VisualKDConfig):
            cfg = cfg_raw
        else:
            raise TypeError("visual_kd_config must be VisualKDConfig or mapping")

        self._visual_kd_config: VisualKDConfig = cfg
        self._visual_kd_enabled = cfg.enabled
        self._visual_kd_weight = cfg.weight
        self._visual_kd_targets = list(cfg.targets)
        self._visual_kd_distance = cfg.distance

        raw_llm_kd_weight = getattr(self.args, "llm_kd_weight", 1.0)
        try:
            llm_kd_weight = float(raw_llm_kd_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"llm_kd_weight must be numeric; received {raw_llm_kd_weight!r}"
            ) from exc
        if not math.isfinite(llm_kd_weight):
            raise ValueError(
                f"llm_kd_weight must be finite; received {raw_llm_kd_weight!r}"
            )
        if llm_kd_weight < 0:
            raise ValueError(
                f"llm_kd_weight must be non-negative; received {raw_llm_kd_weight!r}"
            )
        self._llm_kd_weight = float(llm_kd_weight)

        if self._visual_kd_enabled:
            if self._visual_kd_weight <= 0:
                raise ValueError(
                    "visual_kd enabled but weight<=0; disable visual_kd or set a positive weight"
                )
            elif not self._visual_kd_targets:
                raise ValueError(
                    "visual_kd enabled but no targets provided; specify at least one target or disable visual_kd"
                )

        teacher_available = self._has_teacher_model()
        if self._llm_kd_weight > 0 and not teacher_available:
            raise RuntimeError(
                "llm_kd_weight > 0 requires a teacher model. Provide rlhf.teacher_model or set llm_kd_weight to 0."
            )
        if self._visual_kd_enabled and not teacher_available:
            raise RuntimeError(
                "visual_kd is enabled but no teacher model is attached. Attach a teacher_model or disable visual_kd in the configuration."
            )

        if self._visual_kd_enabled:
            try:
                self._register_visual_hooks()
            except Exception as exc:
                self._remove_visual_hooks()
                raise RuntimeError(
                    "visual_kd failed to register visual feature hooks. Ensure student and teacher expose the requested visual modules."
                    f" Original error: {exc}"
                ) from exc

            if not self._visual_hooks:
                self._remove_visual_hooks()
                raise RuntimeError(
                    "visual_kd enabled but no hooks were registered. Verify the configured targets exist on both student and teacher models."
                )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        self._ensure_visual_kd_state()

        if self._visual_kd_enabled:
            self._clear_visual_caches()

        model_inputs = {
            k: v for k, v in inputs.items() if k not in {"prompt", "labels"}
        }
        use_logits_to_keep = self.get_use_logits_to_keep(True)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(inputs)
            model_inputs["logits_to_keep"] = inputs["logits_to_keep"]

        # Always provide labels so CE/accuracy are available even when sft_alpha == 0.
        model_inputs["labels"] = inputs["labels"]

        outputs_student = model(**model_inputs)

        labels = inputs["labels"]
        student_logits = outputs_student.logits
        dtype = student_logits.dtype
        student_logits_next = student_logits[:, :-1, :]
        labels_next = labels[:, 1:]
        mask = labels_next != -100
        valid_count = int(mask.sum().item())
        vocab_size = student_logits_next.shape[-1]

        if valid_count > 0:
            masked_student_logits = student_logits_next.masked_select(
                mask.unsqueeze(-1)
            ).view(1, valid_count, vocab_size)
            masked_labels = labels_next.masked_select(mask)
        else:
            masked_student_logits = student_logits_next.new_empty((1, 0, vocab_size))
            masked_labels = labels_next.new_empty((0,), dtype=labels_next.dtype)

        mode = "train" if model.training else "eval"

        teacher_outputs = None
        weighted_llm_kd_loss: Optional[torch.Tensor] = None

        teacher_available = self._has_teacher_model()
        llm_kd_requested = self._llm_kd_weight > 0.0
        if llm_kd_requested and not teacher_available:
            raise RuntimeError(
                "llm_kd_weight > 0 requires a teacher model. Verify rlhf.teacher_model in the configuration."
            )
        run_teacher_forward = teacher_available and (
            llm_kd_requested or self._visual_kd_enabled
        )

        if run_teacher_forward:
            teacher_outputs = self._run_teacher_forward(model_inputs, student_logits)

        if teacher_outputs is not None:
            teacher_logits = teacher_outputs.logits.to(dtype)
            teacher_logits_next = teacher_logits[:, :-1, :]
            teacher_vocab_size = teacher_logits_next.shape[-1]

            if teacher_vocab_size != vocab_size:
                raise ValueError(
                    "Teacher/student vocabulary size mismatch detected during loss computation: "
                    f"student={vocab_size}, teacher={teacher_vocab_size}. "
                    "Ensure both models share the same tokenizer and vocabulary."
                )

            if llm_kd_requested:
                if valid_count > 0:
                    masked_teacher_logits = teacher_logits_next.masked_select(
                        mask.unsqueeze(-1)
                    ).view(1, valid_count, teacher_vocab_size)
                    student_for_kl = (
                        masked_student_logits
                        if model.training
                        else masked_student_logits.detach()
                    )
                    llm_kd_loss = self.generalized_jsd_loss(
                        student_logits=student_for_kl,
                        teacher_logits=masked_teacher_logits,
                        beta=self.beta,
                    )
                else:
                    llm_kd_loss = student_logits.new_zeros(())
                weighted_llm_kd_loss = llm_kd_loss * self._llm_kd_weight

        ce_loss = outputs_student.loss
        if not teacher_available or not llm_kd_requested:
            # When LM KD is disabled (weight == 0) or we have no teacher, default to CE-only.
            sft_weight = 1.0
        else:
            sft_weight = float(getattr(self.args, "sft_alpha", 0.0))

        total_loss = student_logits.new_zeros(
            (), dtype=dtype, device=student_logits.device
        )

        if weighted_llm_kd_loss is not None:
            total_loss = total_loss + weighted_llm_kd_loss
            self._metrics[mode]["llm_kd_loss"].append(
                weighted_llm_kd_loss.detach().cpu()
            )

        if ce_loss is not None:
            weighted_sft_loss = ce_loss * sft_weight
            total_loss = total_loss + weighted_sft_loss
            self._metrics[mode]["sft_loss"].append(weighted_sft_loss.detach().cpu())
        else:
            weighted_sft_loss = None

        if self._visual_kd_enabled:
            vision_loss = self._compute_visual_kd_loss()
            if vision_loss is not None:
                weighted_vision_loss = self._visual_kd_weight * vision_loss
                total_loss = total_loss + weighted_vision_loss
                self._metrics[mode]["vision_kd_loss"].append(
                    weighted_vision_loss.detach().cpu()
                )

        self._metrics[mode]["loss"].append(total_loss.detach().cpu())

        if valid_count > 0:
            logits_flat = masked_student_logits.squeeze(0)
            preds = logits_flat.argmax(dim=-1)
            accuracy = (preds == masked_labels).float().mean()
        else:
            accuracy = student_logits.new_zeros(())
        self._metrics[mode]["token_acc"].append(accuracy.detach().cpu())

        return (total_loss, outputs_student) if return_outputs else total_loss

    def _run_teacher_forward(
        self,
        model_inputs: Dict[str, Union[torch.Tensor, Any]],
        student_logits: torch.Tensor,
    ):
        # Remove labels for teacher forward.
        teacher_inputs = {k: v for k, v in model_inputs.items() if k != "labels"}

        logits_device = student_logits.device
        device_type = logits_device.type
        dtype = student_logits.dtype
        using_deepspeed = bool(getattr(self, "deepspeed", None))
        autocast_enabled = (
            not using_deepspeed
            and device_type in {"cuda", "xpu"}
            and dtype in {torch.float16, torch.bfloat16}
        )

        with torch.no_grad():
            if autocast_enabled:
                with torch.autocast(device_type=device_type, dtype=dtype):
                    outputs_teacher = self.teacher_model(**teacher_inputs)
            else:
                outputs_teacher = self.teacher_model(**teacher_inputs)

        return outputs_teacher

    def _register_visual_hooks(self) -> None:
        student_unwrapped = self.accelerator.unwrap_model(self.model)
        teacher_model = getattr(self, "teacher_model", None)
        if teacher_model is None:
            raise RuntimeError("visual_kd requires a teacher model")
        teacher_unwrapped = self.accelerator.unwrap_model(teacher_model)

        student_visual = self._resolve_visual_branch(student_unwrapped)
        teacher_visual = self._resolve_visual_branch(teacher_unwrapped)

        if student_visual is None or teacher_visual is None:
            raise RuntimeError("Failed to resolve visual modules for visual KD")

        if "merger" in self._visual_kd_targets:
            student_merger = getattr(student_visual, "merger", None)
            teacher_merger = getattr(teacher_visual, "merger", None)
            if not isinstance(student_merger, nn.Module) or not isinstance(
                teacher_merger, nn.Module
            ):
                raise RuntimeError("merger not available for visual KD")
            self._visual_hooks.append(
                student_merger.register_forward_hook(self._make_student_hook("merger"))
            )
            self._visual_hooks.append(
                teacher_merger.register_forward_hook(self._make_teacher_hook("merger"))
            )

        if "deepstack" in self._visual_kd_targets:
            student_list = getattr(student_visual, "deepstack_merger_list", None)
            teacher_list = getattr(teacher_visual, "deepstack_merger_list", None)
            if not isinstance(student_list, nn.ModuleList) or not isinstance(
                teacher_list, nn.ModuleList
            ):
                raise RuntimeError("deepstack_merger_list not available for visual KD")
            if len(student_list) != len(teacher_list):
                raise RuntimeError("Student/teacher deepstack size mismatch")
            for idx, (student_mod, teacher_mod) in enumerate(
                zip(student_list, teacher_list)
            ):
                name = f"deepstack_{idx}"
                self._visual_hooks.append(
                    student_mod.register_forward_hook(self._make_student_hook(name))
                )
                self._visual_hooks.append(
                    teacher_mod.register_forward_hook(self._make_teacher_hook(name))
                )

    def _resolve_visual_branch(self, module: nn.Module) -> Optional[nn.Module]:
        visited = set()
        current = module
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "visual") and isinstance(
                getattr(current, "visual"), nn.Module
            ):
                return getattr(current, "visual")
            for attr in ("model", "base_model", "model_wrapped"):
                next_mod = getattr(current, attr, None)
                if isinstance(next_mod, nn.Module):
                    current = next_mod
                    break
            else:
                get_base_model = getattr(current, "get_base_model", None)
                if callable(get_base_model):
                    try:
                        current = get_base_model()
                        continue
                    except Exception:
                        pass
                current = None
        return None

    def _make_student_hook(self, name: str):
        def hook(_module: nn.Module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Expected tensor output for visual hook '{name}', got {type(output)}"
                )
            self._student_visual_cache[name] = output

        return hook

    def _make_teacher_hook(self, name: str):
        def hook(_module: nn.Module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Expected tensor output for visual hook '{name}', got {type(output)}"
                )
            self._teacher_visual_cache[name] = output.detach()

        return hook

    def _clear_visual_caches(self) -> None:
        self._student_visual_cache.clear()
        self._teacher_visual_cache.clear()

    def _remove_visual_hooks(self) -> None:
        hooks = getattr(self, "_visual_hooks", None)
        if not hooks:
            return
        for handle in hooks:
            try:
                handle.remove()
            except Exception:
                pass
        hooks.clear()

    def _compute_visual_kd_loss(self) -> Optional[torch.Tensor]:
        def compute_distance(student_feat: torch.Tensor, teacher_feat: torch.Tensor):
            teacher_feat = teacher_feat.to(student_feat.device, student_feat.dtype)
            if student_feat.shape != teacher_feat.shape:
                raise ValueError(
                    "Visual KD feature shape mismatch: "
                    f"student={student_feat.shape}, teacher={teacher_feat.shape}"
                )
            if self._visual_kd_distance == "mse":
                return F.mse_loss(student_feat, teacher_feat)
            if self._visual_kd_distance == "cosine":
                s_flat = student_feat.view(-1, student_feat.shape[-1])
                t_flat = teacher_feat.view(-1, teacher_feat.shape[-1])
                cosine = F.cosine_similarity(s_flat, t_flat, dim=-1)
                return (1 - cosine).mean()
            raise ValueError(
                f"Unsupported visual KD distance: {self._visual_kd_distance}"
            )

        contributions = []
        missing_targets: list[str] = []
        for name in self._visual_kd_targets:
            if name == "deepstack":
                keys = sorted(
                    key
                    for key in self._student_visual_cache.keys()
                    if key.startswith("deepstack_")
                )
                if not keys:
                    missing_targets.append("deepstack (no hooks fired)")
                    continue
                layer_losses = []
                missing_layers = []
                for key in keys:
                    student_feat = self._student_visual_cache.get(key)
                    teacher_feat = self._teacher_visual_cache.get(key)
                    if student_feat is None or teacher_feat is None:
                        missing_layers.append(key)
                        continue
                    layer_losses.append(compute_distance(student_feat, teacher_feat))
                if layer_losses:
                    contributions.append(
                        torch.stack(
                            [loss_item.float() for loss_item in layer_losses]
                        ).mean()
                    )
                else:
                    if missing_layers:
                        missing_targets.append(
                            "deepstack missing activations: "
                            + ", ".join(sorted(missing_layers))
                        )
                    else:
                        missing_targets.append("deepstack (no matching activations)")
                continue

            student_feat = self._student_visual_cache.get(name)
            teacher_feat = self._teacher_visual_cache.get(name)
            if student_feat is None or teacher_feat is None:
                missing_targets.append(name)
                continue
            contributions.append(compute_distance(student_feat, teacher_feat))

        if not contributions:
            missing_desc = (
                ", ".join(missing_targets) if missing_targets else "all targets"
            )
            raise RuntimeError(
                "visual_kd did not receive any feature activations for the configured targets "
                f"{tuple(self._visual_kd_targets)} (missing {missing_desc}). "
                "Ensure the dataset yields image inputs and that both teacher and student expose the requested visual modules."
            )

        stacked = torch.stack([loss.float() for loss in contributions])
        return stacked.mean()

    def __del__(self):
        try:
            self._remove_visual_hooks()
        except Exception:
            pass

    def _ensure_visual_kd_state(self) -> None:
        if not hasattr(self, "_visual_kd_enabled"):
            self._visual_kd_enabled = False
        if not hasattr(self, "_visual_kd_weight"):
            self._visual_kd_weight = 0.0
        if not hasattr(self, "_visual_kd_targets"):
            self._visual_kd_targets = []
        if not hasattr(self, "_visual_kd_distance"):
            self._visual_kd_distance = "mse"
        if not hasattr(self, "_visual_hooks"):
            self._visual_hooks = []
        if not hasattr(self, "_student_visual_cache"):
            self._student_visual_cache = {}
        if not hasattr(self, "_teacher_visual_cache"):
            self._teacher_visual_cache = {}

    def _has_teacher_model(self) -> bool:
        return hasattr(self, "teacher_model") and self.teacher_model is not None

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        sanitized_logs: Dict[str, float] = dict(logs)
        aggregated_logs: Dict[str, float] = {}
        keys_to_remove: set[str] = set()

        for mode, prefix in (("train", "train/"), ("eval", "eval/")):
            if not self._metrics[mode]:
                continue

            metrics_data = {
                key: list(values) for key, values in self._metrics[mode].items()
            }

            for key, values in metrics_data.items():
                if not values:
                    continue

                if isinstance(values[0], torch.Tensor):
                    stacked = torch.stack(values).float()
                else:
                    stacked = torch.tensor(values, dtype=torch.float32)

                finite_mask = torch.isfinite(stacked)
                if not torch.all(finite_mask):
                    bad_count = int((~finite_mask).sum().item())
                    step_info = ""
                    state = getattr(self, "state", None)
                    if (
                        state is not None
                        and getattr(state, "global_step", None) is not None
                    ):
                        step_info = f" (step={int(state.global_step)})"
                    logger.warning(
                        "Skipping %s non-finite %s entries in %s mode%s",
                        bad_count,
                        key,
                        mode,
                        step_info,
                    )
                finite_values = stacked[finite_mask]
                if finite_values.numel() == 0:
                    continue

                aggregated = float(finite_values.mean().item())
                metric_key = f"{prefix}{key}"
                aggregated_logs[metric_key] = aggregated
                keys_to_remove.add(metric_key)
                if mode == "train" and key == "loss":
                    aggregated_logs["loss"] = aggregated
                    keys_to_remove.add("loss")

            self._metrics[mode].clear()

        for key in keys_to_remove:
            sanitized_logs.pop(key, None)

        sanitized_logs.update(aggregated_logs)

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(sanitized_logs, start_time)
        else:
            super().log(sanitized_logs)
