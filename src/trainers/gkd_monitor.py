"""Custom GKD trainer wrapper that records KL and token accuracy metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from packaging import version
from swift.trainers.rlhf_trainer.gkd_trainer import GKDTrainer as _MsSwiftGKDTrainer

from ..config import VisualKDConfig, VisualKDTargetConfig

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
        self._visual_kd_vit_cfg: VisualKDTargetConfig = cfg.vit
        self._visual_kd_aligner_cfg: VisualKDTargetConfig = cfg.aligner
        self._visual_kd_deepstack_cfg: VisualKDTargetConfig = cfg.deepstack
        self._visual_kd_targets = [
            name
            for name, target_cfg in (
                ("vit", cfg.vit),
                ("aligner", cfg.aligner),
                ("deepstack", cfg.deepstack),
            )
            if target_cfg.enabled
        ]

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

        if self._visual_kd_enabled and not self._visual_kd_targets:
            raise ValueError(
                "visual_kd enabled but no per-target configs are enabled; "
                "enable at least one of vit/aligner/deepstack or disable visual_kd"
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

        # Only register hooks if visual_kd is enabled
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

        # Only clear caches if visual_kd is enabled (avoids unnecessary dict operations)
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

        # CRITICAL DEBUG: Log label distribution to diagnose unmasking
        if valid_count > 0:
            seq_len = (
                labels_next.shape[1] if labels_next.ndim == 2 else labels_next.numel()
            )
            mask_ratio = float(mask.float().mean().item()) if mask.numel() else 0.0
            # Check if labels are mostly unmasked (indicating loss_scale='all' or template_backend='jinja')
            if mask_ratio > 0.3:  # More than 30% unmasked is suspicious
                import logging

                logger = logging.getLogger(__name__)
                # Sample labels from different parts of the sequence (beginning, middle, end)
                flat_labels = labels_next.flatten()
                total = flat_labels.numel()
                sample_indices = torch.cat(
                    [
                        flat_labels[: min(50, total)],  # Beginning
                        flat_labels[
                            max(0, total // 2 - 25) : min(total // 2 + 25, total)
                        ],  # Middle
                        flat_labels[max(0, total - 50) :],  # End
                    ]
                )
                unique_labels = torch.unique(sample_indices)
                # Count how many are -100 vs non--100
                masked_count = int((sample_indices == -100).sum().item())
                unmasked_count = int((sample_indices != -100).sum().item())
                logger.debug(
                    f"🔍 UNMASKING DETECTED: valid_count={valid_count}, seq_len={seq_len}, "
                    f"mask_ratio={mask_ratio:.4f}, sample_masked={masked_count}, sample_unmasked={unmasked_count}, "
                    f"unique_label_values (sample)={unique_labels.tolist()[:20]}"
                )

        self._maybe_log_token_stats(
            valid_count=valid_count,
            labels_next=labels_next,
            mask=mask,
            inputs=inputs,
        )

        if valid_count > 0:
            # Extract masked student logits - keep original tensors for now (needed for CE loss)
            flat_student = student_logits_next[mask]  # (valid_count, vocab_size)
            masked_student_logits = flat_student.unsqueeze(
                0
            )  # (1, valid_count, vocab_size)
            masked_labels = labels_next[mask]
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
        # Only run teacher forward if needed for LLM KD or visual KD
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
                    # Extract masked teacher logits and immediately free the large tensors
                    flat_teacher = teacher_logits_next[
                        mask
                    ]  # (valid_count, vocab_size)
                    masked_teacher_logits = flat_teacher.unsqueeze(
                        0
                    )  # (1, valid_count, vocab_size)

                    # Free memory: delete large intermediate tensors before JSD computation
                    # This frees ~1.16 GB per tensor (for seq_len=4096, vocab_size=151936)
                    del (
                        teacher_logits,
                        teacher_logits_next,
                        flat_teacher,
                        teacher_outputs,
                    )

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

        # Only compute visual KD loss if enabled (skip entirely if disabled)
        if self._visual_kd_enabled:
            vision_loss = self._compute_visual_kd_loss()
            if vision_loss is not None:
                total_loss = total_loss + vision_loss
                self._metrics[mode]["vision_kd_loss"].append(vision_loss.detach().cpu())
                breakdown = getattr(self, "_last_visual_kd_breakdown", None)
                if isinstance(breakdown, dict):
                    for target_name, loss_value in breakdown.items():
                        metric_name = f"vision_kd_loss_{target_name}"
                        self._metrics[mode][metric_name].append(
                            loss_value.detach().cpu()
                        )

        self._metrics[mode]["loss"].append(total_loss.detach().cpu())

        if valid_count > 0:
            logits_flat = masked_student_logits.squeeze(0)
            preds = logits_flat.argmax(dim=-1)
            accuracy = (preds == masked_labels).float().mean()
            correct_tokens = (preds == masked_labels).sum()
        else:
            accuracy = student_logits.new_zeros(())
            correct_tokens = student_logits.new_zeros(())

        total_tokens = torch.tensor(
            float(valid_count), device=student_logits.device, dtype=torch.float32
        )
        self._metrics[mode]["token_acc_correct"].append(correct_tokens.detach().cpu())
        self._metrics[mode]["token_acc_total"].append(total_tokens.detach().cpu())

        return (total_loss, outputs_student) if return_outputs else total_loss

    def _maybe_log_token_stats(
        self,
        *,
        valid_count: int,
        labels_next: torch.Tensor,
        mask: torch.Tensor,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> None:
        """Lightweight logging hook to trace token masking/sequence lengths."""
        # Lazily initialize debug file (stored outside checkpoints under /tmp by default)
        file_path: Optional[str] = getattr(self, "_token_stats_file", None)
        if file_path is False:
            return
        if file_path is None:
            file_path = (
                getattr(self.args, "token_stats_path", None)
                or "/tmp/gkd_token_stats.log"
            )
            # Type guard: ensure file_path is a string
            if not isinstance(file_path, str):
                self._token_stats_file = False
                return
            try:
                with open(file_path, "w", encoding="utf-8") as fout:
                    fout.write(
                        "# step valid_count seq_len mask_ratio input_ids attention_mask\n"
                    )
            except OSError:
                # Disable logging if file is unwritable
                self._token_stats_file = False
                return
            self._token_stats_file = file_path
            self._token_stats_logged = 0

        file_path = getattr(self, "_token_stats_file", None)
        if not file_path or file_path is False:
            return
        # Type guard: ensure file_path is a string
        if not isinstance(file_path, str):
            return

        # Limit logging to early steps or unusually large batches to avoid noise
        logged = getattr(self, "_token_stats_logged", 0)
        max_length = getattr(self.args, "max_length", None) or getattr(
            getattr(self.args, "training_args", None), "max_length", 0
        )
        seq_len = (
            int(labels_next.shape[1])
            if labels_next.ndim == 2
            else int(labels_next.numel())
        )
        mask_ratio = float(mask.float().mean().item()) if mask.numel() else 0.0
        # Enhanced logging: log early steps, large valid_count, or high mask_ratio (potential unmasking issue)
        should_log = (
            logged < 32
            or (max_length and valid_count > max_length)
            or mask_ratio > 0.5  # More than 50% unmasked is suspicious
            or valid_count > 2000  # Large valid_count might indicate unmasking issue
        )
        if not should_log:
            return

        step = getattr(getattr(self, "state", None), "global_step", None)
        input_ids = inputs.get("input_ids")
        attn = inputs.get("attention_mask")
        input_len = None
        attn_ratio = None
        if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
            input_len = int(input_ids.shape[1])
        if isinstance(attn, torch.Tensor):
            attn_ratio = float(attn.float().mean().item())

        try:
            with open(file_path, "a", encoding="utf-8") as fout:
                # Enhanced logging: include valid_count/seq_len ratio and label statistics
                valid_ratio = valid_count / seq_len if seq_len > 0 else 0.0
                labels_non_neg100 = int((labels_next != -100).sum().item())
                labels_total = int(labels_next.numel())
                fout.write(
                    f"{step if step is not None else 'NA'} "
                    f"{valid_count} {seq_len} {mask_ratio:.4f} "
                    f"{input_len if input_len is not None else 'NA'} "
                    f"{attn_ratio if attn_ratio is not None else 'NA'} "
                    f"valid_ratio={valid_ratio:.4f} labels_valid={labels_non_neg100}/{labels_total}\n"
                )
                # Also log diagnostic at debug level if mask_ratio is suspiciously high
                if mask_ratio > 0.5 or valid_count > 2000:
                    logger.debug(
                        f"Step {step}: High mask ratio detected! valid_count={valid_count}, "
                        f"seq_len={seq_len}, mask_ratio={mask_ratio:.4f}, valid_ratio={valid_ratio:.4f}. "
                        f"This might indicate unmasking issue (loss_scale='all' or template_backend='jinja')."
                    )
            self._token_stats_logged = logged + 1
        except OSError:
            # Disable logging on persistent failures
            self._token_stats_file = False

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
        # Ensure internal visual KD state and caches are initialized.
        self._ensure_visual_kd_state()

        student_unwrapped = self.accelerator.unwrap_model(self.model)
        teacher_model = getattr(self, "teacher_model", None)
        if teacher_model is None:
            raise RuntimeError("visual_kd requires a teacher model")
        teacher_unwrapped = self.accelerator.unwrap_model(teacher_model)

        student_visual = self._resolve_visual_branch(student_unwrapped)
        teacher_visual = self._resolve_visual_branch(teacher_unwrapped)

        if student_visual is None or teacher_visual is None:
            raise RuntimeError("Failed to resolve visual modules for visual KD")

        vit_cfg = self._visual_kd_vit_cfg
        aligner_cfg = self._visual_kd_aligner_cfg
        deepstack_cfg = self._visual_kd_deepstack_cfg

        # Shared merger module for ViT (pre-aligner) and aligner (post-merger) targets.
        if vit_cfg.enabled or aligner_cfg.enabled:
            student_merger = getattr(student_visual, "merger", None)
            teacher_merger = getattr(teacher_visual, "merger", None)
            if not isinstance(student_merger, nn.Module) or not isinstance(
                teacher_merger, nn.Module
            ):
                raise RuntimeError("merger not available for visual KD")

            if vit_cfg.enabled:

                def student_vit_hook(_module: nn.Module, inputs, _output):
                    if not inputs:
                        raise RuntimeError(
                            "ViT visual KD hook received no inputs for merger module"
                        )
                    hidden_states = inputs[0]
                    if not isinstance(hidden_states, torch.Tensor):
                        raise TypeError(
                            "Expected tensor input for ViT visual KD hook, "
                            f"got {type(hidden_states)}"
                        )
                    self._student_visual_cache["vit"] = hidden_states

                def teacher_vit_hook(_module: nn.Module, inputs, _output):
                    if not inputs:
                        raise RuntimeError(
                            "ViT visual KD hook received no inputs for merger module"
                        )
                    hidden_states = inputs[0]
                    if not isinstance(hidden_states, torch.Tensor):
                        raise TypeError(
                            "Expected tensor input for ViT visual KD hook, "
                            f"got {type(hidden_states)}"
                        )
                    self._teacher_visual_cache["vit"] = hidden_states.detach()

                self._visual_hooks.append(
                    student_merger.register_forward_hook(student_vit_hook)
                )
                self._visual_hooks.append(
                    teacher_merger.register_forward_hook(teacher_vit_hook)
                )

            if aligner_cfg.enabled:
                self._visual_hooks.append(
                    student_merger.register_forward_hook(
                        self._make_student_hook("aligner")
                    )
                )
                self._visual_hooks.append(
                    teacher_merger.register_forward_hook(
                        self._make_teacher_hook("aligner")
                    )
                )

        if deepstack_cfg.enabled:
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
            # Detach to prevent gradient flow through cached features
            # This prevents double gradient accumulation when the same tensor
            # participates in both CE loss and visual KD loss
            self._student_visual_cache[name] = output.detach()

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
        # Early return if visual_kd is disabled (should not be called, but defensive)
        if not self._visual_kd_enabled:
            return None

        # Ensure visual KD configs and caches are initialized before computing loss.
        self._ensure_visual_kd_state()

        def compute_distance(
            student_feat: torch.Tensor,
            teacher_feat: torch.Tensor,
            distance: str,
        ) -> torch.Tensor:
            teacher_feat = teacher_feat.to(student_feat.device, student_feat.dtype)
            if student_feat.shape != teacher_feat.shape:
                raise ValueError(
                    "Visual KD feature shape mismatch: "
                    f"student={student_feat.shape}, teacher={teacher_feat.shape}"
                )
            if distance == "mse":
                return F.mse_loss(student_feat, teacher_feat)
            if distance == "cosine":
                s_flat = student_feat.view(-1, student_feat.shape[-1])
                t_flat = teacher_feat.view(-1, teacher_feat.shape[-1])
                s_norm = F.normalize(s_flat, p=2.0, dim=-1, eps=1e-8)
                t_norm = F.normalize(t_flat, p=2.0, dim=-1, eps=1e-8)
                cosine = F.cosine_similarity(s_norm, t_norm, dim=-1)
                return (1 - cosine).mean()
            raise ValueError(f"Unsupported visual KD distance: {distance}")

        vit_cfg = self._visual_kd_vit_cfg
        aligner_cfg = self._visual_kd_aligner_cfg
        deepstack_cfg = self._visual_kd_deepstack_cfg

        contributions: list[torch.Tensor] = []
        missing_targets: list[str] = []
        breakdown: Dict[str, torch.Tensor] = {}

        if vit_cfg.enabled:
            student_feat = self._student_visual_cache.get("vit")
            teacher_feat = self._teacher_visual_cache.get("vit")
            if student_feat is None or teacher_feat is None:
                missing_targets.append("vit")
            else:
                base = compute_distance(student_feat, teacher_feat, vit_cfg.distance)
                component = vit_cfg.weight * base
                contributions.append(component)
                breakdown["vit"] = component

        if aligner_cfg.enabled:
            student_feat = self._student_visual_cache.get("aligner")
            teacher_feat = self._teacher_visual_cache.get("aligner")
            if student_feat is None or teacher_feat is None:
                missing_targets.append("aligner")
            else:
                base = compute_distance(
                    student_feat, teacher_feat, aligner_cfg.distance
                )
                component = aligner_cfg.weight * base
                contributions.append(component)
                breakdown["aligner"] = component

        if deepstack_cfg.enabled:
            keys = sorted(
                key
                for key in self._student_visual_cache.keys()
                if key.startswith("deepstack_")
            )
            if not keys:
                missing_targets.append("deepstack (no hooks fired)")
            else:
                layer_losses: list[torch.Tensor] = []
                missing_layers: list[str] = []
                for key in keys:
                    student_feat = self._student_visual_cache.get(key)
                    teacher_feat = self._teacher_visual_cache.get(key)
                    if student_feat is None or teacher_feat is None:
                        missing_layers.append(key)
                        continue
                    base = compute_distance(
                        student_feat, teacher_feat, deepstack_cfg.distance
                    )
                    layer_losses.append(base)
                if layer_losses:
                    stacked = torch.stack(
                        [loss_item.float() for loss_item in layer_losses]
                    ).mean()
                    component = deepstack_cfg.weight * stacked
                    contributions.append(component)
                    breakdown["deepstack"] = component
                else:
                    if missing_layers:
                        missing_targets.append(
                            "deepstack missing activations: "
                            + ", ".join(sorted(missing_layers))
                        )
                    else:
                        missing_targets.append("deepstack (no matching activations)")

        if not contributions:
            missing_desc = (
                ", ".join(missing_targets) if missing_targets else "all targets"
            )
            raise RuntimeError(
                "visual_kd did not receive any feature activations for the configured targets "
                f"{tuple(self._visual_kd_targets)} (missing {missing_desc}). "
                "Ensure the dataset yields image inputs and that both teacher and student expose the requested visual modules."
            )

        self._last_visual_kd_breakdown = breakdown

        total_loss = contributions[0]
        for loss in contributions[1:]:
            total_loss = total_loss + loss

        return total_loss

    def __del__(self):
        try:
            self._remove_visual_hooks()
        except Exception:
            pass

    def _ensure_visual_kd_state(self) -> None:
        if not hasattr(self, "_visual_kd_enabled"):
            self._visual_kd_enabled = False
        if not hasattr(self, "_visual_kd_config"):
            self._visual_kd_config = VisualKDConfig.disabled()

        if (
            not hasattr(self, "_visual_kd_vit_cfg")
            or not hasattr(self, "_visual_kd_aligner_cfg")
            or not hasattr(self, "_visual_kd_deepstack_cfg")
        ):
            cfg = self._visual_kd_config
            self._visual_kd_vit_cfg = getattr(cfg, "vit", VisualKDTargetConfig())
            self._visual_kd_aligner_cfg = getattr(
                cfg, "aligner", VisualKDTargetConfig()
            )
            self._visual_kd_deepstack_cfg = getattr(
                cfg, "deepstack", VisualKDTargetConfig()
            )

        if not hasattr(self, "_visual_kd_targets"):
            cfg = self._visual_kd_config
            self._visual_kd_targets = [
                name
                for name, target_cfg in (
                    ("vit", cfg.vit),
                    ("aligner", cfg.aligner),
                    ("deepstack", cfg.deepstack),
                )
                if target_cfg.enabled
            ]
        if not hasattr(self, "_visual_hooks"):
            self._visual_hooks = []
        if not hasattr(self, "_student_visual_cache"):
            self._student_visual_cache = {}
        if not hasattr(self, "_teacher_visual_cache"):
            self._teacher_visual_cache = {}
        if not hasattr(self, "_last_visual_kd_breakdown"):
            self._last_visual_kd_breakdown = {}

    def _has_teacher_model(self) -> bool:
        return hasattr(self, "teacher_model") and self.teacher_model is not None

    def log(
        self, logs: Optional[Dict[str, float]], start_time: Optional[float] = None
    ) -> None:
        if logs is None:
            logs = {}

        sanitized_logs: Dict[str, float] = dict(logs)
        aggregated_logs: Dict[str, float] = {}
        keys_to_remove: set[str] = set()

        for mode in ("train", "eval"):
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

                metric_key = key if mode == "train" else f"eval_{key}"
                base_key = self._metric_base_key(metric_key)
                aggregate_as_sum = base_key in {
                    "token_acc_correct",
                    "token_acc_total",
                }

                if aggregate_as_sum:
                    aggregated = float(finite_values.sum().item())
                    reduction = "sum"
                else:
                    aggregated = float(finite_values.mean().item())
                    reduction = "mean"

                aggregated = self._sync_metric_value(aggregated, reduction=reduction)
                aggregated_logs[metric_key] = aggregated
                keys_to_remove.add(metric_key)

            self._metrics[mode].clear()

        for key in keys_to_remove:
            logs.pop(key, None)
            sanitized_logs.pop(key, None)

        self._coalesce_accuracy_metrics(aggregated_logs)

        sanitized_logs.update(aggregated_logs)
        logs.update(aggregated_logs)

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(sanitized_logs, start_time)
        else:
            super().log(sanitized_logs)

    @staticmethod
    def _metric_base_key(metric_key: str) -> str:
        if metric_key.startswith("eval_"):
            return metric_key[5:]
        return metric_key

    @staticmethod
    def _coalesce_accuracy_metrics(metrics: Dict[str, float]) -> None:
        """Convert *_token_acc_{correct,total} into *_token_acc ratios."""

        for prefix in ("", "eval_"):
            correct_key = f"{prefix}token_acc_correct"
            total_key = f"{prefix}token_acc_total"
            correct = metrics.pop(correct_key, None)
            total = metrics.pop(total_key, None)
            if correct is None or total is None:
                continue
            if total <= 0:
                continue
            metrics[f"{prefix}token_acc"] = correct / total

    def _sync_metric_value(self, value: float, *, reduction: str) -> float:
        """Synchronize metric scalars across ranks."""

        if not (dist.is_available() and dist.is_initialized()):
            return value

        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        world_size = dist.get_world_size()
        if world_size <= 1:
            return value

        tensor = torch.tensor([value], dtype=torch.float64, device=device)
        dist.all_reduce(tensor)
        if reduction == "mean":
            tensor /= world_size
        return float(tensor.item())
