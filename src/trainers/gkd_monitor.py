"""Custom GKD trainer wrapper that records KL and token accuracy metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from packaging import version

from swift.trainers.rlhf_trainer.gkd_trainer import GKDTrainer as _MsSwiftGKDTrainer


class GKDTrainerWithMetrics(_MsSwiftGKDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure metrics dict exists even if parent implementation changes.
        if not hasattr(self, "_metrics"):
            self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        model_inputs = {
            k: v for k, v in inputs.items() if k not in {"prompt", "labels"}
        }
        use_logits_to_keep = self.get_use_logits_to_keep(True)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(inputs)
            model_inputs["logits_to_keep"] = inputs["logits_to_keep"]
        if self.args.sft_alpha > 0:
            model_inputs["labels"] = inputs["labels"]

        outputs_student = model(**model_inputs)

        model_inputs.pop("labels", None)
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**model_inputs)

        shifted_labels = torch.roll(inputs["labels"], shifts=-1, dims=1)
        mask = shifted_labels != -100
        shifted_student_logits = outputs_student.logits[mask][None]
        shifted_teacher_logits = outputs_teacher.logits[mask][None]

        stu_dim = shifted_student_logits.shape[-1]
        tea_dim = shifted_teacher_logits.shape[-1]
        if stu_dim < tea_dim:
            shifted_student_logits = F.pad(
                shifted_student_logits, (0, tea_dim - stu_dim), "constant", 0
            )
            shifted_student_logits[..., stu_dim:] = shifted_teacher_logits[
                ..., stu_dim:
            ]
        elif stu_dim > tea_dim:
            shifted_teacher_logits = F.pad(
                shifted_teacher_logits, (0, stu_dim - tea_dim), "constant", 0
            )
            shifted_teacher_logits[..., tea_dim:] = shifted_student_logits[
                ..., tea_dim:
            ]

        kl_loss = self.generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            beta=self.beta,
        )
        total_loss = kl_loss
        ce_loss = None
        if getattr(outputs_student, "loss", None) is not None:
            ce_loss = outputs_student.loss
            if self.args.sft_alpha > 0:
                total_loss = total_loss + self.args.sft_alpha * ce_loss

        valid_count = int(mask.sum().item())
        mode = "train" if model.training else "eval"
        self._metrics[mode]["kl_loss"].append(float(kl_loss.detach().item()))
        self._metrics[mode]["token_count"].append(valid_count)
        self._metrics[mode]["loss"].append(float(total_loss.detach().item()))
        if ce_loss is not None:
            self._metrics[mode]["sft_loss"].append(float(ce_loss.detach().item()))
        else:
            self._metrics[mode]["sft_loss"].append(0.0)

        if valid_count > 0:
            logits_flat = shifted_student_logits.squeeze(0)
            labels_flat = shifted_labels[mask]
            preds = logits_flat.argmax(dim=-1)
            accuracy = (preds == labels_flat).float().mean().item()
            self._metrics[mode]["token_accuracy"].append(accuracy)
        else:
            self._metrics[mode]["token_accuracy"].append(0.0)

        return (total_loss, outputs_student) if return_outputs else total_loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics_data = {
            key: list(values) for key, values in self._metrics[mode].items()
        }
        counts = metrics_data.pop("token_count", [])
        total_weight = sum(counts)
        prefix = "train/" if mode == "train" else "eval/"

        for key, values in metrics_data.items():
            if not values:
                continue
            if total_weight > 0 and len(values) == len(counts):
                weighted = sum(v * c for v, c in zip(values, counts)) / total_weight
            else:
                weighted = sum(values) / len(values)
            logs[f"{prefix}{key}"] = weighted
            if key == "loss" and mode == "train":
                logs["loss"] = weighted

        if counts:
            logs[f"{prefix}token_count"] = total_weight / len(counts)

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics[mode].clear()
