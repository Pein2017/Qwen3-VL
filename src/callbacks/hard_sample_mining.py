from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from ..config.schema import HardSampleMiningConfig


@dataclass
class _SampleStat:
    count: int = 0
    ema: float = 0.0
    mean_sum: float = 0.0

    def update(self, value: float, ema_decay: float) -> None:
        self.count += 1
        if self.count == 1:
            self.ema = value
        else:
            self.ema = ema_decay * self.ema + (1 - ema_decay) * value
        self.mean_sum += value

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.mean_sum / self.count

    def score(self, ema_decay: float) -> float:
        if self.count < 3:
            return self.mean
        return self.ema


class HardSampleTracker:
    def __init__(self, ema_decay: float) -> None:
        self.ema_decay = float(ema_decay)
        self.stats: Dict[int, _SampleStat] = {}
        self.meta: Dict[int, Tuple[str, int]] = {}

    def reset(self) -> None:
        self.stats.clear()
        self.meta.clear()

    def update(self, sample_ids: Iterable[int], losses: Iterable[float], datasets: Iterable[str], base_idxs: Iterable[int]) -> None:
        for sid, loss, ds, bidx in zip(sample_ids, losses, datasets, base_idxs):
            try:
                sid_int = int(sid)
            except Exception:
                continue
            stat = self.stats.get(sid_int)
            if stat is None:
                stat = _SampleStat()
                self.stats[sid_int] = stat
            stat.update(float(loss), self.ema_decay)
            if sid_int not in self.meta:
                self.meta[sid_int] = (str(ds), int(bidx))

    def scores_for_dataset(self, dataset_name: str) -> List[Tuple[int, float, int]]:
        out: List[Tuple[int, float, int]] = []
        for sid, stat in self.stats.items():
            meta = self.meta.get(sid)
            if meta is None:
                continue
            ds, base_idx = meta
            if ds != dataset_name:
                continue
            out.append((sid, stat.score(self.ema_decay), base_idx))
        return out


def _gather_tensor(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return t
    world = dist.get_world_size()
    tensors = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(tensors, t)
    return torch.cat(tensors, dim=0)


def _per_sample_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: [B, L, V], labels: [B, L]
    logits = logits.float()
    vocab = logits.size(-1)
    loss_flat = nn.functional.cross_entropy(
        logits.view(-1, vocab), labels.view(-1), ignore_index=-100, reduction="none"
    )
    loss_tok = loss_flat.view(labels.shape)
    mask = (labels != -100).float()
    denom = mask.sum(dim=1)
    denom = torch.clamp(denom, min=1.0)
    per_sample = (loss_tok * mask).sum(dim=1) / denom
    return per_sample


def attach_hsm_compute_loss(trainer, tracker: HardSampleTracker, hsm_cfg: HardSampleMiningConfig):
    orig_compute_loss = trainer.compute_loss

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = orig_compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        if hsm_cfg.enabled and tracker is not None:
            labels = inputs.get("labels")
            sample_ids = inputs.get("sample_id")
            datasets = inputs.get("dataset")
            base_idxs = inputs.get("base_idx")
            if labels is not None and sample_ids is not None and datasets is not None and base_idxs is not None:
                try:
                    logits = outputs.logits if hasattr(outputs, "logits") else None
                except Exception:
                    logits = None
                if logits is not None:
                    with torch.no_grad():
                        per_sample = _per_sample_loss(logits, labels)
                        sid_t = torch.as_tensor(sample_ids, device=per_sample.device, dtype=torch.long)
                        loss_t = per_sample.detach()
                        base_t = torch.as_tensor(base_idxs, device=per_sample.device, dtype=torch.long)
                        # datasets may be list[str]; keep local, not gathered
                        sid_all = _gather_tensor(sid_t)
                        loss_all = _gather_tensor(loss_t)
                        base_all = _gather_tensor(base_t)
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            tracker.update(
                                sid_all.cpu().tolist(),
                                loss_all.cpu().tolist(),
                                [str(d) for d in (datasets if not dist.is_initialized() else [datasets[0]] * len(sid_all))],
                                base_all.cpu().tolist(),
                            )
        return (loss, outputs) if return_outputs else loss

    trainer.compute_loss = types.MethodType(_compute_loss, trainer)


class HardSampleMiningCallback(TrainerCallback):
    def __init__(
        self,
        *,
        tracker: HardSampleTracker,
        config: HardSampleMiningConfig,
        dataset: Any,
        target_dataset: str,
        trainer_ref: Any | None = None,
    ) -> None:
        self.tracker = tracker
        self.config = config
        self.dataset = dataset
        self.target_dataset = target_dataset
        self.trainer = trainer_ref

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        self.tracker.reset()

    def _should_trigger(self, epoch: int) -> bool:
        return self.config.enabled and epoch >= self.config.start_epoch

    def _build_plan(self, target_scores: List[Tuple[int, float, int]]) -> Dict[str, Any]:
        if not target_scores:
            return {}
        target_scores.sort(key=lambda x: x[1], reverse=True)
        hard_k = max(1, self.config.hard_sample_size)
        hard = target_scores[:hard_k]
        reg_pool = target_scores[hard_k:]
        reg_k = max(0, self.config.regular_sample_size)
        if reg_pool:
            reg_selected = reg_pool[:reg_k] if len(reg_pool) >= reg_k else reg_pool + reg_pool[: max(0, reg_k - len(reg_pool))]
        else:
            reg_selected = []
        hard_ids = {b for (_, _, b) in hard}
        reg_ids = [b for (_, _, b) in reg_selected]
        weights: Dict[int, float] = {}
        for b in hard_ids:
            weights[b] = 1.0
        for b in reg_ids:
            weights[b] = weights.get(b, 0.0) + 1.0
        target_epoch_size = len(hard_ids) + len(reg_ids)
        if target_epoch_size == 0:
            target_epoch_size = max(1, self.config.hard_sample_size + self.config.regular_sample_size)
        return {
            "weights": weights,
            "target_epoch_size": target_epoch_size,
            "hard": list(hard_ids),
        }

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        epoch = int(state.epoch) if state.epoch is not None else 0
        if not self._should_trigger(epoch):
            return
        trainer = self.trainer
        if trainer is None:
            return
        # use collected train losses only; no extra full-pass
        scores = self.tracker.scores_for_dataset(self.target_dataset)
        plan = self._build_plan(scores)
        if hasattr(self.dataset, "set_hard_sample_plan"):
            self.dataset.set_hard_sample_plan(plan)
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            num_hard = len(plan.get("hard", [])) if plan else 0
            trainer.log({
                "hsm/triggered": 1.0,
                "hsm/num_hard": float(num_hard),
                "hsm/target_epoch_size": float(plan.get("target_epoch_size") or 0),
            })
