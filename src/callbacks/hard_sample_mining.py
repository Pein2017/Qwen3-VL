from __future__ import annotations

import math
import random
import types
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from ..config.schema import HardSampleMiningConfig


def _gather_tensor(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return t
    out_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, t)
    return torch.cat(out_list, dim=0)


def _gather_list(values: list[Any]) -> list[Any]:
    if not dist.is_available() or not dist.is_initialized():
        return list(values)
    gathered: list[list[Any] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, list(values))
    merged: list[Any] = []
    for part in gathered:
        if part:
            merged.extend(part)
    return merged


def _per_sample_token_acc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: [B, L, V], labels: [B, L]
    logits = logits.float()
    preds = logits.argmax(dim=-1)
    mask = (labels != -100).float()
    correct = (preds == labels).float() * mask
    denom = torch.clamp(mask.sum(dim=1), min=1.0)
    acc = correct.sum(dim=1) / denom
    return acc


@dataclass
class _SampleStat:
    count: int = 0
    ema: float = 0.0
    sum: float = 0.0

    def update(self, value: float, ema_decay: float) -> None:
        self.count += 1
        if self.count == 1:
            self.ema = value
        else:
            self.ema = ema_decay * self.ema + (1 - ema_decay) * value
        self.sum += value

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def difficulty(self, ema_decay: float) -> float:
        # Lower token_acc => higher difficulty; map to 1 - acc
        if self.count < 3:
            acc = self.mean
        else:
            acc = self.ema
        return 1.0 - acc


class HardSampleTracker:
    def __init__(self, ema_decay: float) -> None:
        self.ema_decay = float(ema_decay)
        self.stats: Dict[int, _SampleStat] = {}
        self.meta: Dict[int, Tuple[str, int]] = {}

        self.epoch_hard_acc: list[float] = []
        self.epoch_reg_acc: list[float] = []
        self.epoch_hard_seen: set[int] = set()
        self.epoch_reg_seen: set[int] = set()
        self.active_dataset: str | None = None
        self.active_hard: set[int] = set()

    def reset_epoch(self, dataset: str | None, hard_base_idxs: Iterable[int] | None) -> None:
        self.epoch_hard_acc = []
        self.epoch_reg_acc = []
        self.epoch_hard_seen = set()
        self.epoch_reg_seen = set()
        self.active_dataset = dataset
        self.active_hard = set(int(b) for b in (hard_base_idxs or []))

    def update(self, sample_ids: Iterable[int], accs: Iterable[float], datasets: Iterable[str], base_idxs: Iterable[int]) -> None:
        for sid, acc, ds, bidx in zip(sample_ids, accs, datasets, base_idxs):
            try:
                sid_int = int(sid)
            except Exception:
                continue
            stat = self.stats.get(sid_int)
            if stat is None:
                stat = _SampleStat()
                self.stats[sid_int] = stat
            stat.update(float(acc), self.ema_decay)
            if sid_int not in self.meta:
                self.meta[sid_int] = (str(ds), int(bidx))

            if self.active_dataset is not None and str(ds) == self.active_dataset:
                try:
                    b_int = int(bidx)
                except Exception:
                    b_int = None
                if b_int is not None and b_int in self.active_hard:
                    self.epoch_hard_acc.append(float(acc))
                    self.epoch_hard_seen.add(b_int)
                else:
                    self.epoch_reg_acc.append(float(acc))
                    if b_int is not None:
                        self.epoch_reg_seen.add(b_int)

    def scores_for_dataset(self, dataset_name: str) -> List[Tuple[int, float, int]]:
        out: List[Tuple[int, float, int]] = []
        for sid, stat in self.stats.items():
            meta = self.meta.get(sid)
            if meta is None:
                continue
            ds, base_idx = meta
            if ds != dataset_name:
                continue
            out.append((sid, stat.difficulty(self.ema_decay), base_idx))
        return out


def attach_hsm_compute_metrics(trainer, tracker: HardSampleTracker, hsm_cfg: HardSampleMiningConfig):
    orig_compute_loss = trainer.compute_loss

    def _to_list(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        return list(x)

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = orig_compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        if hsm_cfg.enabled and tracker is not None:
            labels = inputs.get("labels")
            sample_ids = inputs.get("sample_id")
            datasets = inputs.get("dataset")
            base_idxs = inputs.get("base_idx")
            logits = getattr(outputs, "logits", None)
            if logits is not None and labels is not None and sample_ids is not None and datasets is not None and base_idxs is not None:
                with torch.no_grad():
                    acc_per = _per_sample_token_acc(logits, labels).detach()
                    acc_list = _to_list(acc_per)
                    sid_list = _to_list(sample_ids)
                    base_list = _to_list(base_idxs)
                    ds_list = [str(x) for x in _to_list(datasets)]

                    acc_all = _gather_list(acc_list)
                    sid_all = _gather_list(sid_list)
                    base_all = _gather_list(base_list)
                    ds_all = _gather_list(ds_list)

                    if (not dist.is_initialized()) or dist.get_rank() == 0:
                        tracker.update(sid_all, acc_all, ds_all, base_all)
        return (loss, outputs) if return_outputs else loss

    trainer.compute_loss = types.MethodType(_compute_loss, trainer)


class HardSampleDynamicCallback(TrainerCallback):
    def __init__(
        self,
        *,
        tracker: HardSampleTracker,
        config: HardSampleMiningConfig,
        dataset: Any,
        target_dataset: str,
    ) -> None:
        self.tracker = tracker
        self.config = config
        self.dataset = dataset
        self.target_dataset = target_dataset
        self.total_epochs: int = 0
        self.batch_size: int = 1
        self.trainer = None

    # Utility helpers
    def _get_target_size(self) -> int:
        if hasattr(self.dataset, "_record_pools") and hasattr(self.dataset, "_target_name"):
            pools = getattr(self.dataset, "_record_pools")
            tgt = getattr(self.dataset, "_target_name")
            if isinstance(pools, Mapping) and tgt in pools:
                return len(pools[tgt])
        if hasattr(self.dataset, "base_records"):
            try:
                return len(getattr(self.dataset, "base_records"))
            except Exception:
                pass
        try:
            return len(self.dataset)
        except Exception:
            return 0

    def _get_source_indices(self) -> list[Tuple[str, int]]:
        indices: list[Tuple[str, int]] = []
        if hasattr(self.dataset, "_record_pools") and hasattr(self.dataset, "_target_name"):
            pools = getattr(self.dataset, "_record_pools")
            tgt = getattr(self.dataset, "_target_name")
            for name, pool in pools.items():
                if name == tgt:
                    continue
                for idx in range(len(pool)):
                    indices.append((name, idx))
        return indices

    def _mining_active(self, epoch_idx: int) -> bool:
        if self.total_epochs <= 0:
            return False
        progress = (epoch_idx + 1) / self.total_epochs
        return progress >= self.config.activate_after_pct and epoch_idx >= self.config.start_epoch

    def _select_hard_pool(self, scores: List[Tuple[int, float, int]], target_size: int) -> list[int]:
        if target_size <= 0:
            return []
        if not scores:
            return []
        # Sort by difficulty desc, tie-break by sample_id asc for determinism
        scores.sort(key=lambda x: (-x[1], x[0]))
        if self.config.hard_pool_k is not None:
            k = min(max(1, int(self.config.hard_pool_k)), len(scores))
        else:
            k = max(1, math.ceil(target_size * self.config.hard_pool_frac))
        return [bidx for (_sid, _diff, bidx) in scores[:k]]

    def _build_schedule_fusion(
        self,
        *,
        epoch: int,
        hard_pool: list[int],
        target_size: int,
        source_entries: list[Tuple[str, int]],
        mining_active: bool,
    ) -> list[Tuple[str, int]]:
        rng = random.Random((epoch + 1) * 0x9E3779B1)
        schedule: list[Tuple[str, int]] = []

        # Target portion
        if target_size > 0:
            if mining_active and hard_pool:
                hard_count = math.ceil(target_size * self.config.hard_pool_frac)
                hard_samples = [rng.choice(hard_pool) for _ in range(hard_count)]
                reg_count = max(0, target_size - hard_count)
                target_all = list(range(target_size))
                reg_samples = [rng.choice(target_all) for _ in range(reg_count)]
                target_indices = hard_samples + reg_samples
            else:
                target_indices = list(range(target_size))
                if len(target_indices) > 1:
                    rng.shuffle(target_indices)
            schedule.extend((self.target_dataset, idx) for idx in target_indices)

        # Source portion: configurable percentage of target size
        source_quota = 0
        if source_entries and target_size > 0:
            source_quota = max(0, round(self.config.source_ratio * target_size))
        for _ in range(source_quota):
            schedule.append(rng.choice(source_entries))

        # Final shuffle for intermixing target/source
        if len(schedule) > 1:
            rng.shuffle(schedule)
        return schedule

    def _build_schedule_single(
        self,
        *,
        epoch: int,
        hard_pool: list[int],
        target_size: int,
        mining_active: bool,
    ) -> list[int]:
        rng = random.Random((epoch + 1) * 0x9E3779B1)
        if target_size <= 0:
            return []
        if mining_active and hard_pool:
            hard_count = math.ceil(target_size * self.config.hard_pool_frac)
            hard_samples = [rng.choice(hard_pool) for _ in range(hard_count)]
            reg_count = max(0, target_size - hard_count)
            all_indices = list(range(target_size))
            reg_samples = [rng.choice(all_indices) for _ in range(reg_count)]
            perm = hard_samples + reg_samples
        else:
            perm = list(range(target_size))
            if len(perm) > 1:
                rng.shuffle(perm)
        return perm

    def _log_pool_metrics(self, trainer) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        def _mean(vals: list[float]) -> float:
            return float(sum(vals) / len(vals)) if vals else 0.0

        def _p90(vals: list[float]) -> float:
            if not vals:
                return 0.0
            vs = sorted(vals)
            idx = max(0, min(len(vs) - 1, math.ceil(0.9 * len(vs)) - 1))
            return float(vs[idx])

        prefix = self.config.log_prefix or "hsm"
        hard_acc = self.tracker.epoch_hard_acc
        reg_acc = self.tracker.epoch_reg_acc
        hard_seen = len(hard_acc)
        reg_seen = len(reg_acc)
        total_seen = hard_seen + reg_seen
        hard_hit_rate = hard_seen / total_seen if total_seen > 0 else 0.0
        hard_pool_size = len(self.tracker.active_hard)
        hard_cov = (
            len(self.tracker.epoch_hard_seen) / hard_pool_size
            if hard_pool_size > 0
            else 0.0
        )

        if trainer is not None:
            trainer.log(
                {
                    f"{prefix}/train_acc_hard_mean": _mean(hard_acc),
                    f"{prefix}/train_acc_hard_p90": _p90(hard_acc),
                    f"{prefix}/train_acc_regular_mean": _mean(reg_acc),
                    f"{prefix}/train_acc_regular_p90": _p90(reg_acc),
                    f"{prefix}/hard_seen": float(hard_seen),
                    f"{prefix}/regular_seen": float(reg_seen),
                    f"{prefix}/hard_hit_rate": float(hard_hit_rate),
                    f"{prefix}/hard_pool_coverage": float(hard_cov),
                }
            )

    # Trainer callbacks
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        self.total_epochs = int(getattr(args, "num_train_epochs", 0) or 0)
        self.batch_size = int(getattr(args, "per_device_train_batch_size", 1) or 1)
        self.tracker.reset_epoch(self.target_dataset, [])

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        epoch = int(state.epoch) if state.epoch is not None else 0
        mining_active = self.config.enabled and self._mining_active(epoch)
        is_primary = (not dist.is_initialized()) or dist.get_rank() == 0

        target_size = self._get_target_size()
        if is_primary:
            scores = self.tracker.scores_for_dataset(self.target_dataset)
            hard_pool = self._select_hard_pool(scores, target_size) if mining_active else []
            source_entries = self._get_source_indices()
            if hasattr(self.dataset, "_record_pools") and hasattr(self.dataset, "_target_name"):
                schedule = self._build_schedule_fusion(
                    epoch=epoch,
                    hard_pool=hard_pool,
                    target_size=target_size,
                    source_entries=source_entries,
                    mining_active=mining_active,
                )
            else:
                schedule = self._build_schedule_single(
                    epoch=epoch,
                    hard_pool=hard_pool,
                    target_size=target_size,
                    mining_active=mining_active,
                )
        else:
            hard_pool = []
            schedule = []

        # Broadcast plan to all ranks for determinism
        if dist.is_initialized():
            payload = [{"hard_pool": hard_pool, "schedule": schedule, "target_size": target_size, "mining_active": mining_active}]
            dist.broadcast_object_list(payload, src=0)
            hard_pool = payload[0]["hard_pool"]
            schedule = payload[0]["schedule"]
            target_size = payload[0]["target_size"]
            mining_active = payload[0]["mining_active"]

        # Apply schedule to dataset (all ranks)
        if hasattr(self.dataset, "set_external_hsm_schedule"):
            try:
                self.dataset.set_external_hsm_schedule(schedule)
            except Exception:
                pass

        # Log metrics and plan summary (rank0)
        if self.config.log_pool_metrics:
            self._log_pool_metrics(self.trainer)

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            prefix = self.config.log_prefix or "hsm"
            if self.trainer is not None:
                self.trainer.log(
                    {
                        f"{prefix}/hard_pool_size": float(len(hard_pool)),
                        f"{prefix}/target_size": float(target_size),
                        f"{prefix}/schedule_len": float(len(schedule)),
                        f"{prefix}/mining_active": 1.0 if mining_active else 0.0,
                    }
                )

        # Prepare buckets for next epoch using the selected hard pool
        self.tracker.reset_epoch(self.target_dataset, hard_pool)


def build_hsm_dataloader(trainer, dataset, hsm_enabled: bool) -> DataLoader:
    args = trainer.args
    if not hsm_enabled:
        return trainer._get_train_dataloader_default()

    # Use deterministic sampler; no internal shuffle so schedule order is honored
    if args.world_size and args.world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = SequentialSampler(dataset)

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=trainer.data_collator,
        drop_last=args.dataloader_drop_last,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        persistent_workers=args.dataloader_persistent_workers,
    )
    if args.dataloader_prefetch_factor is not None:
        loader_kwargs[\"prefetch_factor\"] = args.dataloader_prefetch_factor
    return DataLoader(**loader_kwargs)
