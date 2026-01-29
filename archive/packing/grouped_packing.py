from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Sized,
)

import logging
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

if TYPE_CHECKING:
    from typing import Protocol

    class HasSetEpoch(Protocol):
        def set_epoch(self, epoch: int) -> None: ...


EncodedRow = Mapping[str, Any]
GroupKeyFn = Callable[[EncodedRow], str]
MetaResolver = Callable[[EncodedRow], Mapping[str, Any]]


def _resolve_length(row: EncodedRow) -> int:
    """Return sequence length for an encoded row."""
    if "length" in row and row["length"] is not None:
        try:
            return int(row["length"])
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid length value on row: {row['length']!r}") from exc
    input_ids = row.get("input_ids")
    if input_ids is None:
        raise ValueError("Packed row must include 'length' or 'input_ids'")
    return len(input_ids)


def default_group_key_fn(row: EncodedRow) -> str:
    """Fallback grouping: single global bucket."""

    return "default"


@dataclass(frozen=True)
class _Pack:
    indices: List[int]
    group: str
    lengths: List[int]

    @property
    def total_length(self) -> int:
        return sum(self.lengths)


class GroupedPackingDataset(Dataset):
    """Order-preserving grouped packing wrapper for map-style datasets.

    Packs sequences only within the same group key and keeps partial packs.
    """

    def __init__(
        self,
        dataset: Dataset,
        template: Any,
        *,
        group_key_fn: GroupKeyFn = default_group_key_fn,
        meta_resolver: Optional[MetaResolver] = None,
        packing_length: Optional[int] = None,
        cached_lengths: Optional[Mapping[int, int]] = None,
        cached_lengths_fail_on_miss: bool = True,
    ) -> None:
        if packing_length is None:
            packing_length = getattr(template, "max_length", None)
        if packing_length is None:
            raise ValueError(
                "packing_length is required when template.max_length is None"
            )
        self.dataset = dataset
        self.template = template
        self.group_key_fn = group_key_fn
        self.meta_resolver = meta_resolver
        self.packing_length = int(packing_length)
        if self.packing_length <= 0:
            raise ValueError("packing_length must be positive")

        self._epoch = 0
        self._packs: List[_Pack] = []
        # Optional mapping for downstream metrics (e.g., target/source)
        self.group_domains: Dict[str, str] = {}
        self._cached_lengths = cached_lengths
        self._cached_lengths_fail_on_miss = bool(cached_lengths_fail_on_miss)

        if self._in_distributed():
            if self._is_rank_zero():
                self._build_packs()
            self._broadcast_packs()
            self._reset_dataset_epoch()
        else:
            self._build_packs()

    @staticmethod
    def _in_distributed() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _is_rank_zero() -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return True
        return dist.get_rank() == 0

    def _reset_dataset_epoch(self) -> None:
        if hasattr(self.dataset, "set_epoch"):
            try:
                self.dataset.set_epoch(self._epoch)  # type: ignore[attr-defined]
            except Exception:
                # Non-fatal: dataset may not expose epoch resetting
                pass

    def _build_packs(self) -> None:
        data: List[tuple[int, int, str, Optional[str]]] = []
        self.group_domains = {}
        dataset_len = len(self.dataset) if isinstance(self.dataset, Sized) else 0  # type: ignore[arg-type]
        groups_seen: Dict[str, int] = {}
        for idx in range(dataset_len):
            row = (
                self._get_row_with_cached_length(idx)
                if self._cached_lengths is not None
                else self.dataset[idx]
            )
            length = _resolve_length(row)
            group = self.group_key_fn(row)
            groups_seen[group] = groups_seen.get(group, 0) + 1
            domain: Optional[str] = None
            if self.meta_resolver is not None:
                meta = self.meta_resolver(row) or {}
                domain = meta.get("domain")
            # Also try to get domain directly from metadata if meta_resolver didn't find it
            if domain is None and isinstance(row, Mapping):
                row_meta = row.get("metadata", {})
                if isinstance(row_meta, Mapping):
                    domain = row_meta.get("_fusion_domain")
            if domain is not None:
                self.group_domains.setdefault(str(group), str(domain))
            data.append((idx, length, str(group), domain))

        # Debug: log groups seen during packing
        try:
            logger = logging.getLogger(__name__)
            logger.debug(
                f"GroupedPackingDataset: groups resolved during packing: {groups_seen}"
            )
            logger.debug(
                f"GroupedPackingDataset: group_domains mapping: {self.group_domains}"
            )
        except Exception:
            pass

        self._packs = self._pack_sequential(data)
        # Reset base dataset epoch to keep augmentation RNG aligned with training access
        self._reset_dataset_epoch()

        try:
            logger = logging.getLogger(__name__)
            logger.debug(
                "GroupedPackingDataset: built %d packs on rank %s",
                len(self._packs),
                dist.get_rank() if self._in_distributed() else "local",
            )
        except Exception:
            pass

    def _get_row_with_cached_length(self, idx: int) -> EncodedRow:
        """Return a lightweight row with cached length and metadata, avoiding augmentation/encode."""
        # FusionCaptionDataset path
        if hasattr(self.dataset, "_schedule") and hasattr(
            self.dataset, "_record_pools"
        ):
            try:
                from src.datasets.dense_caption import BaseCaptionDataset
            except Exception:
                BaseCaptionDataset = None  # type: ignore

            try:
                dataset_name, base_idx = self.dataset._schedule[idx]
                record_pool = getattr(self.dataset, "_record_pools", {})
                base_pool = (
                    record_pool.get(dataset_name)
                    if isinstance(record_pool, Mapping)
                    else None
                )
                record = (
                    base_pool[base_idx]
                    if base_pool is not None and 0 <= base_idx < len(base_pool)
                    else None
                )
                if record is not None:
                    sample_id = (
                        BaseCaptionDataset._make_sample_id(dataset_name, base_idx)  # type: ignore[attr-defined]
                        if BaseCaptionDataset is not None
                        else base_idx
                    )
                    if sample_id in self._cached_lengths:
                        row = dict(record)
                        row["length"] = int(self._cached_lengths[sample_id])
                        return row
                    if self._cached_lengths_fail_on_miss:
                        raise KeyError(sample_id)
            except Exception:
                if self._cached_lengths_fail_on_miss:
                    raise
                # fall through to standard path

        # BaseCaptionDataset path (non-fusion)
        if hasattr(self.dataset, "base_records") and hasattr(
            self.dataset, "_index_perm"
        ):
            try:
                from src.datasets.dense_caption import BaseCaptionDataset
            except Exception:
                BaseCaptionDataset = None  # type: ignore
            try:
                perm = getattr(self.dataset, "_index_perm", [])
                if idx < len(perm):
                    base_idx = perm[idx]
                else:
                    base_idx = idx
                base_records = getattr(self.dataset, "base_records", [])
                record = (
                    base_records[base_idx]
                    if isinstance(base_records, list)
                    and 0 <= base_idx < len(base_records)
                    else None
                )
                dataset_name = getattr(self.dataset, "dataset_name", "dataset")
                if record is not None:
                    sample_id = (
                        BaseCaptionDataset._make_sample_id(dataset_name, base_idx)  # type: ignore[attr-defined]
                        if BaseCaptionDataset is not None
                        else base_idx
                    )
                    if sample_id in self._cached_lengths:
                        row = dict(record)
                        row["length"] = int(self._cached_lengths[sample_id])
                        return row
                    if self._cached_lengths_fail_on_miss:
                        raise KeyError(sample_id)
            except Exception:
                if self._cached_lengths_fail_on_miss:
                    raise
                # fall through

        # Fallback: use full dataset access
        return self.dataset[idx]

    def _pack_sequential(
        self, items: Sequence[tuple[int, int, str, Optional[str]]]
    ) -> List[_Pack]:
        packs: List[_Pack] = []
        current_indices: List[int] = []
        current_lengths: List[int] = []
        current_group: Optional[str] = None

        def _flush():
            if current_indices:
                packs.append(
                    _Pack(
                        indices=list(current_indices),
                        group=current_group or "default",
                        lengths=list(current_lengths),
                    )
                )

        for idx, length, group, _domain in items:
            # Long sequence: emit as single-item pack
            if length >= self.packing_length:
                _flush()
                packs.append(_Pack(indices=[idx], group=group, lengths=[length]))
                current_indices, current_lengths, current_group = [], [], None
                continue

            if (
                current_group is None
                or group != current_group
                or (sum(current_lengths) + length) > self.packing_length
            ):
                _flush()
                current_indices = [idx]
                current_lengths = [length]
                current_group = group
            else:
                current_indices.append(idx)
                current_lengths.append(length)

        _flush()
        return packs

    def _broadcast_packs(self) -> None:
        """Broadcast pack metadata from rank0 to all ranks in distributed mode."""

        if not self._in_distributed():
            return

        is_rank0 = self._is_rank_zero()
        packs_data = (
            [(list(p.indices), p.group, list(p.lengths)) for p in self._packs]
            if is_rank0
            else None
        )
        domains_data = dict(self.group_domains) if is_rank0 else None

        obj_list: list[Any] = [packs_data, domains_data]
        dist.broadcast_object_list(obj_list, src=0)
        packs_recv, domains_recv = obj_list

        if not is_rank0:
            self._packs = [
                _Pack(indices=list(idx), group=grp, lengths=list(lengths))
                for idx, grp, lengths in packs_recv or []
            ]
            self.group_domains = domains_recv or {}

        # Ensure all ranks sync before iteration
        dist.barrier()

        try:
            logger = logging.getLogger(__name__)
            if is_rank0:
                logger.debug(
                    "GroupedPackingDataset: broadcasted %d packs to %d ranks",
                    len(self._packs),
                    dist.get_world_size(),
                )
            else:
                logger.debug(
                    "GroupedPackingDataset: received %d packs from rank0",
                    len(self._packs),
                )
        except Exception:
            pass

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._reset_dataset_epoch()
        if self._in_distributed():
            if self._is_rank_zero():
                self._build_packs()
            self._broadcast_packs()
        else:
            self._build_packs()

    def __len__(self) -> int:
        return len(self._packs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pack = self._packs[index]
        rows = [self.dataset[i] for i in pack.indices]
        return {
            "packed_items": rows,
            "packed_group": pack.group,
            "packed_segments": list(pack.lengths),
            "packed_length": pack.total_length,
        }


class GroupedIterablePackingDataset(IterableDataset):
    """Iterable view over GroupedPackingDataset."""

    def __init__(self, packing_dataset: GroupedPackingDataset) -> None:
        self.packing_dataset = packing_dataset

    def __iter__(self):
        for idx in range(len(self.packing_dataset)):
            yield self.packing_dataset[idx]

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.packing_dataset)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.packing_dataset, "set_epoch"):
            self.packing_dataset.set_epoch(epoch)  # type: ignore[attr-defined]


def build_grouped_packing_collator(
    template: Any,
    base_collator: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """Collator shim that unwraps packed rows, runs template collator, and reattaches metadata."""

    collate_fn = base_collator or template.data_collator

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if batch and isinstance(batch[0], Mapping) and "packed_items" in batch[0]:
            groups: List[Any] = []
            segments: List[int] = []
            items: List[Dict[str, Any]] = []
            for entry in batch:
                packed_items = entry["packed_items"]
                if isinstance(packed_items, list):
                    items.extend(packed_items)
                else:
                    items.append(packed_items)
                groups.append(entry.get("packed_group"))
                entry_segments = entry.get("packed_segments") or []
                if not entry_segments and entry.get("packed_length") is not None:
                    entry_segments = [int(entry["packed_length"])]
                segments.append(int(sum(entry_segments)))

            # Delegate to template collator (handles packing & padding-free)
            collated = collate_fn(items)
            collated["packed_group"] = groups
            collated["packed_segments"] = segments
            return collated
        return collate_fn(batch)

    return _collate


class GroupedMetricsMixin:
    """Trainer mixin that logs per-group loss/accuracy and strips packing metadata."""

    group_field = "packed_group"
    segment_field = "packed_segments"

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        groups = inputs.pop(self.group_field, None)
        segments = inputs.pop(self.segment_field, None)

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        if groups is not None and segments is not None:
            try:
                self._log_group_metrics(outputs, inputs, groups, segments)
                # Sync per-group metrics across ranks so master logging sees all groups.
                self._sync_group_metrics()
            except Exception:
                # Avoid breaking training due to metrics; continue with primary loss
                pass

        return (loss, outputs) if return_outputs else loss

    def _log_group_metrics(
        self,
        outputs: Any,
        inputs: Mapping[str, Any],
        groups: Sequence[Any],
        segments: Sequence[int],
    ) -> None:
        logits = getattr(outputs, "logits", None)
        labels = inputs.get("labels")
        if logits is None or labels is None:
            return

        if isinstance(groups, torch.Tensor):
            groups = groups.tolist()
        if isinstance(segments, torch.Tensor):
            segments = segments.tolist()

        model = getattr(self, "model", None)
        if model is None:
            return
        mode = "train" if model.training else "eval"
        custom_metrics = getattr(self, "custom_metrics", None)
        if custom_metrics is None:
            return
        metrics = custom_metrics[mode]
        domain_map = getattr(self, "packing_group_domains", None)

        # Align with next-token prediction used in training (shifted by 1).
        logits_next = logits[:, :-1, :]
        labels_next = labels[:, 1:]
        logits_seq = logits_next.reshape(-1, logits_next.shape[-1])
        labels_seq = labels_next.reshape(-1)

        offset = 0
        for grp, seg_len in zip(groups, segments):
            seg_len_int = int(seg_len)
            # Shifted sequence length (drop first token)
            seg_len_shifted = max(seg_len_int - 1, 0)
            if seg_len_shifted <= 0:
                continue
            group_domain = None
            if isinstance(domain_map, Mapping):
                group_domain = domain_map.get(str(grp))
            if mode == "eval" and group_domain == "source":
                offset += seg_len_shifted
                continue
            start, end = offset, min(offset + seg_len_shifted, logits_seq.shape[0])
            offset = end
            if start >= end:
                continue
            seg_logits = logits_seq[start:end]
            seg_labels = labels_seq[start:end]
            mask = seg_labels != -100
            if mask.any():
                with torch.no_grad():
                    seg_loss = F.cross_entropy(
                        seg_logits[mask], seg_labels[mask], reduction="mean"
                    )
                    preds = seg_logits.argmax(dim=-1)
                    seg_acc = (preds[mask] == seg_labels[mask]).float().mean()
                    metrics[f"{grp}_loss"].update(float(seg_loss.detach().cpu()))
                    metrics[f"{grp}_token_acc"].update(float(seg_acc.detach().cpu()))

    def _sync_group_metrics(self) -> None:
        """Sync metric keys across ranks in deterministic order; aggregation happens at log time."""
        if not dist.is_available() or not dist.is_initialized():
            return
        mode = (
            "train"
            if getattr(self, "model", None) is None or self.model.training
            else "eval"
        )
        custom_metrics = getattr(self, "custom_metrics", None)
        if custom_metrics is None or mode not in custom_metrics:
            return
        metrics = custom_metrics[mode]

        local_keys = list(metrics.keys())
        gathered_keys = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_keys, local_keys)

        union_keys = set()
        for keys in gathered_keys:
            if keys:
                union_keys.update(keys)

        # Trigger defaultdict factory (device/group aware) for missing keys.
        # Sorted order keeps key creation deterministic across ranks.
        for key in sorted(union_keys):
            if key not in metrics:
                _ = metrics[key]
        # NOTE: We intentionally do not all-reduce here; MeanMetric.compute()
        # performs a single reduction during logging, avoiding double reduction.
