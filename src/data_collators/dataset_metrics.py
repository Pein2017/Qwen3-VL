from typing import Any, Callable, Dict, List, Mapping

import torch


def _resolve_label(row: Mapping[str, Any]) -> str:
    if not isinstance(row, Mapping):
        return "default"
    meta = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else None
    if meta:
        label = meta.get("_fusion_source") or meta.get("dataset")
        if label:
            return str(label)
    dataset_name = row.get("dataset") or row.get("dataset_name")
    return str(dataset_name) if dataset_name else "default"


def build_dataset_metrics_collator(
    template: Any,
    base_collator: Callable[[List[Dict[str, Any]]], Dict[str, Any]] | None = None,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """Wrap the template collator to attach per-sample dataset labels (and lengths).

    Works with padded batches (no packing). Lengths are derived from attention_mask
    when available, otherwise from input_ids shape. The trainer-side mixin only
    requires `dataset_labels`; `dataset_segments` is kept for compatibility/debug.
    """

    collate_fn = base_collator or template.data_collator

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        dataset_labels = [_resolve_label(row) for row in batch]
        collated = collate_fn(batch)

        # Derive per-sample lengths from attention_mask if present, else input_ids.
        segments: List[int]
        if "attention_mask" in collated:
            am = collated["attention_mask"]
            if isinstance(am, torch.Tensor):
                segments = am.long().sum(dim=-1).tolist()
            else:
                segments = [int(sum(x)) for x in am]
        elif "input_ids" in collated:
            ids = collated["input_ids"]
            if isinstance(ids, torch.Tensor):
                segments = [ids.shape[-1]] * ids.shape[0]
            else:
                segments = [len(x) for x in ids]
        else:
            segments = [0 for _ in dataset_labels]

        collated["dataset_labels"] = dataset_labels
        collated["dataset_segments"] = segments
        return collated

    return _collate
