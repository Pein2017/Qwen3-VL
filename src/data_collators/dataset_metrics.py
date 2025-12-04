from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch

from src.config.schema import TokenTypeMetricsConfig
from src.data_collators.token_types import TokenType, compute_token_types
from src.utils import get_logger

logger = get_logger(__name__)


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
    token_type_cfg: Optional[TokenTypeMetricsConfig] = None,
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

        _maybe_attach_token_types(
            collated=collated,
            raw_batch=batch,
            dataset_labels=dataset_labels,
            template=template,
            cfg=token_type_cfg,
        )
        return collated

    return _collate


def _maybe_attach_token_types(
    *,
    collated: Dict[str, Any],
    raw_batch: Sequence[Mapping[str, Any]],
    dataset_labels: Sequence[str],
    template: Any,
    cfg: Optional[TokenTypeMetricsConfig],
) -> None:
    if cfg is None or not cfg.enabled:
        return

    labels_tensor = collated.get("labels")
    attention_mask = collated.get("attention_mask")
    if labels_tensor is None or not isinstance(labels_tensor, torch.Tensor):
        return

    tokenizer = getattr(template, "tokenizer", None)
    template_meta = getattr(template, "template_meta", None)
    suffix_tokens = getattr(template_meta, "suffix", None) if template_meta else None

    if tokenizer is None:
        return

    token_type_list: List[torch.Tensor] = []
    has_included = False

    include_set = set(cfg.include)
    exclude_set = set(cfg.exclude)

    for idx, (raw, label) in enumerate(zip(raw_batch, dataset_labels)):
        labels_row: torch.Tensor = labels_tensor[idx]
        attn_row: Optional[torch.Tensor] = None
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            attn_row = attention_mask[idx]

        label_str = str(label) if label is not None else ""
        label_key = label_str.lower()
        included = label_key in include_set and label_key not in exclude_set
        payload = raw.get("assistant_payload")

        if not included or payload is None:
            token_type_list.append(torch.full_like(labels_row, TokenType.IGNORE))
            continue

        has_included = True
        token_types = compute_token_types(
            tokenizer=tokenizer,
            payload=payload,
            labels=labels_row,
            attention_mask=attn_row,
            suffix_tokens=suffix_tokens,
        )
        if token_types is None or token_types.shape[0] != labels_row.shape[0]:
            logger.debug(
                "Falling back to IGNORE token types (label=%s, got=%s, expected=%s)",
                label_key,
                None if token_types is None else token_types.shape,
                labels_row.shape,
            )
            token_types = torch.full_like(labels_row, TokenType.IGNORE)
        token_type_list.append(token_types)

    if not has_included:
        return

    collated["token_types"] = torch.stack(token_type_list, dim=0)
