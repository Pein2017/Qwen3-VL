from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F


class DatasetMetricsMixin:
    """Trainer mixin to log per-dataset loss/accuracy on padded batches.

    - Train: logs `_loss` and `_token_acc` for every dataset present in the batch
    - Eval: logs only target-domain datasets when `dataset_domains[label] == "target"`

    Requires batches to include `dataset_labels` (list/1D tensor). `dataset_segments`
    is accepted and stripped but no longer required; lengths are inferred from labels
    and padding masks.
    """

    label_field = "dataset_labels"
    segment_field = "dataset_segments"

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        dataset_labels = inputs.pop(self.label_field, None)
        _ = inputs.pop(self.segment_field, None)  # Optional legacy field
        token_types = inputs.pop("token_types", None)

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        if dataset_labels is not None:
            try:
                self._log_dataset_metrics(outputs, inputs, dataset_labels, token_types)
                self._sync_dataset_metrics()
            except Exception:
                pass

        return (loss, outputs) if return_outputs else loss

    def _log_dataset_metrics(
        self,
        outputs: Any,
        inputs: Mapping[str, Any],
        dataset_labels: Any,
        token_types: Any,
    ) -> None:
        logits = getattr(outputs, "logits", None)
        labels = inputs.get("labels")
        if logits is None or labels is None:
            return

        if isinstance(dataset_labels, torch.Tensor):
            dataset_labels = dataset_labels.tolist()

        if not isinstance(dataset_labels, (list, tuple)):
            return

        mode = "train" if getattr(self, "model", None) is None or self.model.training else "eval"  # type: ignore[attr-defined]
        custom_metrics = getattr(self, "custom_metrics", None)
        if custom_metrics is None or mode not in custom_metrics:
            return
        metrics = custom_metrics[mode]
        domain_map = getattr(self, "dataset_domains", None)

        batch_size = logits.shape[0]
        if len(dataset_labels) != batch_size:
            return

        logits_next = logits[:, :-1, :]
        labels_next = labels[:, 1:]
        token_types_next = token_types[:, 1:] if isinstance(token_types, torch.Tensor) else None

        for idx, lbl in enumerate(dataset_labels):
            if lbl is None:
                continue
            label_str = str(lbl)

            domain_skip = False
            if mode == "eval" and isinstance(domain_map, Mapping):
                dom = domain_map.get(label_str)
                domain_skip = dom == "source"

            sample_logits = logits_next[idx]
            sample_labels = labels_next[idx]
            mask = sample_labels != -100
            if not mask.any():
                continue

            supervised_token_count = int(mask.sum().detach())

            if not domain_skip:
                with torch.no_grad():
                    seg_loss = F.cross_entropy(
                        sample_logits[mask], sample_labels[mask], reduction="mean"
                    )
                    preds = sample_logits.argmax(dim=-1)
                    seg_acc = (preds[mask] == sample_labels[mask]).float().mean()

                metrics[f"{label_str}_loss"].update(float(seg_loss.detach().item()))
                metrics[f"{label_str}_token_acc"].update(float(seg_acc.detach().item()))
                metrics[f"{label_str}_token_count"].update(float(supervised_token_count))

            # Token-type metrics (optional)
            if token_types_next is None:
                continue
            sample_types = token_types_next[idx]
            if sample_types.shape != sample_labels.shape:
                continue
            for typ_id, suffix in ((1, "desc"), (2, "coord"), (3, "format")):
                type_mask = (sample_types == typ_id) & mask
                if not type_mask.any():
                    continue
                type_token_count = int(type_mask.sum().detach())
                with torch.no_grad():
                    type_logits = sample_logits[type_mask]
                    type_labels = sample_labels[type_mask]
                    type_preds = type_logits.argmax(dim=-1)
                    acc = (type_preds == type_labels).float().mean()
                    # Entropy: mean over tokens, sum over vocab
                    probs = F.softmax(type_logits, dim=-1)
                    entropy = (-probs * probs.log()).sum(dim=-1).mean()
                metrics[f"{label_str}_{suffix}_token_acc"].update(
                    float(acc.detach().item())
                )
                metrics[f"{label_str}_{suffix}_entropy"].update(
                    float(entropy.detach().item())
                )
                metrics[f"{label_str}_{suffix}_token_count"].update(
                    float(type_token_count)
                )

    def _sync_dataset_metrics(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        mode = "train" if getattr(self, "model", None) is None or self.model.training else "eval"  # type: ignore[attr-defined]
        custom_metrics = getattr(self, "custom_metrics", None)
        if custom_metrics is None or mode not in custom_metrics:
            return
        metrics = custom_metrics[mode]

        local_keys = list(metrics.keys())
        key_cache = getattr(self, "_dataset_metric_key_cache", {})
        cached = key_cache.get(mode, set())
        local_set = set(local_keys)
        if local_set.issubset(cached):
            return

        gathered_keys = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_keys, local_keys)

        union_keys = set()
        for keys in gathered_keys:
            if keys:
                union_keys.update(keys)

        for key in sorted(union_keys):
            if key not in metrics:
                _ = metrics[key]

        cached.update(union_keys)
        key_cache[mode] = cached
        setattr(self, "_dataset_metric_key_cache", key_cache)
