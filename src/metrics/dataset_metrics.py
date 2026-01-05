from typing import Any, Mapping, cast

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

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Some trainers (e.g., TRL GRPO) pass inputs as a list during eval.
        # In that case, bypass dataset metrics and delegate directly.
        if isinstance(inputs, list):
            if len(inputs) == 1 and isinstance(inputs[0], Mapping):
                inputs = inputs[0]
            else:
                parent = cast(Any, super())
                return parent.compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

        if not isinstance(inputs, Mapping):
            parent = cast(Any, super())
            return parent.compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        # Work on a mutable copy to avoid mutating Mapping inputs.
        inputs_map: dict[str, Any] = dict(inputs)
        dataset_labels = inputs_map.pop(self.label_field, None)
        _ = inputs_map.pop(self.segment_field, None)  # Optional legacy field
        token_types = inputs_map.pop("token_types", None)

        parent = cast(Any, super())
        result = parent.compute_loss(
            model,
            inputs_map,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        outputs = None
        if isinstance(result, tuple):
            loss = result[0]
            if len(result) > 1:
                outputs = result[1]
        else:
            loss = result

        if dataset_labels is not None and outputs is not None:
            # Log metrics per-sample; skip distributed sync to avoid deadlocks.
            # The trainer's own aggregation handles cross-rank reduction safely.
            self._log_dataset_metrics(outputs, inputs_map, dataset_labels, token_types)
            # Skip sync entirely during eval to prevent deadlocks when ranks have imbalanced batches.
            # During training, all ranks process batches uniformly, so sync is safe but unnecessary.
            # We disable it completely to avoid any potential issues.
            # self._sync_dataset_metrics()  # Disabled: trainer handles metric aggregation

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

        model_attr = getattr(self, "model", None)
        mode = "train" if model_attr is None or model_attr.training else "eval"
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
        token_types_next = (
            token_types[:, 1:] if isinstance(token_types, torch.Tensor) else None
        )

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

            if not domain_skip:
                with torch.no_grad():
                    seg_loss = F.cross_entropy(
                        sample_logits[mask], sample_labels[mask], reduction="mean"
                    )
                    preds = sample_logits.argmax(dim=-1)
                    seg_acc = (preds[mask] == sample_labels[mask]).float().mean()

                    # Guard against non-finite values (e.g., NaN/inf) from unstable logits.
                    if not torch.isfinite(seg_loss) or not torch.isfinite(seg_acc):
                        continue

                metrics[f"{label_str}_loss"].update(float(seg_loss.detach().item()))
                metrics[f"{label_str}_token_acc"].update(float(seg_acc.detach().item()))

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
                with torch.no_grad():
                    type_logits = sample_logits[type_mask]
                    type_labels = sample_labels[type_mask]
                    type_preds = type_logits.argmax(dim=-1)
                    acc = (type_preds == type_labels).float().mean()

                    # Skip token-type metrics when accuracy is non-finite.
                    if not torch.isfinite(acc):
                        continue
                metrics[f"{label_str}_{suffix}_token_acc"].update(
                    float(acc.detach().item())
                )

    def _sync_dataset_metrics(self) -> None:
        """Synchronize dataset metric key structure.

        NOTE: To avoid potential distributed deadlocks during evaluation when some
        ranks receive no batches, this is intentionally a no-op in multi-GPU runs.
        Metrics remain correct per-rank and are still aggregated by the trainer.
        """
        if not dist.is_available() or not dist.is_initialized():
            return

        # Disable cross-rank key synchronization when running with >1 rank.
        # This prevents hangs caused by ranks that never enter compute_loss()
        # during eval (common with small or imbalanced eval splits).
        world_size = dist.get_world_size()
        if world_size <= 1:
            return

        # For now, do not perform any collective operations here.
        # If future use-cases require strict cross-rank key alignment, this
        # should be reintroduced with a protocol that guarantees participation
        # from all ranks in each eval phase.
        return
