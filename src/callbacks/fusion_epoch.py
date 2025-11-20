"""Training callback that resamples auxiliary quotas when an epoch begins."""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class FusionEpochCallback(TrainerCallback):
    """Ensure a fusion dataset rebuilds its epoch schedule each epoch."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        self._last_epoch: int | None = None

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        epoch = int(state.epoch) if state.epoch is not None else int(state.global_step or 0)
        if self._last_epoch == epoch:
            return
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        self._last_epoch = epoch
