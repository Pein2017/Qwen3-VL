"""Training callback that resamples auxiliary quotas when an epoch begins."""

from __future__ import annotations

from typing import Protocol, cast
from typing_extensions import override

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class _EpochAware(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


class FusionEpochCallback(TrainerCallback):
    """Ensure a fusion dataset rebuilds its epoch schedule each epoch."""

    def __init__(self, dataset: object) -> None:
        self.dataset: object = dataset
        self._last_epoch: int | None = None

    @override
    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        epoch = (
            int(state.epoch) if state.epoch is not None else int(state.global_step or 0)
        )
        if self._last_epoch == epoch:
            return
        if hasattr(self.dataset, "set_epoch"):
            cast(_EpochAware, self.dataset).set_epoch(epoch)
        self._last_epoch = epoch
