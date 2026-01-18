from __future__ import annotations

from copy import deepcopy
from collections.abc import MutableMapping
from typing_extensions import override

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from ..datasets.augmentation.curriculum import AugmentationCurriculumScheduler


class AugmentationCurriculumCallback(TrainerCallback):
    """Synchronize scheduler output with dataset state."""

    def __init__(
        self,
        scheduler: AugmentationCurriculumScheduler,
        curriculum_state: MutableMapping[str, object],
    ) -> None:
        self.scheduler: AugmentationCurriculumScheduler = scheduler
        self.curriculum_state: MutableMapping[str, object] = curriculum_state
        self._last_step: int | None = None

    @override
    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        requires_total_steps = bool(
            getattr(self.scheduler, "_requires_total_steps", False)
        )
        final_bypass = getattr(self.scheduler, "_final_bypass", None)
        if requires_total_steps and final_bypass is None:
            total_steps = getattr(state, "max_steps", None)
            if not total_steps:
                total_steps = getattr(args, "max_steps", None)
            if not total_steps:
                raise ValueError(
                    "Cannot resolve percent curriculum: total_steps unavailable"
                )
            self.scheduler.set_total_steps(int(total_steps))
        global_step = int(state.global_step)
        self._update_state(global_step)

    def _update_state(self, global_step: int) -> None:
        if self._last_step == global_step:
            return
        new_state = self.scheduler.get_state(global_step)
        self.curriculum_state["step"] = global_step
        self.curriculum_state["bypass_prob"] = new_state["bypass_prob"]
        self.curriculum_state["ops"] = deepcopy(new_state["ops"])
        self._last_step = global_step
