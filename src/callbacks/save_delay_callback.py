"""
SaveDelayCallback: Prevent checkpoint saving during early training.

This callback blocks checkpoint saving until a minimum number of training steps
have been completed, preventing unnecessary saves during the initial warmup phase
when eval_loss is decreasing rapidly.
"""

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from ..config import SaveDelayConfig
from ..utils import get_logger

logger = get_logger(__name__)


class SaveDelayCallback(TrainerCallback):
    """Delay checkpoint saving until reaching a configured milestone."""

    def __init__(
        self,
        save_delay_steps: int | None = None,
        save_delay_epochs: float | None = None,
        *,
        config: SaveDelayConfig | None = None,
    ) -> None:
        if config is None:
            if save_delay_steps is None and save_delay_epochs is None:
                raise ValueError(
                    "Provide save_delay_steps/save_delay_epochs or a SaveDelayConfig"
                )
            config = SaveDelayConfig.from_raw(save_delay_steps, save_delay_epochs)

        if not isinstance(config, SaveDelayConfig):
            raise TypeError("config must be a SaveDelayConfig instance")
        if not config.active:
            raise ValueError("SaveDelayConfig must have steps or epochs > 0")

        self.config = config
        self.save_delay_steps = config.steps
        self.save_delay_epochs = config.epochs
        self._logged_delay = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Override should_save if we're still in the delay period.
        """
        # Check if we're still in delay period
        in_delay_period = False

        if self.save_delay_steps is not None:
            if state.global_step < self.save_delay_steps:
                in_delay_period = True
        elif self.save_delay_epochs is not None:
            if state.epoch is not None and state.epoch < self.save_delay_epochs:
                in_delay_period = True

        # Block saving if in delay period
        if in_delay_period and control.should_save:
            if not self._logged_delay:
                delay_info = (
                    f"{self.save_delay_steps} steps"
                    if self.save_delay_steps is not None
                    else f"{self.save_delay_epochs} epochs"
                )
                # Format epoch safely; state.epoch can be None
                _epoch = state.epoch if state.epoch is not None else 0.0
                logger.info(
                    f"SaveDelayCallback: Blocking checkpoint saves until {delay_info} "
                    f"(current: step={state.global_step}, epoch={_epoch:.2f})"
                )
                self._logged_delay = True

            control.should_save = False

        # Log when delay period ends
        if not in_delay_period and not self._logged_delay:
            logger.info(
                f"SaveDelayCallback: Delay period ended at step {state.global_step}. "
                f"Checkpoint saving is now enabled."
            )
            self._logged_delay = True

        return control
