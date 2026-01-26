"""
SaveDelayCallback: Prevent checkpoint saving during early training.

This callback blocks checkpoint saving until a minimum number of training steps
have been completed, preventing unnecessary saves during the initial warmup phase
when eval_loss is decreasing rapidly.
"""

from collections.abc import Mapping
from typing import cast
from typing_extensions import override

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import SaveStrategy

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

        if not config.active:
            raise ValueError("SaveDelayConfig must have steps or epochs > 0")

        self.config: SaveDelayConfig = config
        self.save_delay_steps: int | None = config.steps
        self.save_delay_epochs: float | None = config.epochs

        # Runtime state
        self._delay_active: bool | None = None
        self._block_logged: bool = False
        self._release_logged: bool = False
        self._pending_reset: bool = False
        self._warned_missing_metric: bool = False
        self._metric_key_cache: str | None = None
        self._metric_epsilon: float = 1e-12

    def _in_delay_period(self, state: TrainerState) -> bool:
        if self.save_delay_steps is not None:
            return state.global_step < self.save_delay_steps
        if self.save_delay_epochs is not None:
            epoch = state.epoch if state.epoch is not None else 0.0
            return epoch < self.save_delay_epochs
        return False

    def _resolve_metric_key(self, args: TrainingArguments) -> str | None:
        if self._metric_key_cache is not None:
            return self._metric_key_cache

        metric_name = getattr(args, "metric_for_best_model", None)
        if not metric_name:
            return None

        key = metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"
        self._metric_key_cache = key
        return key

    def _guard_metric(
        self, metric_value: float, greater_is_better: bool | None
    ) -> float:
        if greater_is_better is False:
            return metric_value - self._metric_epsilon
        # Default to treating higher as better when unset; transformers also defaults this way for non-loss metrics.
        return metric_value + self._metric_epsilon

    @staticmethod
    def _baseline_metric(greater_is_better: bool | None) -> float:
        if greater_is_better is False:
            return float("inf")
        return float("-inf")

    @override
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Override should_save if we're still in the delay period."""

        in_delay_period = self._in_delay_period(state)

        if in_delay_period and control.should_save:
            if not self._block_logged:
                delay_info = (
                    f"{self.save_delay_steps} steps"
                    if self.save_delay_steps is not None
                    else f"{self.save_delay_epochs} epochs"
                )
                epoch_value = state.epoch if state.epoch is not None else 0.0
                logger.info(
                    "SaveDelayCallback: Blocking checkpoint saves until %s (current: step=%d, epoch=%.2f)",
                    delay_info,
                    state.global_step,
                    epoch_value,
                )
                self._block_logged = True

            control.should_save = False

        if self._delay_active is None:
            # Initialize state tracking on the first step callback without triggering transitions.
            self._delay_active = in_delay_period
            return

        if not in_delay_period and self._delay_active:
            # Transition out of the delay window; defer baseline reset to the next evaluation.
            if not self._release_logged:
                logger.info(
                    "SaveDelayCallback: Delay period ended at step %d. Checkpoint saving will resume after the next evaluation.",
                    state.global_step,
                )
                self._release_logged = True
            self._pending_reset = True

        self._delay_active = in_delay_period

    @override
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if args.save_strategy != SaveStrategy.BEST:
            return

        in_delay_period = self._in_delay_period(state)

        if in_delay_period:
            control.should_save = False

            metrics_raw = kwargs.get("metrics")
            if not isinstance(metrics_raw, Mapping):
                # Worker ranks do not receive metrics but must still block saving.
                return
            metrics = cast(Mapping[str, float], metrics_raw)

            metric_key = self._resolve_metric_key(args)
            if metric_key is None:
                if not self._warned_missing_metric:
                    logger.warning(
                        "SaveDelayCallback: save_strategy='best' requires metric_for_best_model to guard saves during the delay window."
                    )
                    self._warned_missing_metric = True
                return

            metric_value = metrics.get(metric_key)
            if metric_value is None:
                if not self._warned_missing_metric:
                    logger.warning(
                        "SaveDelayCallback: Metric '%s' not found in evaluation results; cannot delay best-checkpoint saves.",
                        metric_key,
                    )
                    self._warned_missing_metric = True
                return

            greater_is_better = getattr(args, "greater_is_better", None)

            state.best_metric = self._guard_metric(metric_value, greater_is_better)
            state.best_model_checkpoint = None
            state.best_global_step = 0
            return

        if self._pending_reset:
            greater_is_better = getattr(args, "greater_is_better", None)
            state.best_metric = self._baseline_metric(greater_is_better)
            state.best_model_checkpoint = None
            state.best_global_step = 0
            self._pending_reset = False
            self._warned_missing_metric = False
