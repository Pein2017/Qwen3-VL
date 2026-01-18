"""Trainer mixin to guarantee the last checkpoint exists after training."""

from __future__ import annotations

import logging
import os
import weakref
from typing import TYPE_CHECKING, Protocol, TypeVar, cast
from typing_extensions import override

from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, SaveStrategy

if TYPE_CHECKING:  # pragma: no cover
    from transformers import Trainer as _TrainerBase


logger = logging.getLogger(__name__)

TrainerT = TypeVar("TrainerT", bound="_TrainerBase")


class _TrainerLike(Protocol):
    args: TrainingArguments
    state: TrainerState
    model: object
    callback_handler: object

    def add_callback(self, callback: TrainerCallback) -> None: ...

    def _save_checkpoint(
        self, model: object, trial: object | None, metrics: object | None = None
    ) -> None: ...


class _FinalCheckpointCallback(TrainerCallback):
    """Callback bound to a specific trainer instance to enforce the final save."""

    def __init__(self, owner: "FinalCheckpointMixin") -> None:
        self._owner_ref: weakref.ReferenceType[FinalCheckpointMixin] = weakref.ref(
            owner
        )

    @override
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        owner = self._owner_ref()
        if owner is None:
            return
        owner._maybe_save_final_checkpoint(args, state, control)
        return None


class FinalCheckpointMixin:
    """Adds a post-training checkpoint check without modifying upstream trainers."""

    _final_checkpoint_callback_attr: str = "_final_checkpoint_callback"
    _final_checkpoint_wrapper_cache: dict[
        type["_TrainerBase"], type["_TrainerBase"]
    ] = {}

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        if not hasattr(self, self._final_checkpoint_callback_attr):
            callback = _FinalCheckpointCallback(self)
            setattr(self, self._final_checkpoint_callback_attr, callback)
            try:
                trainer = cast(_TrainerLike, cast(object, self))
                trainer.add_callback(callback)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to register final-checkpoint callback: %s", exc)

    # ------------------------------------------------------------------
    # Final checkpoint helpers
    # ------------------------------------------------------------------
    def _maybe_save_final_checkpoint(  # noqa: PLR0912
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        """Persist the last checkpoint if the training loop skipped it."""

        trainer = cast(_TrainerLike, cast(object, self))

        save_strategy = getattr(args, "save_strategy", SaveStrategy.NO)
        # Normalize loose string inputs such as "none" or "NO".
        try:
            save_strategy_enum = SaveStrategy(save_strategy)
        except Exception:
            save_strategy_enum = (
                SaveStrategy.NO
                if str(save_strategy).lower() in ("no", "none")
                else SaveStrategy.STEPS
            )

        if save_strategy_enum == SaveStrategy.NO:
            logger.debug("Final checkpoint skipped because save_strategy is 'no'.")
            return

        should_save_rank = bool(getattr(args, "should_save", False))
        world_size = getattr(args, "world_size", 1)

        if not should_save_rank:
            if world_size <= 1:
                logger.debug(
                    "Final checkpoint skipped because no process is permitted to save checkpoints."
                )
                return
            logger.debug(
                "Final checkpoint: this rank will participate in the distributed save without writing to disk."
            )

        global_step = getattr(state, "global_step", 0)
        if not isinstance(global_step, int) or global_step <= 0:
            logger.debug("Final checkpoint skipped because global_step=%s", global_step)
            return

        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            logger.debug("Final checkpoint skipped because output_dir is undefined.")
            return

        if self._final_checkpoint_exists(output_dir, global_step):
            if should_save_rank:
                logger.debug(
                    "Final checkpoint already present for step %s", global_step
                )
            return

        checkpoint_dir = self._format_checkpoint_dir(output_dir, global_step)
        if should_save_rank:
            logger.info(
                "No checkpoint found at %s; forcing a final save.", checkpoint_dir
            )

        # Keep the forced checkpoint independent from Trainer-managed rotation so
        # save_total_limit continues to govern only the regular save cadence.
        original_limit = getattr(args, "save_total_limit", None)
        limit_suspended = (
            should_save_rank and isinstance(original_limit, int) and original_limit > 0
        )
        if limit_suspended:
            try:
                setattr(args, "save_total_limit", None)
                logger.info(
                    "Temporarily disabling save_total_limit=%s while writing the final checkpoint.",
                    original_limit,
                )
            except Exception:  # pragma: no cover - defensive, log + continue
                limit_suspended = False

        try:
            save_checkpoint = getattr(trainer, "_save_checkpoint", None)
            if not callable(save_checkpoint):
                raise AttributeError("Trainer does not expose _save_checkpoint")
            try:
                _ = save_checkpoint(trainer.model, None)
            except TypeError:
                # Some trainer overrides accept metrics in a third positional slot.
                _ = save_checkpoint(trainer.model, None, None)
        finally:
            if limit_suspended:
                try:
                    setattr(args, "save_total_limit", original_limit)
                except Exception:  # pragma: no cover - defensive
                    logger.warning(
                        "Unable to restore save_total_limit after final checkpoint save; current value may remain unset."
                    )

        # Mirror Trainer.train() behaviour so callbacks observe the save event.
        try:
            on_save = getattr(trainer.callback_handler, "on_save", None)
            if callable(on_save):
                _ = on_save(args, state, control)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Final checkpoint save completed but on_save callbacks failed: %s", exc
            )

    def _final_checkpoint_exists(self, output_dir: str, step: int) -> bool:
        """Return True if the checkpoint directory (or flash record) already exists."""

        trainer = cast(_TrainerLike, cast(object, self))

        if getattr(trainer.args, "use_flash_ckpt", False):
            last_step_fn = getattr(trainer, "_get_last_checkpoint_step", None)
            if callable(last_step_fn):
                try:
                    last_step = last_step_fn()
                except Exception:  # pragma: no cover - defensive
                    last_step = None
            else:
                last_step = None
            if isinstance(last_step, int) and last_step >= step:
                return True

        checkpoint_dir = self._format_checkpoint_dir(output_dir, step)
        if os.path.isdir(checkpoint_dir):
            return True

        last_model_checkpoint = getattr(trainer.state, "last_model_checkpoint", None)
        if isinstance(last_model_checkpoint, str) and os.path.isdir(
            last_model_checkpoint
        ):
            if os.path.basename(last_model_checkpoint) == os.path.basename(
                checkpoint_dir
            ):
                return True

        return False

    @staticmethod
    def _format_checkpoint_dir(output_dir: str, step: int) -> str:
        return os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-{step}")


def with_final_checkpoint(trainer_cls: type[TrainerT] | object) -> type[TrainerT]:
    """Return a trainer subclass that includes :class:`FinalCheckpointMixin`."""

    if not isinstance(trainer_cls, type):
        raise TypeError("trainer_cls must be a class")
    trainer_type = cast(type[TrainerT], trainer_cls)

    if issubclass(trainer_type, FinalCheckpointMixin):
        return trainer_type

    cache = FinalCheckpointMixin._final_checkpoint_wrapper_cache
    if trainer_type in cache:
        return cast(type[TrainerT], cache[trainer_type])

    wrapped = type(
        f"{trainer_type.__name__}WithFinalCheckpoint",
        (FinalCheckpointMixin, trainer_type),
        {},
    )
    wrapped.__module__ = trainer_type.__module__
    cache[trainer_type] = wrapped
    return cast(type[TrainerT], wrapped)


__all__ = ["FinalCheckpointMixin", "with_final_checkpoint"]
