"""Trainer mixin to guarantee the last checkpoint exists after training."""

from __future__ import annotations

import logging
import os
import weakref
from typing import TYPE_CHECKING, Dict, Type, cast

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, SaveStrategy

if TYPE_CHECKING:  # pragma: no cover
    from transformers import Trainer as _TrainerBase


logger = logging.getLogger(__name__)


class _FinalCheckpointCallback(TrainerCallback):
    """Callback bound to a specific trainer instance to enforce the final save."""

    def __init__(self, owner: "FinalCheckpointMixin") -> None:
        self._owner_ref = weakref.ref(owner)

    def on_train_end(  # type: ignore[override]
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        owner = self._owner_ref()
        if owner is None:
            return control
        owner._maybe_save_final_checkpoint(args, state, control)
        return control


class FinalCheckpointMixin:
    """Adds a post-training checkpoint check without modifying upstream trainers."""

    _final_checkpoint_callback_attr = "_final_checkpoint_callback"
    _final_checkpoint_wrapper_cache: Dict[Type, Type] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        if not hasattr(self, self._final_checkpoint_callback_attr):
            callback = _FinalCheckpointCallback(self)
            setattr(self, self._final_checkpoint_callback_attr, callback)
            try:
                trainer = cast("_TrainerBase", self)
                trainer.add_callback(callback)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to register final-checkpoint callback: %s", exc)

    # ------------------------------------------------------------------
    # Final checkpoint helpers
    # ------------------------------------------------------------------
    def _maybe_save_final_checkpoint(  # noqa: PLR0912
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        """Persist the last checkpoint if the training loop skipped it."""

        trainer = cast("_TrainerBase", self)

        save_strategy = getattr(args, "save_strategy", SaveStrategy.NO)
        if save_strategy == SaveStrategy.NO:
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
                logger.debug("Final checkpoint already present for step %s", global_step)
            return

        checkpoint_dir = self._format_checkpoint_dir(output_dir, global_step)
        if should_save_rank:
            logger.info("No checkpoint found at %s; forcing a final save.", checkpoint_dir)

        # Keep the forced checkpoint independent from Trainer-managed rotation so
        # save_total_limit continues to govern only the regular save cadence.
        original_limit = getattr(args, "save_total_limit", None)
        limit_suspended = (
            should_save_rank
            and isinstance(original_limit, int)
            and original_limit > 0
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
            try:
                trainer._save_checkpoint(trainer.model, None)  # type: ignore[misc,arg-type]
            except TypeError:
                # Some trainer overrides accept metrics; fall back to keyword form.
                trainer._save_checkpoint(trainer.model, None, metrics=None)  # type: ignore[misc,call-arg]
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
            trainer.callback_handler.on_save(args, state, control)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Final checkpoint save completed but on_save callbacks failed: %s", exc
            )

    def _final_checkpoint_exists(self, output_dir: str, step: int) -> bool:
        """Return True if the checkpoint directory (or flash record) already exists."""

        trainer = cast("_TrainerBase", self)

        if getattr(trainer.args, "use_flash_ckpt", False) and hasattr(
            trainer, "_get_last_checkpoint_step"
        ):
            try:
                last_step = trainer._get_last_checkpoint_step()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
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


def with_final_checkpoint(trainer_cls: Type) -> Type:
    """Return a trainer subclass that includes :class:`FinalCheckpointMixin`."""

    if not isinstance(trainer_cls, type):
        raise TypeError("trainer_cls must be a class")

    if issubclass(trainer_cls, FinalCheckpointMixin):
        return trainer_cls

    cache = FinalCheckpointMixin._final_checkpoint_wrapper_cache
    if trainer_cls in cache:
        return cache[trainer_cls]

    wrapped = type(
        f"{trainer_cls.__name__}WithFinalCheckpoint",
        (FinalCheckpointMixin, trainer_cls),
        {},
    )
    wrapped.__module__ = trainer_cls.__module__
    cache[trainer_cls] = wrapped
    return wrapped


__all__ = ["FinalCheckpointMixin", "with_final_checkpoint"]
