from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import SaveStrategy

_FINAL_CKPT_SPEC = importlib.util.spec_from_file_location(
    "_final_checkpoint_under_test",
    Path(__file__).resolve().parents[1] / "src/callbacks/final_checkpoint.py",
)
assert _FINAL_CKPT_SPEC is not None and _FINAL_CKPT_SPEC.loader is not None
_final_ckpt_module = importlib.util.module_from_spec(_FINAL_CKPT_SPEC)
_FINAL_CKPT_SPEC.loader.exec_module(_final_ckpt_module)
FinalCheckpointMixin = _final_ckpt_module.FinalCheckpointMixin


class _CallbackRecorder:
    def __init__(self) -> None:
        self.seen = 0

    def on_save(self, *args, **kwargs) -> None:  # noqa: ANN002 - signature mirrors HF handler
        self.seen += 1


class _BaseTrainer:
    def __init__(self, *, output_dir: Path, save_total_limit: int | None):
        self.args = SimpleNamespace(
            save_strategy=SaveStrategy.STEPS,
            should_save=True,
            output_dir=str(output_dir),
            save_total_limit=save_total_limit,
            use_flash_ckpt=False,
        )
        self.state = TrainerState()
        self.state.global_step = 0
        self.control = TrainerControl()
        self.model = object()
        self.callback_handler = _CallbackRecorder()
        self._callbacks: list[TrainerCallback] = []
        self._checkpoint_steps: list[int] = []
        self._limit_snapshots: list[int | None] = []

        # Pre-populate history from on-disk checkpoints so rotation logic emulates transformers.Trainer
        output_path = Path(self.args.output_dir)
        for folder in sorted(output_path.glob("checkpoint-*")):
            suffix = folder.name.split("-", 1)[-1]
            if suffix.isdigit():
                self._checkpoint_steps.append(int(suffix))

    # -- Hooks expected by FinalCheckpointMixin ---------------------------------
    def add_callback(
        self, callback: TrainerCallback
    ) -> None:  # pragma: no cover - trivial
        self._callbacks.append(callback)

    def _save_checkpoint(self, model, trial, metrics=None):  # noqa: ANN001, D401, PLR0912
        """Minimal checkpoint writer that enforces save_total_limit."""

        step = self.state.global_step
        checkpoint = Path(self.args.output_dir) / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True, exist_ok=True)

        self._limit_snapshots.append(getattr(self.args, "save_total_limit", None))
        self._checkpoint_steps.append(step)

        limit = getattr(self.args, "save_total_limit", None)
        if isinstance(limit, int) and limit > 0:
            # Delete oldest checkpoints beyond the limit to simulate Trainer rotation.
            while len(self._checkpoint_steps) > limit:
                removed_step = self._checkpoint_steps.pop(0)
                victim = Path(self.args.output_dir) / f"checkpoint-{removed_step}"
                if victim.exists():
                    shutil.rmtree(victim)


class _TrainerWithFinal(FinalCheckpointMixin, _BaseTrainer):
    def __init__(self, *, output_dir: Path, save_total_limit: int | None):
        super().__init__(output_dir=output_dir, save_total_limit=save_total_limit)


def _make_checkpoint(output_dir: Path, step: int) -> None:
    folder = output_dir / f"checkpoint-{step}"
    folder.mkdir(parents=True, exist_ok=True)


def test_final_checkpoint_bypasses_save_total_limit(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path, 100)
    _make_checkpoint(tmp_path, 200)

    trainer = _TrainerWithFinal(output_dir=tmp_path, save_total_limit=2)
    trainer.state.global_step = 250

    trainer._maybe_save_final_checkpoint(trainer.args, trainer.state, trainer.control)

    checkpoints = sorted(p.name for p in tmp_path.glob("checkpoint-*"))
    assert checkpoints == ["checkpoint-100", "checkpoint-200", "checkpoint-250"]
    assert trainer.args.save_total_limit == 2  # restored after the forced save
    assert (
        trainer._limit_snapshots[-1] is None
    )  # rotation was suspended during the forced save


def test_final_checkpoint_respects_save_strategy_no(tmp_path: Path) -> None:
    trainer = _TrainerWithFinal(output_dir=tmp_path, save_total_limit=1)
    trainer.args.save_strategy = SaveStrategy.NO
    trainer.state.global_step = 10

    trainer._maybe_save_final_checkpoint(trainer.args, trainer.state, trainer.control)

    assert list(tmp_path.glob("checkpoint-*")) == []
    assert trainer._limit_snapshots == []
