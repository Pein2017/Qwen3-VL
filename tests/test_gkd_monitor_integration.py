from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.config.loader import ConfigLoader
from src.config import TrainingConfig, CustomConfig, SaveDelayConfig
from src.sft import resolve_trainer_cls
from src.trainers import GKDTrainerWithMetrics
from src.stage_a.cli import StageAConfig
from src.stage_a.prompts import SUPPORTED_MISSIONS


def test_load_training_config_returns_dataclasses(monkeypatch):
    monkeypatch.setattr(
        "swift.llm.argument.train_args.TrainArguments._init_deepspeed",
        lambda self: None,
    )

    def _noop_ds_init(self, cfg):
        self.config = {"zero_optimization": {}}

    monkeypatch.setattr(
        "transformers.integrations.deepspeed.HfTrainerDeepSpeedConfig.__init__",
        _noop_ds_init,
    )
    monkeypatch.setattr(
        "accelerate.utils.dataclasses.DeepSpeedPlugin.__post_init__",
        lambda self: None,
    )

    train_args, training_config = ConfigLoader.load_training_config(
        "configs/debug.yaml"
    )

    assert isinstance(training_config, TrainingConfig)
    assert isinstance(training_config.custom, CustomConfig)

    save_delay_cfg = getattr(train_args, "save_delay_config", None)
    assert isinstance(save_delay_cfg, SaveDelayConfig)
    if save_delay_cfg.steps is not None:
        assert getattr(train_args, "save_delay_steps") == save_delay_cfg.steps
    if save_delay_cfg.epochs is not None:
        assert getattr(train_args, "save_delay_epochs") == save_delay_cfg.epochs


def test_stage_a_config_validation(tmp_path):
    input_dir = tmp_path / "input"
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    checkpoint_dir.mkdir()

    cfg = StageAConfig(
        checkpoint=str(checkpoint_dir),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        mission=SUPPORTED_MISSIONS[0],
    )

    cfg.validate()

    with pytest.raises(ValueError):
        replace(cfg, top_p=1.5).validate()


class _StaticOutputModel(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self._outputs = outputs

    def forward(self, **kwargs):
        return self._outputs


def test_stage3_config_references_real_assets():
    cfg = ConfigLoader.load_yaml_with_extends("configs/stage_3_gkd.yaml")

    model_path = cfg["model"]["model"]
    teacher_path = cfg["rlhf"]["teacher_model"]
    assert model_path.startswith("output/")
    assert teacher_path.startswith("model_cache/")
    assert teacher_path != model_path
    assert cfg["custom"]["trainer_variant"] == "gkd_monitor"
    assert cfg["custom"]["train_jsonl"] == "data/bbu_full_768/train.jsonl"


def test_resolve_trainer_cls_returns_monitor():
    args = SimpleNamespace(rlhf_type="gkd", trainer_variant="gkd_monitor")

    trainer_cls = resolve_trainer_cls(args)

    assert trainer_cls is GKDTrainerWithMetrics


def test_resolve_trainer_cls_falls_back(monkeypatch):
    sentinel = object()

    class DummyFactory:
        @staticmethod
        def get_trainer_cls(_):
            return sentinel

    monkeypatch.setattr("src.sft.TrainerFactory", DummyFactory)

    args = SimpleNamespace(rlhf_type="gkd", trainer_variant=None)

    trainer_cls = resolve_trainer_cls(args)

    assert trainer_cls is sentinel


def test_gkd_monitor_logs_losses(monkeypatch):
    captured_logs = {}

    def dummy_super_log(self, logs, start_time=None):
        captured_logs.update(logs)

    from src.trainers.gkd_monitor import _MsSwiftGKDTrainer

    monkeypatch.setattr(_MsSwiftGKDTrainer, "log", dummy_super_log)

    trainer = object.__new__(GKDTrainerWithMetrics)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._metrics["train"]["kl_loss"].extend([0.1, 0.2])
    trainer._metrics["train"]["sft_loss"].extend([0.4, 0.5])
    trainer._metrics["train"]["loss"].extend([0.5, 0.7])
    trainer._metrics["train"]["token_accuracy"].extend([0.8, 0.9])
    trainer.model = SimpleNamespace(training=True)

    trainer.log({"eval/loss": 1.23})

    assert pytest.approx(captured_logs["train/kl_loss"], rel=1e-6) == (0.1 + 0.2) / 2
    assert pytest.approx(captured_logs["train/sft_loss"], rel=1e-6) == (0.4 + 0.5) / 2
    assert pytest.approx(captured_logs["train/loss"], rel=1e-6) == (0.5 + 0.7) / 2
    assert (
        pytest.approx(captured_logs["train/token_accuracy"], rel=1e-6)
        == (0.8 + 0.9) / 2
    )
    assert captured_logs["loss"] == pytest.approx(captured_logs["train/loss"])
    assert captured_logs["eval/loss"] == pytest.approx(1.23)
    assert "train/eval/loss" not in captured_logs
    assert not trainer._metrics["train"]

    trainer._metrics["eval"]["kl_loss"].extend([0.3])
    trainer._metrics["eval"]["sft_loss"].extend([0.6])
    trainer._metrics["eval"]["loss"].extend([0.9])
    trainer._metrics["eval"]["token_accuracy"].extend([0.4])

    trainer.log({})

    assert pytest.approx(captured_logs["eval/kl_loss"], rel=1e-6) == 0.3
    assert pytest.approx(captured_logs["eval/sft_loss"], rel=1e-6) == 0.6
    assert pytest.approx(captured_logs["eval/loss"], rel=1e-6) == 0.9
    assert pytest.approx(captured_logs["eval/token_accuracy"], rel=1e-6) == 0.4
    assert not trainer._metrics["eval"]


def test_gkd_compute_loss_aligns_tokens():
    labels = torch.tensor([[1, 3, -100]])
    vocab_size = 5

    student_logits = torch.full((1, 3, vocab_size), -9.0)
    student_logits[0, 0, 3] = 7.0  # Token aligning with label 3 at next-step prediction
    student_outputs = SimpleNamespace(
        logits=student_logits,
        loss=torch.tensor(0.0, requires_grad=True),
    )

    teacher_logits = torch.full((1, 3, vocab_size), -4.0)
    teacher_outputs = SimpleNamespace(logits=teacher_logits, loss=None)

    student_model = _StaticOutputModel(student_outputs)
    student_model.train()
    teacher_model = _StaticOutputModel(teacher_outputs)

    trainer = object.__new__(GKDTrainerWithMetrics)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.teacher_model = teacher_model  # type: ignore[assignment]
    trainer.beta = 0.25
    trainer.args = SimpleNamespace(sft_alpha=0.0)  # type: ignore[assignment]
    trainer.get_use_logits_to_keep = lambda default_value=True: False  # type: ignore[method-assign]
    trainer.prepare_logits_to_keep = lambda inputs: None  # type: ignore[method-assign]

    captured = {}

    def fake_jsd(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1,
        reduction="batchmean",
    ):
        captured["student"] = student_logits.detach()
        captured["teacher"] = teacher_logits.detach()
        captured["beta"] = beta
        return torch.tensor(0.5, requires_grad=True)

    trainer.generalized_jsd_loss = fake_jsd  # type: ignore[method-assign]

    loss = trainer.compute_loss(student_model, {"labels": labels})
    assert isinstance(loss, torch.Tensor)

    assert loss.item() == pytest.approx(0.5)
    assert captured["student"].shape == (1, 1, vocab_size)
    assert captured["teacher"].shape == (1, 1, vocab_size)
    labels_next = labels[:, 1:]
    mask = labels_next != -100
    expected_student = (
        student_logits[:, :-1, :]
        .masked_select(mask.unsqueeze(-1))
        .view(1, -1, vocab_size)
    )
    expected_teacher = (
        teacher_logits[:, :-1, :]
        .masked_select(mask.unsqueeze(-1))
        .view(1, -1, vocab_size)
    )
    torch.testing.assert_close(captured["student"], expected_student)
    torch.testing.assert_close(captured["teacher"], expected_teacher)
    assert captured["beta"] == pytest.approx(0.25)
    assert trainer._metrics["train"]["kl_loss"][0].item() == pytest.approx(0.5)
    assert trainer._metrics["train"]["token_accuracy"][0].item() == pytest.approx(1.0)


def test_gkd_compute_loss_raises_on_vocab_mismatch():
    labels = torch.tensor([[1, 2, -100]])

    student_logits = torch.zeros((1, 3, 4))
    student_outputs = SimpleNamespace(logits=student_logits, loss=torch.tensor(0.0))

    teacher_logits = torch.zeros((1, 3, 6))
    teacher_outputs = SimpleNamespace(logits=teacher_logits, loss=None)

    student_model = _StaticOutputModel(student_outputs)
    student_model.train()
    teacher_model = _StaticOutputModel(teacher_outputs)

    trainer = object.__new__(GKDTrainerWithMetrics)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.teacher_model = teacher_model  # type: ignore[assignment]
    trainer.beta = 0.1
    trainer.args = SimpleNamespace(sft_alpha=0.0)  # type: ignore[assignment]
    trainer.get_use_logits_to_keep = lambda default_value=True: False  # type: ignore[method-assign]
    trainer.prepare_logits_to_keep = lambda inputs: None  # type: ignore[method-assign]
    trainer.generalized_jsd_loss = lambda *args, **kwargs: torch.tensor(0.0)  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="vocabulary size mismatch"):
        trainer.compute_loss(student_model, {"labels": labels})


class _ReturnTensorModule(nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def forward(self, *args, **kwargs):
        return self.tensor


class _DummyVisual(nn.Module):
    def __init__(self, merger_tensor, deepstack_tensor):
        super().__init__()
        self.merger = _ReturnTensorModule(merger_tensor)
        self.deepstack_merger_list = nn.ModuleList(
            [_ReturnTensorModule(deepstack_tensor)]
        )

    def forward(self, *args, **kwargs):
        merger = self.merger(None)
        deepstack = [module(None) for module in self.deepstack_merger_list]
        return merger, deepstack


class _DummyInnerModel(nn.Module):
    def __init__(self, logits, loss_tensor, merger_tensor, deepstack_tensor):
        super().__init__()
        self.visual = _DummyVisual(merger_tensor, deepstack_tensor)
        self._outputs = SimpleNamespace(logits=logits, loss=loss_tensor)

    def forward(self, **kwargs):
        # Trigger visual hooks so caches populate
        self.visual()
        return self._outputs


class _DummyTopModel(nn.Module):
    def __init__(self, logits, loss_tensor, merger_tensor, deepstack_tensor):
        super().__init__()
        self.model = _DummyInnerModel(
            logits, loss_tensor, merger_tensor, deepstack_tensor
        )

    def forward(self, **kwargs):
        return self.model(**kwargs)


def _build_visual_models(student_tensors, teacher_tensors):
    student_logits = student_tensors["logits"]
    teacher_logits = teacher_tensors["logits"]
    student_model = _DummyTopModel(
        student_logits,
        student_tensors["loss"],
        student_tensors["merger"],
        student_tensors["deepstack"],
    )
    teacher_model = _DummyTopModel(
        teacher_logits,
        teacher_tensors["loss"],
        teacher_tensors["merger"],
        teacher_tensors["deepstack"],
    )
    return student_model, teacher_model


def test_visual_kd_adds_weighted_loss():
    labels = torch.tensor([[1, 2, -100]])
    vocab_size = 4
    base_loss = torch.tensor(0.0, requires_grad=True)

    student_tensors = {
        "logits": torch.zeros((1, 3, vocab_size), requires_grad=True),
        "loss": base_loss,
        "merger": torch.tensor([[1.0, 0.0]], requires_grad=True),
        "deepstack": torch.tensor([[0.5, 0.5]], requires_grad=True),
    }
    teacher_tensors = {
        "logits": torch.zeros((1, 3, vocab_size)),
        "loss": None,
        "merger": torch.zeros((1, 2)),
        "deepstack": torch.zeros((1, 2)),
    }

    student_model, teacher_model = _build_visual_models(
        student_tensors, teacher_tensors
    )
    student_model.train()
    teacher_model.eval()

    trainer = object.__new__(GKDTrainerWithMetrics)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.teacher_model = teacher_model  # type: ignore[assignment]
    trainer.beta = 0.0
    trainer.args = SimpleNamespace(sft_alpha=0.0)  # type: ignore[assignment]
    trainer.get_use_logits_to_keep = lambda default_value=True: False  # type: ignore[method-assign]
    trainer.prepare_logits_to_keep = lambda inputs: None  # type: ignore[method-assign]
    trainer.generalized_jsd_loss = lambda *args, **kwargs: torch.tensor(0.0)  # type: ignore[method-assign]
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda m: m)  # type: ignore[assignment]
    trainer.deepspeed = None

    trainer._visual_hooks = []
    trainer._student_visual_cache = {}
    trainer._teacher_visual_cache = {}
    trainer._visual_kd_enabled = True
    trainer._visual_kd_weight = 0.1
    trainer._visual_kd_targets = ["merger", "deepstack"]
    trainer._visual_kd_distance = "mse"

    trainer.model = student_model
    trainer._register_visual_hooks()

    loss = trainer.compute_loss(student_model, {"labels": labels})
    assert isinstance(loss, torch.Tensor)

    # MSE merger = 0.5, deepstack = 0.25 => mean = 0.375; weighted = 0.0375
    expected_weighted_loss = torch.tensor(0.0375)
    assert pytest.approx(loss.item(), rel=1e-6) == pytest.approx(
        expected_weighted_loss.item()
    )
    logged_loss = trainer._metrics["train"]["vision_kd_loss"][0]
    assert pytest.approx(logged_loss.item(), rel=1e-6) == expected_weighted_loss.item()
    logged_total = trainer._metrics["train"]["loss"][0]
    assert pytest.approx(logged_total.item(), rel=1e-6) == expected_weighted_loss.item()


def test_visual_kd_skips_when_disabled():
    labels = torch.tensor([[1, -100]])
    vocab_size = 2

    student_tensors = {
        "logits": torch.zeros((1, 2, vocab_size), requires_grad=True),
        "loss": torch.tensor(0.0, requires_grad=True),
        "merger": torch.ones((1, 2), requires_grad=True),
        "deepstack": torch.ones((1, 2), requires_grad=True),
    }
    teacher_tensors = {
        "logits": torch.zeros((1, 2, vocab_size)),
        "loss": None,
        "merger": torch.zeros((1, 2)),
        "deepstack": torch.zeros((1, 2)),
    }

    student_model, teacher_model = _build_visual_models(
        student_tensors, teacher_tensors
    )
    student_model.train()
    teacher_model.eval()

    trainer = object.__new__(GKDTrainerWithMetrics)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.teacher_model = teacher_model  # type: ignore[assignment]
    trainer.beta = 0.0
    trainer.args = SimpleNamespace(sft_alpha=0.0)  # type: ignore[assignment]
    trainer.get_use_logits_to_keep = lambda default_value=True: False  # type: ignore[method-assign]
    trainer.prepare_logits_to_keep = lambda inputs: None  # type: ignore[method-assign]
    trainer.generalized_jsd_loss = lambda *args, **kwargs: torch.tensor(0.0)  # type: ignore[method-assign]
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda m: m)  # type: ignore[assignment]
    trainer.deepspeed = None

    trainer._visual_hooks = []
    trainer._student_visual_cache = {}
    trainer._teacher_visual_cache = {}
    trainer._visual_kd_enabled = False
    trainer._visual_kd_weight = 0.0
    trainer._visual_kd_targets = []
    trainer._visual_kd_distance = "mse"

    trainer.model = student_model

    loss = trainer.compute_loss(student_model, {"labels": labels})
    assert isinstance(loss, torch.Tensor)

    assert loss.item() == pytest.approx(0.0)
    assert "vision_kd_loss" not in trainer._metrics["train"]


def test_gkd_eval_logs_kl_loss():
    labels = torch.tensor([[1, 2, -100]])
    vocab_size = 4

    student_tensors = {
        "logits": torch.zeros((1, 3, vocab_size), requires_grad=True),
        "loss": torch.tensor(0.0),
        "merger": torch.tensor([[1.0, 0.0]], requires_grad=True),
        "deepstack": torch.tensor([[0.5, 0.5]], requires_grad=True),
    }
    teacher_tensors = {
        "logits": torch.zeros((1, 3, vocab_size)),
        "loss": None,
        "merger": torch.zeros((1, 2)),
        "deepstack": torch.zeros((1, 2)),
    }

    student_model, teacher_model = _build_visual_models(
        student_tensors, teacher_tensors
    )
    student_model.eval()
    teacher_model.eval()

    trainer = object.__new__(GKDTrainerWithMetrics)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.teacher_model = teacher_model  # type: ignore[assignment]
    trainer.beta = 0.0
    trainer.args = SimpleNamespace(sft_alpha=0.0)  # type: ignore[assignment]
    trainer.get_use_logits_to_keep = lambda default_value=True: False  # type: ignore[method-assign]
    trainer.prepare_logits_to_keep = lambda inputs: None  # type: ignore[method-assign]
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda m: m)  # type: ignore[assignment]
    trainer.deepspeed = None

    def fake_jsd(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1,
        reduction="batchmean",
    ):
        assert not student_logits.requires_grad
        return torch.tensor(0.25)

    trainer.generalized_jsd_loss = fake_jsd  # type: ignore[method-assign]

    trainer._visual_hooks = []
    trainer._student_visual_cache = {}
    trainer._teacher_visual_cache = {}
    trainer._visual_kd_enabled = False
    trainer._visual_kd_weight = 0.0
    trainer._visual_kd_targets = []
    trainer._visual_kd_distance = "mse"

    trainer.model = student_model

    loss = trainer.compute_loss(student_model, {"labels": labels})
    assert isinstance(loss, torch.Tensor)

    assert pytest.approx(loss.item(), rel=1e-6) == 0.0
    eval_metrics = trainer._metrics["eval"]
    assert pytest.approx(eval_metrics["kl_loss"][0].item(), rel=1e-6) == 0.25
    assert pytest.approx(eval_metrics["loss"][0].item(), rel=1e-6) == 0.0
    assert pytest.approx(eval_metrics["sft_loss"][0].item(), rel=1e-6) == 0.0
