from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.config import (
    CustomConfig,
    PromptOverrides,
    SaveDelayConfig,
    TrainingConfig,
    VisualKDConfig,
    VisualKDTargetConfig,
)
from src.config.loader import ConfigLoader
from src.config.missions import SUPPORTED_MISSIONS
from src.sft import resolve_trainer_cls
from src.stage_a.cli import StageAConfig
from src.trainers import GKDTrainerWithMetrics


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
    monkeypatch.setattr(
        "transformers.training_args.is_torch_bf16_gpu_available",
        lambda: True,
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


class _DummyTrainArguments:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.training_args = SimpleNamespace()


class _DummyRLHFArguments(_DummyTrainArguments):
    pass


def _build_minimal_training_config(
    *,
    llm_kd_weight=None,
    teacher_model="teacher",
    visual_enabled=False,
) -> TrainingConfig:
    prompts = PromptOverrides(system=None, user="user prompt", output_variant="dense")
    rlhf_section = {
        "rlhf_type": "gkd",
        "beta": 0.1,
    }
    if teacher_model is not None:
        rlhf_section["teacher_model"] = teacher_model
    if llm_kd_weight is not None:
        rlhf_section["llm_kd_weight"] = llm_kd_weight

    visual_kd_section = {"enabled": visual_enabled}
    if visual_enabled:
        visual_kd_section.update(
            {
                "vit": {"enabled": False},
                "aligner": {
                    "enabled": True,
                    "weight": 0.1,
                    "distance": "mse",
                },
                "deepstack": {"enabled": False},
            }
        )

    payload = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "data/train.jsonl",
            "user_prompt": "user prompt",
            "emit_norm": "norm1000",
            "json_format": "standard",
            "visual_kd": visual_kd_section,
        },
        # TrainingConfig requires a non-empty model path; for unit tests we can use a
        # dummy value since we never instantiate the actual model here.
        "model": {"model": "model_cache/dummy"},
        "quantization": {},
        "data": {},
        "tuner": {},
        "training": {
            "output_dir": "./out",
            "logging_dir": "./tb",
            "run_name": "test",
            "num_train_epochs": 1.0,
            "learning_rate": 0.0,
            "vit_lr": 0.0,
            "aligner_lr": 0.0,
        },
        "rlhf": rlhf_section,
    }
    return TrainingConfig.from_mapping(payload, prompts)


def test_config_loader_attaches_default_llm_kd_weight(monkeypatch):
    monkeypatch.setattr(
        "src.config.loader.TrainArguments", _DummyTrainArguments, raising=False
    )
    monkeypatch.setattr(
        "src.config.loader.RLHFArguments", _DummyRLHFArguments, raising=False
    )

    config = _build_minimal_training_config()
    train_args = ConfigLoader.build_train_arguments(config)

    assert isinstance(train_args, _DummyRLHFArguments)
    assert "llm_kd_weight" not in train_args.kwargs
    assert getattr(train_args, "llm_kd_weight") == pytest.approx(1.0)
    assert getattr(train_args.training_args, "llm_kd_weight") == pytest.approx(1.0)


def test_config_loader_respects_custom_llm_kd_weight(monkeypatch):
    monkeypatch.setattr(
        "src.config.loader.TrainArguments", _DummyTrainArguments, raising=False
    )
    monkeypatch.setattr(
        "src.config.loader.RLHFArguments", _DummyRLHFArguments, raising=False
    )

    config = _build_minimal_training_config(llm_kd_weight=0.25)
    train_args = ConfigLoader.build_train_arguments(config)

    assert getattr(train_args, "llm_kd_weight") == pytest.approx(0.25)
    assert getattr(train_args.training_args, "llm_kd_weight") == pytest.approx(0.25)
    assert "llm_kd_weight" not in train_args.kwargs


def test_config_loader_rejects_negative_llm_kd_weight(monkeypatch):
    monkeypatch.setattr(
        "src.config.loader.TrainArguments", _DummyTrainArguments, raising=False
    )
    monkeypatch.setattr(
        "src.config.loader.RLHFArguments", _DummyRLHFArguments, raising=False
    )

    config = _build_minimal_training_config(llm_kd_weight=-0.2)

    with pytest.raises(ValueError):
        ConfigLoader.build_train_arguments(config)


def test_config_loader_requires_teacher_for_llm_kd(monkeypatch):
    monkeypatch.setattr(
        "src.config.loader.TrainArguments", _DummyTrainArguments, raising=False
    )
    monkeypatch.setattr(
        "src.config.loader.RLHFArguments", _DummyRLHFArguments, raising=False
    )

    config = _build_minimal_training_config(llm_kd_weight=0.5, teacher_model=None)

    with pytest.raises(ValueError):
        ConfigLoader.build_train_arguments(config)


def test_config_loader_requires_teacher_for_visual_kd(monkeypatch):
    monkeypatch.setattr(
        "src.config.loader.TrainArguments", _DummyTrainArguments, raising=False
    )
    monkeypatch.setattr(
        "src.config.loader.RLHFArguments", _DummyRLHFArguments, raising=False
    )

    config = _build_minimal_training_config(
        llm_kd_weight=0.0, teacher_model=None, visual_enabled=True
    )

    with pytest.raises(ValueError):
        ConfigLoader.build_train_arguments(config)


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
    config_path = Path("configs/stage_3_gkd.yaml")
    if not config_path.exists():
        pytest.skip("configs/stage_3_gkd.yaml is not present in this checkout")
    cfg = ConfigLoader.load_yaml_with_extends(str(config_path))

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

    assert issubclass(trainer_cls, GKDTrainerWithMetrics)


def test_resolve_trainer_cls_falls_back(monkeypatch):
    class DummyTrainer:
        pass

    class DummyFactory:
        @staticmethod
        def get_trainer_cls(_):
            return DummyTrainer

    monkeypatch.setattr("src.sft.TrainerFactory", DummyFactory)

    args = SimpleNamespace(rlhf_type="gkd", trainer_variant=None)

    trainer_cls = resolve_trainer_cls(args)

    assert issubclass(trainer_cls, DummyTrainer)


def test_gkd_monitor_logs_losses(monkeypatch):
    captured_logs = {}

    def dummy_super_log(self, logs, start_time=None):
        captured_logs.update(logs)

    from src.trainers.gkd_monitor import _MsSwiftGKDTrainer

    monkeypatch.setattr(_MsSwiftGKDTrainer, "log", dummy_super_log)

    trainer = object.__new__(GKDTrainerWithMetrics)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._llm_kd_weight = 1.0
    trainer._metrics["train"]["llm_kd_loss"].extend([0.1, 0.2])
    trainer._metrics["train"]["sft_loss"].extend([0.4, 0.5])
    trainer._metrics["train"]["loss"].extend([0.5, 0.7])
    trainer._metrics["train"]["token_acc_correct"].extend([8.0, 9.0])
    trainer._metrics["train"]["token_acc_total"].extend([10.0, 10.0])
    trainer.model = SimpleNamespace(training=True)

    trainer.log({"eval/loss": 1.23})

    train_loss_avg = (0.5 + 0.7) / 2
    assert pytest.approx(captured_logs["llm_kd_loss"], rel=1e-6) == (0.1 + 0.2) / 2
    assert pytest.approx(captured_logs["sft_loss"], rel=1e-6) == (0.4 + 0.5) / 2
    assert pytest.approx(captured_logs["loss"], rel=1e-6) == train_loss_avg
    assert pytest.approx(captured_logs["token_acc"], rel=1e-6) == 17.0 / 20.0
    assert pytest.approx(captured_logs["eval/loss"], rel=1e-6) == 1.23
    assert "train/eval/loss" not in captured_logs
    assert not trainer._metrics["train"]

    trainer._metrics["eval"]["llm_kd_loss"].extend([0.3])
    trainer._metrics["eval"]["sft_loss"].extend([0.6])
    trainer._metrics["eval"]["loss"].extend([0.9])
    trainer._metrics["eval"]["token_acc_correct"].extend([4.0])
    trainer._metrics["eval"]["token_acc_total"].extend([5.0])

    eval_metrics_payload: dict[str, float] = {}
    trainer.log(eval_metrics_payload)

    assert pytest.approx(captured_logs["eval_llm_kd_loss"], rel=1e-6) == 0.3
    assert pytest.approx(captured_logs["eval_sft_loss"], rel=1e-6) == 0.6
    assert pytest.approx(captured_logs["eval_loss"], rel=1e-6) == 0.9
    assert pytest.approx(captured_logs["eval_token_acc"], rel=1e-6) == 0.8
    assert pytest.approx(eval_metrics_payload["eval_token_acc"], rel=1e-6) == 0.8
    assert pytest.approx(eval_metrics_payload["eval_loss"], rel=1e-6) == 0.9
    assert not trainer._metrics["eval"]


def test_gkd_monitor_coalesce_accuracy_metrics():
    payload = {
        "token_acc_correct": 8.0,
        "token_acc_total": 10.0,
        "eval_token_acc_correct": 5.0,
        "eval_token_acc_total": 5.0,
    }

    GKDTrainerWithMetrics._coalesce_accuracy_metrics(payload)

    assert payload["token_acc"] == pytest.approx(0.8)
    assert payload["eval_token_acc"] == pytest.approx(1.0)
    assert "token_acc_correct" not in payload
    assert "token_acc_total" not in payload
    assert "eval_token_acc_correct" not in payload
    assert "eval_token_acc_total" not in payload


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
    trainer._llm_kd_weight = 1.0
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
    assert trainer._metrics["train"]["llm_kd_loss"][0].item() == pytest.approx(0.5)
    assert trainer._metrics["train"]["token_acc_correct"][0].item() == pytest.approx(
        1.0
    )
    assert trainer._metrics["train"]["token_acc_total"][0].item() == pytest.approx(1.0)


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
    trainer._llm_kd_weight = 1.0
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
    trainer._llm_kd_weight = 0.0
    trainer.teacher_model = teacher_model  # type: ignore[assignment]
    trainer.beta = 0.5
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
    trainer._visual_kd_config = VisualKDConfig(
        enabled=True,
        vit=VisualKDTargetConfig(enabled=False),
        aligner=VisualKDTargetConfig(enabled=True, weight=0.1, distance="mse"),
        deepstack=VisualKDTargetConfig(enabled=True, weight=0.1, distance="mse"),
    )

    trainer.model = student_model
    trainer._register_visual_hooks()

    loss = trainer.compute_loss(student_model, {"labels": labels})
    assert isinstance(loss, torch.Tensor)

    # MSE merger = 0.5, deepstack = 0.25 => weighted sum = 0.05 + 0.025 = 0.075
    expected_weighted_loss = torch.tensor(0.075)
    assert pytest.approx(loss.item(), rel=1e-6) == pytest.approx(
        expected_weighted_loss.item()
    )
    logged_loss = trainer._metrics["train"]["vision_kd_loss"][0]
    assert pytest.approx(logged_loss.item(), rel=1e-6) == expected_weighted_loss.item()
    logged_total = trainer._metrics["train"]["loss"][0]
    assert pytest.approx(logged_total.item(), rel=1e-6) == expected_weighted_loss.item()


def test_ce_loss_unscaled_when_llm_kd_disabled():
    labels = torch.tensor([[1, 2, -100]])
    vocab_size = 4

    student_logits = torch.zeros((1, 3, vocab_size), requires_grad=True)
    ce_loss = torch.tensor(0.2, requires_grad=True)
    student_outputs = SimpleNamespace(logits=student_logits, loss=ce_loss)

    student_model = _StaticOutputModel(student_outputs)
    student_model.train()

    trainer = object.__new__(GKDTrainerWithMetrics)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._llm_kd_weight = 0.0
    trainer.beta = 0.5
    trainer.args = SimpleNamespace(sft_alpha=3.0)  # type: ignore[assignment]
    trainer.get_use_logits_to_keep = lambda default_value=True: False  # type: ignore[method-assign]
    trainer.prepare_logits_to_keep = lambda inputs: None  # type: ignore[method-assign]
    trainer._visual_kd_enabled = False

    loss = trainer.compute_loss(student_model, {"labels": labels})
    assert isinstance(loss, torch.Tensor)

    expected_loss = ce_loss.detach().item()
    assert pytest.approx(loss.item(), rel=1e-6) == expected_loss

    logged_sft = trainer._metrics["train"]["sft_loss"][0]
    logged_total = trainer._metrics["train"]["loss"][0]
    assert pytest.approx(logged_sft.item(), rel=1e-6) == expected_loss
    assert pytest.approx(logged_total.item(), rel=1e-6) == expected_loss


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
    trainer._llm_kd_weight = 0.0
    trainer.teacher_model = teacher_model  # type: ignore[assignment]
    trainer.beta = 0.5
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
    trainer._visual_kd_config = VisualKDConfig.disabled()

    trainer.model = student_model

    loss = trainer.compute_loss(student_model, {"labels": labels})
    assert isinstance(loss, torch.Tensor)

    assert loss.item() == pytest.approx(0.0)
    assert "vision_kd_loss" not in trainer._metrics["train"]


def test_gkd_eval_logs_llm_kd_loss():
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
    trainer._llm_kd_weight = 1.0
    trainer.teacher_model = teacher_model  # type: ignore[assignment]
    trainer.beta = 0.5
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
    trainer._visual_kd_config = VisualKDConfig.disabled()

    trainer.model = student_model

    loss = trainer.compute_loss(student_model, {"labels": labels})
    assert isinstance(loss, torch.Tensor)

    assert pytest.approx(loss.item(), rel=1e-6) == 0.25
    eval_metrics = trainer._metrics["eval"]
    assert pytest.approx(eval_metrics["llm_kd_loss"][0].item(), rel=1e-6) == 0.25
    assert pytest.approx(eval_metrics["loss"][0].item(), rel=1e-6) == 0.25
    assert pytest.approx(eval_metrics["sft_loss"][0].item(), rel=1e-6) == 0.0
