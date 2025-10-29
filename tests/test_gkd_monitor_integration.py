from collections import defaultdict
from types import SimpleNamespace

import pytest

from src.config.loader import ConfigLoader
from src.sft import resolve_trainer_cls
from src.trainers import GKDTrainerWithMetrics


def test_stage3_config_references_real_assets():
    cfg = ConfigLoader.load_yaml_with_extends("configs/stage_3_gkd.yaml")

    assert cfg["model"]["model"] == "output/stage_2_merged-10-25"
    assert (
        cfg["rlhf"]["teacher_model"] == "model_cache/models/Qwen/Qwen3-VL-4B-Instruct"
    )
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
    trainer._metrics["train"]["token_count"].extend([10, 20])
    trainer.model = SimpleNamespace(training=True)

    trainer.log({})

    assert (
        pytest.approx(captured_logs["train/kl_loss"], rel=1e-6)
        == (0.1 * 10 + 0.2 * 20) / 30
    )
    assert (
        pytest.approx(captured_logs["train/sft_loss"], rel=1e-6)
        == (0.4 * 10 + 0.5 * 20) / 30
    )
    assert (
        pytest.approx(captured_logs["train/loss"], rel=1e-6)
        == (0.5 * 10 + 0.7 * 20) / 30
    )
    assert (
        pytest.approx(captured_logs["train/token_accuracy"], rel=1e-6)
        == (0.8 * 10 + 0.9 * 20) / 30
    )
    assert captured_logs["train/token_count"] == pytest.approx(15)
    assert captured_logs["loss"] == pytest.approx(captured_logs["train/loss"])
    assert not trainer._metrics["train"]
