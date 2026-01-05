import json
from pathlib import Path


from src.datasets.fusion import FusionConfig, _compute_target_quotas


def test_compute_target_quotas_with_ratios():
    targets = [
        FusionConfig._as_target(  # type: ignore[attr-defined]
            FusionConfig._parse_dataset_entry(  # type: ignore[attr-defined]
                {
                    "dataset": "bbu",
                    "train_jsonl": "t1.jsonl",
                    "name": "bbu",
                    "template": "target_dense_bbu",
                },
                require_ratio=False,
                allow_ratio=True,
            )[0],
            0.5,
        ),
        FusionConfig._as_target(  # type: ignore[attr-defined]
            FusionConfig._parse_dataset_entry(
                {
                    "dataset": "rru",
                    "train_jsonl": "t2.jsonl",
                    "name": "rru",
                    "template": "target_dense_rru",
                },
                require_ratio=False,
                allow_ratio=True,
            )[0],
            None,
        ),
        FusionConfig._as_target(  # type: ignore[attr-defined]
            FusionConfig._parse_dataset_entry(
                {
                    "dataset": "bbu",
                    "train_jsonl": "t3.jsonl",
                    "name": "t3",
                    "template": "target_dense_bbu",
                },
                require_ratio=False,
                allow_ratio=True,
            )[0],
            1.5,
        ),
    ]
    quotas, base = _compute_target_quotas(
        targets, {"bbu": 100, "rru": 200, "t3": 300}
    )
    assert base is None
    assert quotas["bbu"] == 50  # downsample
    assert quotas["rru"] == 200  # defaults to full coverage when ratio is None
    assert quotas["t3"] == 450  # upsample beyond pool size


def test_fusion_config_parses_multi_targets(tmp_path: Path):
    config_path = tmp_path / "fusion.yaml"
    payload = {
        "targets": [
            {
                "dataset": "bbu",
                "name": "bbu",
                "train_jsonl": str(tmp_path / "bbu.jsonl"),
                "val_jsonl": str(tmp_path / "bbu_val.jsonl"),
                "template": "target_dense_bbu",
                "ratio": 1.0,
            },
            {
                "dataset": "rru",
                "name": "rru",
                "train_jsonl": str(tmp_path / "rru.jsonl"),
                "val_jsonl": str(tmp_path / "rru_val.jsonl"),
                "template": "target_dense_rru",
                "ratio": 2.0,
            },
        ],
        "sources": [
            {
                "dataset": "lvis",
                "name": "lvis",
                "train_jsonl": str(tmp_path / "lvis.jsonl"),
                "template": "source_dense",
                "ratio": 0.1,
            }
        ],
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = FusionConfig.from_file(str(config_path))
    assert len(cfg.targets) == 2
    assert cfg.targets[0].name == "bbu"
    assert cfg.targets[1].ratio == 2.0
    assert len(cfg.sources) == 1


def test_fusion_config_overlay_appends_new_entries(tmp_path: Path):
    base_path = tmp_path / "base.json"
    overlay_path = tmp_path / "overlay.json"
    base_payload = {
        "targets": [
            {
                "dataset": "bbu",
                "name": "bbu",
                "train_jsonl": str(tmp_path / "bbu.jsonl"),
                "template": "target_dense_bbu",
            }
        ],
        "sources": [],
    }
    overlay_payload = {
        "extends": "base.json",
        "targets": [
            {
                "dataset": "rru",
                "name": "rru",
                "train_jsonl": str(tmp_path / "rru.jsonl"),
                "template": "target_dense_rru",
            }
        ],
    }
    base_path.write_text(json.dumps(base_payload), encoding="utf-8")
    overlay_path.write_text(json.dumps(overlay_payload), encoding="utf-8")

    cfg = FusionConfig.from_file(str(overlay_path))
    assert [spec.name for spec in cfg.targets] == ["bbu", "rru"]
