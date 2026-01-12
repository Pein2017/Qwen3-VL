#!/usr/bin/env python3
"""
Fast, model-free validation for SFT YAML configs.

This only loads/parses the YAML into typed dataclasses (no model weights).
Useful for CI checks and quick iteration before launching `scripts/train.sh`.

Usage:
  conda run -n ms python scripts/validate_sft_config.py --config configs/debug.yaml
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigLoader, TrainingConfig
from src.config.grpo import validate_grpo_config
from src.datasets.fusion import FusionConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate SFT training YAML config (no model)"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to training YAML"
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Optional base YAML for inheritance",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if args.base_config is not None and not args.base_config.is_file():
        raise FileNotFoundError(f"Base config not found: {args.base_config}")

    raw = ConfigLoader.load_yaml_with_extends(str(args.config))
    if args.base_config is not None:
        base_raw = ConfigLoader.load_yaml_with_extends(str(args.base_config))
        raw = ConfigLoader.merge_configs(base_raw, raw)

    prompts = ConfigLoader.resolve_prompts(raw)
    training_cfg = TrainingConfig.from_mapping(raw, prompts)

    validate_grpo_config(training_cfg)
    fusion_path = training_cfg.custom.fusion_config
    if fusion_path:
        fusion_file = Path(fusion_path)
        if not fusion_file.is_file():
            raise FileNotFoundError(f"Fusion config not found: {fusion_path}")
        FusionConfig.from_file(str(fusion_file))
    custom = training_cfg.custom
    print("[OK] Parsed training config.")
    print(f"  output_variant: {custom.output_variant}")
    print(f"  train_jsonl:    {custom.train_jsonl}")
    print(f"  val_jsonl:      {custom.val_jsonl}")
    print(f"  emit_norm:      {custom.emit_norm}")
    print(f"  json_format:    {custom.json_format}")
    print(f"  use_summary:    {custom.use_summary}")
    print("  trainer:        (skipped; this script does not build TrainArguments)")


if __name__ == "__main__":
    main()
