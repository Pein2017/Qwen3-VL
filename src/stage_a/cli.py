#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entry point for Stage-A inference."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from .inference import run_stage_a_inference
from src.config.missions import SUPPORTED_MISSIONS, validate_mission


@dataclass(frozen=True)
class StageAConfig:
    checkpoint: str
    input_dir: str
    output_dir: str
    mission: str
    device: str = "cuda:0"
    batch_size: int = 8
    max_pixels: int = 786432
    max_new_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    include_mission_focus: bool = True
    verify_inputs: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "StageAConfig":
        return cls(
            checkpoint=args.checkpoint,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            mission=args.mission,
            device=args.device,
            batch_size=int(args.batch_size),
            max_pixels=int(args.max_pixels),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            include_mission_focus=not args.no_mission_focus,
            verify_inputs=bool(args.verify_inputs),
            log_level=args.log_level.upper(),
        )

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.max_pixels <= 0:
            raise ValueError("max_pixels must be a positive integer")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("log_level must be one of DEBUG|INFO|WARNING|ERROR")

        validate_mission(self.mission)

        input_path = Path(self.input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory not found: {input_path}")

        checkpoint_path = Path(self.checkpoint)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage-A per-image inference for Qwen3-VL (GRPO preparation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process one mission
  python -m src.stage_a.cli \\
    --checkpoint /data/models/qwen3vl-4b-stage2 \\
    --input_dir /data/bbu_groups \\
    --output_dir /data/stage_a_results \\
    --mission "挡风板安装检查" \\
    --device cuda:0

  # With custom generation params
  python -m src.stage_a.cli \\
    --checkpoint /data/models/qwen3vl-4b-stage2 \\
    --input_dir /data/bbu_groups \\
    --output_dir /data/stage_a_results \\
    --mission "BBU安装方式检查（正装）" \\
    --batch_size 16 \\
    --temperature 0.5

  # Sequential inference (no batching)
  python -m src.stage_a.cli \\
    --checkpoint /data/models/qwen3vl-4b-stage2 \\
    --input_dir /data/bbu_groups \\
    --output_dir /data/stage_a_results \\
    --mission "BBU接地线检查" \\
    --batch_size 1
        """,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="HuggingFace checkpoint path (Qwen3-VL model)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory with <mission>/{审核通过|审核不通过}/<group_id>/*.{jpg,jpeg,png}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--mission",
        type=str,
        required=True,
        choices=SUPPORTED_MISSIONS,
        help="Mission to process (must be one of the 4 supported missions)",
    )

    # Optional runtime parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (cuda:N or cpu). Default: cuda:0",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (1=sequential, 8=default). Default: 8",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=786432,
        help="Maximum pixels for image resizing (786432=1024x768, lower=faster). Default: 786432",
    )

    # Generation parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate. Default: 256",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (0.0=greedy). Default: 0.3",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p. Default: 0.9",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Repetition penalty. Default: 1.05",
    )
    parser.add_argument(
        "--no_mission_focus",
        action="store_true",
        help="Disable mission-specific focus in the user prompt to match training prompts.",
    )
    parser.add_argument(
        "--verify_inputs",
        action="store_true",
        help="Verify that images are loaded and encoded (logs checksums/sizes and grid/token counts)",
    )

    # Logging
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for Stage-A inference CLI."""
    args = parse_args()
    cfg = StageAConfig.from_namespace(args)

    try:
        cfg.validate()
    except Exception as exc:
        logging.basicConfig(
            level=logging.ERROR,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        logger = logging.getLogger("stage_a.cli")
        logger.error(str(exc))
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger = logging.getLogger("stage_a.cli")

    input_path = Path(cfg.input_dir)
    checkpoint_path = Path(cfg.checkpoint)

    # Build generation params
    gen_params = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": (cfg.temperature > 0.0),
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "repetition_penalty": cfg.repetition_penalty,
    }

    # Run inference
    try:
        run_stage_a_inference(
            checkpoint=str(checkpoint_path),
            input_dir=str(input_path),
            output_dir=cfg.output_dir,
            mission=cfg.mission,
            device=cfg.device,
            gen_params=gen_params,
            batch_size=cfg.batch_size,
            max_pixels=cfg.max_pixels,
            include_mission_focus=cfg.include_mission_focus,
            verify_inputs=cfg.verify_inputs,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
