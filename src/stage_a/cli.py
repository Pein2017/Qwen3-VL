#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entry point for Stage-A inference."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .inference import run_stage_a_inference
from .prompts import SUPPORTED_MISSIONS, validate_mission


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
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    logger = logging.getLogger("stage_a.cli")
    
    # Validate mission
    try:
        validate_mission(args.mission)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Validate paths
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        sys.exit(1)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Build generation params
    gen_params = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": (args.temperature > 0.0),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    
    # Run inference
    try:
        run_stage_a_inference(
            checkpoint=str(checkpoint_path),
            input_dir=str(input_path),
            output_dir=args.output_dir,
            mission=args.mission,
            device=args.device,
            gen_params=gen_params,
            batch_size=args.batch_size,
            max_pixels=args.max_pixels,
            include_mission_focus=(not args.no_mission_focus),
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

