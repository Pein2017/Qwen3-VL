#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script to run inference on a single image, simulating stage_a.sh behavior."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path (must be before imports from project modules)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from PIL import Image  # noqa: E402

from data_conversion.utils.exif_utils import apply_exif_orientation  # noqa: E402
from src.stage_a.inference import infer_one_image, load_generation_engine  # noqa: E402
from src.stage_a.prompts import build_system_prompt, build_user_prompt  # noqa: E402
from src.utils import configure_logging, get_logger  # noqa: E402

# Configuration matching stage_a.sh
CHECKPOINT = "output/12-23/summary_merged/epoch_2-bbu_rru-more_irrelevant-ocr"
MISSION = "挡风板安装检查"
DEVICE = "cuda:0"
MAX_PIXELS = 1048576
BATCH_SIZE = 64  # Not used for single image, but kept for reference
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0001
TOP_P = 1.0
REP_PENALTY = 1.05

# Image path
IMAGE_PATH = "group_data/bbu_scene_2.0_order/挡风板安装检查/审核不通过/QC-TEMP-20241022-0015062/QC-TEMP-20241022-0015062_4176693.jpeg"


def main():
    """Run inference on a single image."""
    # Configure logging
    configure_logging(level=logging.INFO, debug=False, verbose=False)
    logger = get_logger("test_single_image")

    # Resolve paths
    project_root = Path(__file__).parent.parent
    checkpoint_path = project_root / CHECKPOINT
    image_path = project_root / IMAGE_PATH

    # Validate paths
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Single Image Inference Test")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Image: {image_path}")
    logger.info(f"Mission: {MISSION}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Max pixels: {MAX_PIXELS}")
    logger.info(f"Temperature: {TEMPERATURE}")
    logger.info(f"Max new tokens: {MAX_NEW_TOKENS}")
    logger.info("=" * 70)

    # Load generation engine
    logger.info("Loading generation engine...")
    engine = load_generation_engine(
        str(checkpoint_path), DEVICE, max_pixels=MAX_PIXELS
    )
    logger.info("Engine loaded successfully")

    # Build prompts (matching stage_a.sh behavior with mission focus)
    logger.info("Building prompts...")
    user_text = build_user_prompt(MISSION)
    system_text = build_system_prompt(MISSION)

    logger.info(f"System prompt length: {len(system_text)} chars")
    logger.info(f"User prompt length: {len(user_text)} chars")

    # Load and prepare image
    logger.info(f"Loading image: {image_path}")
    image = apply_exif_orientation(Image.open(image_path))
    logger.info(f"Image size: {image.size} (width x height)")

    # Build generation config (matching stage_a.sh)
    gen_config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": (TEMPERATURE > 0.0),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REP_PENALTY,
    }

    logger.info("Generation config:")
    for key, value in gen_config.items():
        logger.info(f"  {key}: {value}")

    # Run inference
    logger.info("Running inference...")
    try:
        raw_text, clean_text = infer_one_image(
            engine=engine,
            image=image,
            user_text=user_text,
            gen_config=gen_config,
            verify=False,
            system_prompt=system_text,
        )

        # Print results
        logger.info("=" * 70)
        logger.info("Inference Results")
        logger.info("=" * 70)
        logger.info(f"Raw text length: {len(raw_text)} chars")
        logger.info(f"Clean text length: {len(clean_text)} chars")
        logger.info("")
        logger.info("Raw output:")
        logger.info(f"  {raw_text}")
        logger.info("")
        logger.info("Clean output (summary):")
        logger.info(f"  {clean_text}")
        logger.info("=" * 70)

        # Also print to stdout for easy capture
        print("\n" + "=" * 70)
        print("SUMMARY OUTPUT:")
        print("=" * 70)
        print(clean_text)
        print("=" * 70)

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
