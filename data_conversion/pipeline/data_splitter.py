#!/usr/bin/env python3
"""
Data Splitter for Train/Validation Split

Handles splitting samples into training and validation sets with proper
randomization and reproducible results.
"""

import logging
import random
import sys
from typing import List, Tuple


# Configure UTF-8 encoding for stdout/stderr if supported
try:
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
except (AttributeError, TypeError):
    pass

try:
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")
except (AttributeError, TypeError):
    pass

logger = logging.getLogger(__name__)


class DataSplitter:
    """Splits data into train and validation sets."""

    def __init__(self, val_ratio: float = 0.1, seed: int = 42):
        if not 0.0 < val_ratio < 1.0:
            raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

        self.val_ratio = val_ratio
        self.seed = seed

        logger.info(f"Initialized DataSplitter with val_ratio={val_ratio}, seed={seed}")

    def split(self, samples: List[dict]) -> Tuple[List[dict], List[dict]]:
        """
        Split samples into train and validation sets.

        Args:
            samples: List of sample dictionaries

        Returns:
            Tuple of (train_samples, val_samples)
        """
        if not samples:
            raise ValueError("Cannot split empty sample list")

        logger.info(f"Splitting {len(samples)} samples with ratio {self.val_ratio}")

        # Create a copy to avoid modifying original list
        samples_copy = samples.copy()

        # Shuffle with fixed seed for reproducibility
        random.seed(self.seed)
        random.shuffle(samples_copy)

        # Calculate split point
        val_size = int(len(samples_copy) * self.val_ratio)

        # Ensure at least 1 validation sample if we have any samples
        if val_size == 0 and len(samples_copy) > 1:
            val_size = 1

        # Split the data
        val_samples = samples_copy[:val_size]
        train_samples = samples_copy[val_size:]

        logger.info(
            f"Split result: {len(train_samples)} training, {len(val_samples)} validation samples"
        )

        # Validate split
        if len(train_samples) == 0:
            raise ValueError("Training set is empty after split")

        return train_samples, val_samples

    def get_split_info(self, samples: List[dict]) -> dict:
        """Get information about how the split would be performed without actually splitting."""
        total_samples = len(samples)
        val_size = int(total_samples * self.val_ratio)

        # Ensure at least 1 validation sample if we have any samples
        if val_size == 0 and total_samples > 1:
            val_size = 1

        train_size = total_samples - val_size

        return {
            "total_samples": total_samples,
            "train_samples": train_size,
            "val_samples": val_size,
            "actual_val_ratio": val_size / total_samples if total_samples > 0 else 0.0,
        }
