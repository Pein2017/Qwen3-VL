#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility helpers for seeding random number generators."""

from __future__ import annotations

import os
import random
from typing import Final

__all__: Final = ["seed_everything"]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs (best effort)."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:  # pragma: no cover - optional dependency
        pass

    try:
        import torch  # type: ignore

        _ = torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - device dependent
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch may be unavailable
        pass
