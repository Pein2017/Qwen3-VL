#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-free Stage-B pipeline exports."""

from .config import StageBConfig, load_stage_b_config
from .ingest import ingest_stage_a
from .io import GuidanceRepository
from .sampling.prompts import build_messages
from .reflection import ReflectionEngine


# Lazy import to avoid RuntimeWarning when running as module: python -m src.stage_b.runner
# The runner module should not be imported at package init time when it's being executed as a script
def __getattr__(name: str):
    if name == "run_all":
        from .runner import run_all

        return run_all
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "StageBConfig",
    "load_stage_b_config",
    "GuidanceRepository",
    "ReflectionEngine",
    "ingest_stage_a",
    "build_messages",
    "run_all",
]
