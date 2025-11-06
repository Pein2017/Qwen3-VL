"""Scoring utilities for Stage-B (signals, selection, judge)."""

from .selection import select_for_group
from .signals import attach_signals

__all__ = [
    "attach_signals",
    "select_for_group",
]

