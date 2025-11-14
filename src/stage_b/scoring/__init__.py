"""Scoring utilities for Stage-B (selection only).

Legacy modules (judge, signals) were removed; attach_signals lives at stage_b.signals.
"""

from .selection import select_for_group

__all__ = [
    "select_for_group",
]

