#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rule-search utilities for Stage-B (metric-gated guidance growth).

This module is intentionally model-free where possible so that metric/gate logic
can be unit-tested without running rollouts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import re

from .utils.chinese import normalize_spaces, to_simplified


@dataclass(frozen=True)
class TicketRolloutStats:
    ticket_key: str
    gt_label: str
    pass_count: int
    fail_count: int
    invalid_count: int
    total_samples: int
    majority_pred: str | None
    majority_correct: bool
    agreement: float
    difficulty: float
    hard_wrong: float
    verdict_samples: tuple[str | None, ...]


@dataclass(frozen=True)
class EvalMetrics:
    acc: float
    fn: int
    fp: int
    tp: int
    fn_rate: float
    fp_rate: float
    fn_over_tp: float
    fp_over_tp: float
    n: int


@dataclass(frozen=True)
class GateStats:
    relative_error_reduction: float
    changed_fraction: float
    bootstrap_prob: float


_SIGNATURE_PUNCT_RE = re.compile(
    r"[，,。.!！？;；:：\-—_()（）\[\]{}<>《》“”\"'·~]",
    re.UNICODE,
)


def normalize_rule_signature(text: str) -> str:
    simplified = to_simplified(text or "")
    simplified = normalize_spaces(simplified).lower()
    simplified = simplified.replace("如果", "若").replace("当", "若").replace("就", "则")
    simplified = _SIGNATURE_PUNCT_RE.sub(" ", simplified)
    simplified = normalize_spaces(simplified)
    return simplified.strip()


def _majority_pred(pass_count: int, fail_count: int) -> str | None:
    if pass_count <= 0 and fail_count <= 0:
        return None
    if pass_count > fail_count:
        return "pass"
    if fail_count > pass_count:
        return "fail"
    # deterministic tie-break: fail-first (conservative)
    return "fail"


def build_ticket_stats(
    *,
    ticket_key: str,
    gt_label: str,
    verdicts: Sequence[str | None],
) -> TicketRolloutStats:
    pass_count = 0
    fail_count = 0
    invalid = 0
    for verdict in verdicts:
        if verdict == "pass":
            pass_count += 1
        elif verdict == "fail":
            fail_count += 1
        else:
            invalid += 1

    total = len(verdicts)
    valid = pass_count + fail_count
    majority = _majority_pred(pass_count, fail_count)
    majority_correct = bool(majority is not None and majority == gt_label)

    agreement = 0.0
    difficulty = 0.5
    hard_wrong = 0.0
    if valid > 0:
        agreement = max(pass_count, fail_count) / valid
        difficulty = 1.0 - agreement
        if majority is not None and majority != gt_label:
            hard_wrong = agreement

    return TicketRolloutStats(
        ticket_key=ticket_key,
        gt_label=gt_label,
        pass_count=pass_count,
        fail_count=fail_count,
        invalid_count=invalid,
        total_samples=total,
        majority_pred=majority,
        majority_correct=majority_correct,
        agreement=agreement,
        difficulty=difficulty,
        hard_wrong=hard_wrong,
        verdict_samples=tuple(verdicts),
    )


def compute_metrics(stats: Iterable[TicketRolloutStats]) -> EvalMetrics:
    correct = 0
    fn = 0
    fp = 0
    tp = 0
    total = 0
    for entry in stats:
        total += 1
        pred = entry.majority_pred
        gt = entry.gt_label
        if pred is None:
            if gt == "pass":
                fn += 1
            elif gt == "fail":
                fp += 1
            continue

        if pred == gt:
            correct += 1
            if pred == "pass":
                tp += 1
        elif pred == "pass" and gt == "fail":
            fp += 1
        elif pred == "fail" and gt == "pass":
            fn += 1

    acc = correct / total if total else 0.0
    fn_rate = fn / total if total else 0.0
    fp_rate = fp / total if total else 0.0
    fn_over_tp = fn / tp if tp > 0 else float("inf") if fn > 0 else 0.0
    fp_over_tp = fp / tp if tp > 0 else float("inf") if fp > 0 else 0.0
    return EvalMetrics(
        acc=acc,
        fn=fn,
        fp=fp,
        tp=tp,
        fn_rate=fn_rate,
        fp_rate=fp_rate,
        fn_over_tp=fn_over_tp,
        fp_over_tp=fp_over_tp,
        n=total,
    )


def relative_error_reduction(
    base: EvalMetrics,
    new: EvalMetrics,
    *,
    eps: float = 1e-12,
) -> float:
    """Compute relative error reduction from accuracy metrics."""
    base_err = max(0.0, 1.0 - base.acc)
    new_err = max(0.0, 1.0 - new.acc)
    denom = max(base_err, eps)
    return (base_err - new_err) / denom


def changed_fraction(
    base_stats: Mapping[str, TicketRolloutStats],
    new_stats: Mapping[str, TicketRolloutStats],
) -> float:
    keys = sorted(set(base_stats.keys()) & set(new_stats.keys()))
    if not keys:
        return 0.0
    changed = 0
    for key in keys:
        if base_stats[key].majority_pred != new_stats[key].majority_pred:
            changed += 1
    return changed / len(keys)


def bootstrap_rer_probability(
    base_correct: Sequence[int],
    new_correct: Sequence[int],
    *,
    threshold: float,
    iterations: int,
    seed: int,
    eps: float = 1e-12,
) -> float:
    """Bootstrap P(RER >= threshold) using ticket-level resampling."""

    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if len(base_correct) != len(new_correct):
        raise ValueError("base_correct and new_correct must have same length")

    n = len(base_correct)
    if n == 0:
        return 0.0

    import random

    rng = random.Random(int(seed))
    passed = 0

    for _ in range(iterations):
        # Resample tickets with replacement
        correct0 = 0
        correct1 = 0
        for _j in range(n):
            idx = rng.randrange(n)
            correct0 += 1 if base_correct[idx] else 0
            correct1 += 1 if new_correct[idx] else 0
        acc0 = correct0 / n
        acc1 = correct1 / n
        base_err = max(0.0, 1.0 - acc0)
        new_err = max(0.0, 1.0 - acc1)
        rer = (base_err - new_err) / max(base_err, eps)
        if rer >= threshold:
            passed += 1

    return passed / iterations


def pick_reflection_ticket_keys(
    stats_by_ticket: Mapping[str, TicketRolloutStats],
    *,
    reflect_size: int,
    reflect_order: str = "hard_first",
) -> list[str]:
    """Pick high-value mismatches for the proposer.

    Priority (reflect_order=hard_first):
    1) high-confidence wrong (hard_wrong high, majority_pred != gt)
    2) then most ambiguous wrong (difficulty high)

    Priority (reflect_order=easy_first):
    1) lower difficulty (agreement high)
    2) then higher hard_wrong (more confident mistakes)
    """

    mismatches: list[TicketRolloutStats] = []
    for entry in stats_by_ticket.values():
        if entry.majority_pred is None:
            continue
        if entry.majority_pred != entry.gt_label:
            mismatches.append(entry)

    if reflect_order == "easy_first":
        mismatches.sort(key=lambda x: (x.difficulty, -x.hard_wrong))
    else:
        mismatches.sort(key=lambda x: (x.hard_wrong, x.difficulty), reverse=True)
    return [m.ticket_key for m in mismatches[: max(0, int(reflect_size))]]


def build_gate_stats(
    *,
    base_stats: Mapping[str, TicketRolloutStats],
    new_stats: Mapping[str, TicketRolloutStats],
    rer_threshold: float,
    bootstrap_iterations: int,
    bootstrap_min_prob: float,
    bootstrap_seed: int,
    max_changed_fraction: float,
) -> tuple[GateStats, bool]:
    """Compute gate stats and return (stats, passed)."""

    base_metrics = compute_metrics(base_stats.values())
    new_metrics = compute_metrics(new_stats.values())

    rer = relative_error_reduction(base_metrics, new_metrics)
    changed = changed_fraction(base_stats, new_stats)

    keys = sorted(set(base_stats.keys()) & set(new_stats.keys()))
    base_correct = [1 if base_stats[k].majority_correct else 0 for k in keys]
    new_correct = [1 if new_stats[k].majority_correct else 0 for k in keys]
    bootstrap_prob = bootstrap_rer_probability(
        base_correct,
        new_correct,
        threshold=rer_threshold,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )

    stats = GateStats(
        relative_error_reduction=rer,
        changed_fraction=changed,
        bootstrap_prob=bootstrap_prob,
    )
    passed = bool(
        rer >= rer_threshold
        and changed <= max_changed_fraction
        and bootstrap_prob >= bootstrap_min_prob
    )
    return stats, passed


__all__ = [
    "TicketRolloutStats",
    "EvalMetrics",
    "GateStats",
    "normalize_rule_signature",
    "build_ticket_stats",
    "compute_metrics",
    "relative_error_reduction",
    "changed_fraction",
    "bootstrap_rer_probability",
    "pick_reflection_ticket_keys",
    "build_gate_stats",
]
