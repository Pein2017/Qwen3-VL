"""Deterministic signal extraction for Stage-B candidates."""

from __future__ import annotations

from collections import Counter
from typing import List, Mapping, Sequence

from .config import SignalsConfig
from .types import (
    DeterministicSignals,
    GroupTicket,
    ParsedTrajectory,
    TrajectoryWithSignals,
)

DEFAULT_WEIGHTS: Mapping[str, float] = {
    "label_match": 1.0,
    "self_consistency": 0.5,
    "candidate_agreement": 0.25,
    "confidence": 0.25,
}


def _metric_value(name: str, signals: DeterministicSignals) -> float:
    if name == "label_match":
        if signals.label_match is None:
            return 0.0
        return 1.0 if signals.label_match else 0.0
    if name == "self_consistency":
        return signals.self_consistency or 0.0
    if name == "candidate_agreement":
        if signals.candidate_agreement is None:
            return 0.0
        return 1.0 if signals.candidate_agreement else 0.0
    if name == "confidence":
        return signals.confidence or 0.0
    return 0.0


def _semantic_advantage(
    signals: DeterministicSignals, weights: Mapping[str, float]
) -> float:
    total = 0.0
    for key, weight in weights.items():
        if weight == 0:
            continue
        total += weight * _metric_value(key, signals)
    return total


def attach_signals(
    ticket: GroupTicket,
    candidates: Sequence[ParsedTrajectory],
    config: SignalsConfig,
) -> List[TrajectoryWithSignals]:
    """Attach deterministic signals to parsed trajectories."""

    verdict_counts = Counter(
        candidate.verdict for candidate in candidates if candidate.verdict is not None
    )
    total_considered = sum(verdict_counts.values())
    majority_verdict = verdict_counts.most_common(1)[0][0] if verdict_counts else None

    weights: Mapping[str, float] = config.weights or DEFAULT_WEIGHTS

    annotated: List[TrajectoryWithSignals] = []
    for candidate in candidates:
        label_match = None
        if candidate.verdict is not None:
            # Convert Chinese verdict to English label for comparison
            # "通过" -> "pass", "不通过" -> "fail"
            verdict_as_label = "pass" if candidate.verdict == "通过" else "fail"
            label_match = verdict_as_label == ticket.label

        if (
            config.enable_consistency
            and candidate.verdict is not None
            and total_considered > 0
        ):
            same_count = verdict_counts.get(candidate.verdict, 0)
            self_consistency = same_count / total_considered
            candidate_agreement = (
                majority_verdict == candidate.verdict
                if majority_verdict is not None
                else None
            )
        else:
            self_consistency = None
            candidate_agreement = None

        confidence_value = candidate.confidence if config.store_confidence else None

        # Compute label_trust based on available signals
        # In the baseline, label_trust is derived from confidence and label_match
        # For more sophisticated trust scoring, this would integrate verifier plugins
        label_trust = None
        if confidence_value is not None and label_match is not None:
            # Simple heuristic: trust is higher when label matches and confidence is high
            if label_match:
                label_trust = confidence_value
            else:
                # Lower trust when label doesn't match (even if confidence is high)
                label_trust = max(0.0, confidence_value - 0.3)
        elif label_match is not None:
            # If we have label_match but no confidence, use label_match as proxy
            label_trust = 1.0 if label_match else 0.3

        signals = DeterministicSignals(
            label_match=label_match,
            self_consistency=self_consistency,
            candidate_agreement=candidate_agreement,
            confidence=confidence_value,
            label_trust=label_trust,
            semantic_advantage=0.0,
        )
        semantic_score = _semantic_advantage(signals, weights)
        signals = DeterministicSignals(
            label_match=signals.label_match,
            self_consistency=signals.self_consistency,
            candidate_agreement=signals.candidate_agreement,
            confidence=signals.confidence,
            label_trust=signals.label_trust,
            semantic_advantage=semantic_score,
        )

        annotated.append(TrajectoryWithSignals(parsed=candidate, signals=signals))

    return annotated


__all__ = ["attach_signals"]
