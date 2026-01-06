"""Dense GRPO reward implementations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, List, cast

from swift.plugin.orm import ORM

from .context import DenseSample, build_samples
from .matching import (
    DEFAULT_COCO_THRESHOLDS,
    DEFAULT_LINE_TOL,
    DEFAULT_PRIMARY_THRESHOLD,
    build_overlap_matrix,
    fbeta,
    greedy_match,
)
from .parsing import (
    JsonDuplicateKeyError,
    check_dense_completion_format,
    is_dense_mode,
)


def _resolve_payload_from_kwargs(
    completions: Any | None, kwargs: Mapping[str, object]
) -> tuple[Iterable[Any], Any, Any]:
    """Resolve (completions, metadata, assistant_payload) from ms-swift kwargs."""

    payload: object | None = kwargs.get("payload")
    metadata: object | None = kwargs.get("metadata")
    assistant_payload: object | None = kwargs.get("assistant_payload")

    if completions is None:
        payload = kwargs.get("completions") or payload
        if payload is None:
            return ([], metadata, assistant_payload)
        if isinstance(payload, Mapping):
            payload_map = cast(Mapping[str, object], payload)
            maybe_completions = payload_map.get("completions")
            if isinstance(maybe_completions, Iterable) and not isinstance(
                maybe_completions, (str, bytes)
            ):
                completions = cast(Iterable[Any], maybe_completions)
                metadata = payload_map.get("metadata")
                assistant_payload = payload_map.get("assistant_payload")
        else:
            completions = payload

    if completions is None:
        completions_iterable: Iterable[Any] = []
    elif isinstance(completions, Iterable) and not isinstance(completions, (str, bytes)):
        completions_iterable = cast(Iterable[Any], completions)
    else:
        completions_iterable = [completions]

    return (completions_iterable, metadata, assistant_payload)


class DenseReward(ORM):
    """Base class for dense-mode GRPO rewards."""

    def score(self, sample: DenseSample) -> float:
        raise NotImplementedError

    def __call__(self, completions: Any | None = None, **kwargs: object) -> List[float]:
        completions_iter, metadata, assistant_payload = _resolve_payload_from_kwargs(
            completions, kwargs
        )
        samples = build_samples(completions_iter, metadata, assistant_payload)
        return [float(self.score(sample)) for sample in samples]


class DenseFormatReward(DenseReward):
    """Checks dense format: strict 2-line header + JSON object line (cheap; no JSON parsing)."""

    def score(self, sample: DenseSample) -> float:
        if not check_dense_completion_format(lines=sample.lines, meta=sample.metadata):
            return 0.0
        return 1.0


class DenseParseSchemaStrictReward(DenseReward):
    """Strict JSON + schema validation (duplicate keys, desc, exactly one geometry, valid poly/line).

    This is a guardrail reward: success is neutral, failures receive a negative penalty.
    """

    def score(self, sample: DenseSample) -> float:
        if not is_dense_mode(sample.metadata):
            return 0.0
        try:
            sample.pred_strict()
        except JsonDuplicateKeyError:
            return -1.0
        except Exception:
            return -1.0
        # Guardrail reward: success is neutral; failures get a negative penalty.
        return 0.0


class DenseLocalizationMeanFBetaReward(DenseReward):
    """Mean Fβ across COCO IoU thresholds (localization-only; recall-biased by default)."""

    beta: float = 2.0

    def score(self, sample: DenseSample) -> float:
        if not is_dense_mode(sample.metadata):
            return 0.0
        try:
            pred = sample.pred_strict().objects
            gt = sample.gt_payload().objects
        except Exception:
            return 0.0
        if not gt and not pred:
            return 0.0

        overlap = build_overlap_matrix(gt, pred, line_tol=DEFAULT_LINE_TOL)
        scores: list[float] = []
        for thr in DEFAULT_COCO_THRESHOLDS:
            matches = greedy_match(
                overlap,
                threshold=float(thr),
                gt_objects=gt,
                pred_objects=pred,
                require_category_match=False,
            )
            scores.append(
                fbeta(
                    matched=len(matches),
                    gt_total=len(gt),
                    pred_total=len(pred),
                    beta=float(self.beta),
                )
            )
        return (sum(scores) / len(scores)) if scores else 0.0


class DenseLocalizationSoftRecallReward(DenseReward):
    """Smooth recall shaping using per-GT best overlap (missing-object reduction)."""

    def score(self, sample: DenseSample) -> float:
        if not is_dense_mode(sample.metadata):
            return 0.0
        try:
            pred = sample.pred_strict().objects
            gt = sample.gt_payload().objects
        except Exception:
            return 0.0
        if not gt:
            return 0.0

        if not pred:
            return 0.0

        overlap = build_overlap_matrix(gt, pred, line_tol=DEFAULT_LINE_TOL)
        best: list[float] = []
        for gi, row in enumerate(overlap):
            _ = gi
            best.append(max(row) if row else 0.0)
        return (sum(best) / len(best)) if best else 0.0


class DenseCategoryMeanF1Reward(DenseReward):
    """Category-aware mean F1 across thresholds (requires `类别` equality on matched pairs)."""

    def score(self, sample: DenseSample) -> float:
        if not is_dense_mode(sample.metadata):
            return 0.0
        try:
            pred = sample.pred_strict().objects
            gt = sample.gt_payload().objects
        except Exception:
            return 0.0
        if not gt and not pred:
            return 0.0

        overlap = build_overlap_matrix(gt, pred, line_tol=DEFAULT_LINE_TOL)
        f1s: list[float] = []
        for thr in DEFAULT_COCO_THRESHOLDS:
            matches = greedy_match(
                overlap,
                threshold=float(thr),
                gt_objects=gt,
                pred_objects=pred,
                require_category_match=True,
            )
            prec = (len(matches) / len(pred)) if pred else 0.0
            rec = (len(matches) / len(gt)) if gt else 0.0
            denom = prec + rec
            f1 = (2.0 * prec * rec / denom) if denom > 0 else 0.0
            f1s.append(f1)
        return (sum(f1s) / len(f1s)) if f1s else 0.0


def _attr_weight(key: str, *, category: str) -> float:
    if key == "可见性":
        return 0.1
    if key == "站点距离":
        return 4.0
    return 1.0


def _is_bonus_key(key: str) -> bool:
    return key in {"文本", "备注"}


def _is_station_distance_category(category: str) -> bool:
    return category == "站点距离"


def _as_int_strict(value: str) -> int | None:
    if not value:
        return None
    if not value.isdigit():
        return None
    try:
        return int(value)
    except Exception:
        return None


def _pair_attr_score(gt_obj: Any, pred_obj: Any) -> float:
    gt_desc = gt_obj.desc
    pred_desc = pred_obj.desc

    category = gt_obj.category

    core_total = 0.0
    core_matched = 0.0
    bonus_total = 0.0
    bonus_matched = 0.0

    for attr in gt_desc.attrs:
        key = attr.key
        if not key or key == "类别":
            continue
        gt_val = attr.value
        if not gt_val:
            continue

        pred_val = pred_desc.get(key)

        if _is_bonus_key(key):
            bonus_total += 6.0
            if pred_val and pred_val == gt_val:
                bonus_matched += 6.0
            continue

        weight = _attr_weight(key, category=category)
        core_total += weight

        if key == "站点距离" and _is_station_distance_category(category):
            gt_i = _as_int_strict(gt_val)
            pred_i = _as_int_strict(pred_val)
            if gt_i is not None and pred_i is not None and gt_i == pred_i:
                core_matched += weight
            continue

        if pred_val and pred_val == gt_val:
            core_matched += weight

    core_recall = (core_matched / core_total) if core_total > 0 else 0.0
    bonus_add = (
        (bonus_matched / (core_total + bonus_total))
        if bonus_total > 0 and (core_total + bonus_total) > 0
        else 0.0
    )
    return float(core_recall + bonus_add)


class DenseAttrWeightedRecallReward(DenseReward):
    """Attribute scoring on matched pairs (exact string match; business weighting)."""

    def score(self, sample: DenseSample) -> float:
        if not is_dense_mode(sample.metadata):
            return 0.0
        try:
            pred = sample.pred_strict().objects
            gt = sample.gt_payload().objects
        except Exception:
            return 0.0

        if not gt or not pred:
            return 0.0

        overlap = build_overlap_matrix(gt, pred, line_tol=DEFAULT_LINE_TOL)
        matches = greedy_match(
            overlap,
            threshold=float(DEFAULT_PRIMARY_THRESHOLD),
            gt_objects=gt,
            pred_objects=pred,
            require_category_match=False,
        )
        if not matches:
            return 0.0

        scores: list[float] = []
        for gi, pi in matches:
            scores.append(_pair_attr_score(gt[gi], pred[pi]))
        return (sum(scores) / len(scores)) if scores else 0.0


__all__ = [
    "DenseAttrWeightedRecallReward",
    "DenseCategoryMeanF1Reward",
    "DenseFormatReward",
    "DenseLocalizationMeanFBetaReward",
    "DenseLocalizationSoftRecallReward",
    "DenseParseSchemaStrictReward",
]
