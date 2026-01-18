"""Summary GRPO reward implementations."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, List, Tuple, cast

from swift.plugin.orm import ORM

from .context import SummarySample
from .facts import (
    extract_summary_fact_counts,
    f1_from_fact_counts,
    f1_from_sets,
    filter_fact_counts,
    tversky_from_fact_counts,
)
from .parsing import (
    _HEADER_PATTERN,
    _IRRELEVANT_TEXT,
    JsonDuplicateKeyError,
    ensure_list,
    loads_json_rejecting_duplicate_keys,
    normalize_bbu_ocr_text,
    normalize_free_text,
    normalize_summary,
)
from src.utils import require_mapping
from src.utils.unstructured import UnstructuredMapping


class SummaryReward(ORM):
    """Base class for summary GRPO rewards."""

    def score(self, sample: SummarySample) -> float:
        raise NotImplementedError

    def __call__(self, completions: Any | None = None, **kwargs: object) -> List[float]:
        """ms-swift GRPO entrypoint.

        ms-swift calls rewards as `reward_func(completions, **kwargs)` where:
        - `completions` is a list of decoded assistant strings
        - `kwargs` contains batched per-row fields such as `metadata`

        Backward compatibility: also supports the legacy `payload=` / `completions=` kwargs
        structure used by older call sites.
        """

        payload: object | None = kwargs.get("payload")
        metadata: object | None = kwargs.get("metadata")

        if completions is None:
            payload = kwargs.get("completions") or payload
            if payload is None:
                return []
            if isinstance(payload, Mapping):
                payload_map = cast(Mapping[str, object], payload)
                maybe_completions = payload_map.get("completions")
                if isinstance(maybe_completions, Iterable) and not isinstance(
                    maybe_completions, (str, bytes)
                ):
                    completions = cast(Iterable[Any], maybe_completions)
                    metadata = payload_map.get("metadata")
            else:
                completions = payload

        if completions is None:
            completions_iterable: Iterable[Any] = []
        elif isinstance(completions, Iterable) and not isinstance(
            completions, (str, bytes)
        ):
            completions_iterable = cast(Iterable[Any], completions)
        else:
            completions_iterable = [completions]

        completions_list = list(completions_iterable)
        metas = ensure_list(metadata, len(completions_list))

        rewards: list[float] = [0.0] * len(completions_list)
        for idx, (completion, meta) in enumerate(zip(completions_list, metas)):
            # Mixed-mode safety: summary rewards MUST no-op on dense samples.
            if (
                isinstance(meta, Mapping)
                and cast(Mapping[str, object], meta).get("_fusion_mode") == "dense"
            ):
                rewards[idx] = 0.0
                continue
            rewards[idx] = float(
                self.score(SummarySample.from_inputs(completion, meta))
            )

        return rewards


class SummaryFormatReward(SummaryReward):
    """Checks summary format: header+JSON for non-irrelevant, single-line for irrelevant."""

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 1.0 if sample.text == _IRRELEVANT_TEXT else 0.0
        lines = sample.lines
        if len(lines) != 2:
            return 0.0
        header = lines[0].strip()
        json_line = lines[1].strip()
        if not _HEADER_PATTERN.match(header):
            return 0.0
        if not json_line.endswith("}"):
            return 0.0
        return 1.0


class SummaryHeaderReward(SummaryReward):
    """Validates <DOMAIN> and <TASK=SUMMARY> for non-irrelevant samples."""

    def score(self, sample: SummarySample) -> float:
        lines = sample.lines
        if sample.is_irrelevant:
            if lines and lines[0].strip().startswith("<DOMAIN="):
                return -1.0
            if "<TASK=" in sample.text:
                return -1.0
            return 0.0
        if not lines:
            return 0.0
        if sample.domain_token is None:
            return 0.0
        expected = f"<DOMAIN={sample.domain_token}>, <TASK=SUMMARY>"
        return 1.0 if lines[0].strip() == expected else 0.0


class SummaryStrictPenaltyReward(SummaryReward):
    """Penalize non-irrelevant outputs that violate the strict 2-line header contract."""

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0
        lines = sample.lines
        if len(lines) != 2:
            return -1.0
        if sample.domain_token is None:
            return 0.0
        expected = f"<DOMAIN={sample.domain_token}>, <TASK=SUMMARY>"
        return 0.0 if lines[0].strip() == expected else -1.0


class SummaryParsePenalty(SummaryReward):
    """Applies a negative reward only when JSON parsing fails."""

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0
        lines = sample.lines
        if len(lines) < 2:
            return -1.0
        json_line = lines[1].strip()
        try:
            json.loads(json_line)
        except Exception:
            return -1.0
        return 0.0


class SummaryContentEqReward(SummaryReward):
    """Checks order-invariant JSON equivalence against the ground-truth summary."""

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0
        pred_json = sample.pred_json()
        if pred_json is None:
            return 0.0
        summary_ref = sample.summary_ref
        if not summary_ref or summary_ref == _IRRELEVANT_TEXT:
            return 0.0
        ref_json = sample.ref_json()
        if ref_json is None:
            return 0.0
        norm_pred, pred_ok = normalize_summary(pred_json, sample.domain_token)
        norm_ref, ref_ok = normalize_summary(ref_json, sample.domain_token)
        if not (pred_ok and ref_ok):
            return 0.0
        return 1.0 if norm_pred == norm_ref else 0.0


class SummaryDatasetReward(SummaryReward):
    """Checks header domain matches the expected token (BBU/RRU)."""

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0
        if sample.domain_token is None:
            return 0.0
        if not sample.strict_json_line:
            return 0.0
        if not sample.lines:
            return 0.0
        header = sample.lines[0].strip()
        expected_prefix = f"<DOMAIN={sample.domain_token}>"
        return 1.0 if header.startswith(expected_prefix) else 0.0


class SummaryNoDupKeysPenalty(SummaryReward):
    """Hard penalty when JSON contains duplicate keys (including nested dicts)."""

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0
        if not sample.strict_json_line:
            return 0.0
        try:
            loads_json_rejecting_duplicate_keys(sample.strict_json_line)
        except JsonDuplicateKeyError:
            return -1.0
        except Exception:
            return 0.0
        return 0.0


def _extract_categories(obj: UnstructuredMapping) -> set[str]:
    obj = require_mapping(obj, context="summary.categories")
    stats = obj.get("统计")
    if not isinstance(stats, list):
        return set()
    cats: set[str] = set()
    for entry in cast(list[object], stats):
        if not isinstance(entry, dict):
            continue
        entry_map = cast(dict[str, object], entry)
        cat = entry_map.get("类别")
        if isinstance(cat, str):
            cat = cat.strip()
            if cat:
                cats.add(cat)
    return cats


def _extract_category_attr_pairs(
    obj: UnstructuredMapping,
    *,
    exclude_text: bool,
) -> set[tuple[str, str]]:
    """Extract (category, attr) pairs from `统计`.

    This intentionally ignores values/counts and focuses on whether a summary
    consistently emits expected attribute *keys* for each category.

    Motivation: Stage-B relies on the presence of certain fields (e.g. 电线/捆扎,
    光纤/套管保护, 光纤/弯曲半径). In practice, models may correctly identify the
    category but omit the attribute key. Key-level recall rewards stabilize the
    JSON schema without requiring perfect count calibration.
    """

    obj = require_mapping(obj, context="summary.category_attr_pairs")
    stats = obj.get("统计")
    if not isinstance(stats, list):
        return set()

    pairs: set[tuple[str, str]] = set()
    for entry in cast(list[object], stats):
        if not isinstance(entry, dict):
            continue
        entry_map = cast(dict[str, object], entry)
        cat_raw = entry_map.get("类别")
        if not isinstance(cat_raw, str):
            continue
        cat = cat_raw.strip()
        if not cat:
            continue

        for attr_raw in entry_map.keys():
            if attr_raw == "类别":
                continue
            attr = str(attr_raw).strip()
            if not attr:
                continue
            if exclude_text and attr == "文本":
                continue
            pairs.add((cat, attr))

    return pairs


def _extract_category_attr_map(
    obj: UnstructuredMapping,
    *,
    exclude_text: bool,
) -> dict[str, set[str]]:
    """Extract per-category attribute key sets from `统计`.

    Returns a mapping: {类别: {属性名, ...}}.

    This is similar to `_extract_category_attr_pairs`, but preserves per-category
    grouping so missing a key on a small category (e.g. 电线/捆扎) isn't drowned
    out by categories with many attributes.
    """

    obj = require_mapping(obj, context="summary.category_attr_map")
    stats = obj.get("统计")
    if not isinstance(stats, list):
        return {}

    by_cat: dict[str, set[str]] = {}
    for entry in cast(list[object], stats):
        if not isinstance(entry, dict):
            continue
        entry_map = cast(dict[str, object], entry)
        cat_raw = entry_map.get("类别")
        if not isinstance(cat_raw, str):
            continue
        cat = cat_raw.strip()
        if not cat:
            continue

        attrs = by_cat.setdefault(cat, set())
        for attr_raw in entry_map.keys():
            if attr_raw == "类别":
                continue
            attr = str(attr_raw).strip()
            if not attr:
                continue
            if exclude_text and attr == "文本":
                continue
            attrs.add(attr)

    return by_cat


_STABLE_ATTR_PATH_PARENTS: frozenset[str] = frozenset(
    {
        # Minimum stable parent keys required by OpenSpec change
        # `2026-01-18-add-summary-attr-path-reward`.
        "捆扎",
        "可见性",
        "符合性",
        "弯曲半径",
        # Expanded based on Stage-A rollout observations:
        # - Cable-related: encourage the model to not emit a category without
        #   also stating key protection/installation attributes used downstream.
        "套管保护",
        "保护",
        "保护措施",
        # - Installation/location semantics are stable and frequently used.
        "位置",
        "安装状态",
        "方向",
        "安装方向",
        # Additional stable keys can be added here as we observe schema drift.
    }
)

_MAX_DYNAMIC_CHILD_KEY_LENGTH = 32


def _is_excluded_attr_path_child_key(key: str) -> bool:
    """Return True if the nested child key is considered dynamic/OCR-like noise."""

    stripped = (key or "").strip()
    if not stripped:
        return True
    # Required by spec: digit-only keys (e.g. "263") are never required.
    if stripped.isdigit():
        return True
    # Spec says SHOULD exclude: key=value-style child keys.
    if "=" in stripped:
        return True
    # Spec says SHOULD exclude: very long keys (OCR-like or concatenated tokens).
    if len(stripped) > _MAX_DYNAMIC_CHILD_KEY_LENGTH:
        return True
    return False


def _extract_category_attr_path_map(
    counts: Mapping[Tuple[str, ...], int],
) -> dict[str, set[str]]:
    """Extract {类别: {父key/子key, ...}} from fact-count keys.

    We intentionally build this on top of `extract_summary_fact_counts` so we
    reuse the existing summary parsing/normalization logic and avoid duplicating
    JSON walking code.
    """

    by_cat: dict[str, set[str]] = {}
    for key in counts.keys():
        # Fact keys for nested attrs look like: ("统计", category, attr, value_str)
        if len(key) != 4 or key[0] != "统计":
            continue
        _, category, parent, child = key

        cat = str(category).strip()
        if not cat:
            continue
        parent_key = str(parent).strip()
        if not parent_key or parent_key == "文本":
            # Required by spec: ignore 文本 entirely (free text / OCR-like).
            continue
        if parent_key not in _STABLE_ATTR_PATH_PARENTS:
            continue
        child_key = str(child).strip()
        if _is_excluded_attr_path_child_key(child_key):
            continue

        by_cat.setdefault(cat, set()).add(f"{parent_key}/{child_key}")

    return by_cat


class SummaryCategoryRecallReward(SummaryReward):
    """Recall-only category coverage: |pred∩ref| / |ref| over `统计[*].类别`."""

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0
        pred_json = sample.pred_json()
        if not isinstance(pred_json, dict):
            return 0.0
        pred_json = cast(UnstructuredMapping, pred_json)
        summary_ref = sample.summary_ref
        if not summary_ref or summary_ref == _IRRELEVANT_TEXT:
            return 0.0
        ref_json = sample.ref_json()
        if not isinstance(ref_json, dict):
            return 0.0
        ref_json = cast(UnstructuredMapping, ref_json)

        _, pred_ok = normalize_summary(pred_json, sample.domain_token)
        _, ref_ok = normalize_summary(ref_json, sample.domain_token)
        if not (pred_ok and ref_ok):
            return 0.0

        pred_cats = _extract_categories(pred_json)
        ref_cats = _extract_categories(ref_json)
        if not ref_cats:
            return 1.0 if not pred_cats else 0.0
        return float(len(pred_cats & ref_cats) / len(ref_cats))


class SummaryAttrKeyRecallReward(SummaryReward):
    """Recall-only attribute key coverage over (类别, 属性名) pairs.

    This ignores counts/values (handled by structured content rewards) and
    focuses on schema stability: does the model output the expected attribute
    keys for each category that appears in the reference summary?
    """

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0

        pred_json = sample.pred_json()
        if not isinstance(pred_json, dict):
            return 0.0
        pred_json = cast(UnstructuredMapping, pred_json)

        summary_ref = sample.summary_ref
        if not summary_ref or summary_ref == _IRRELEVANT_TEXT:
            return 0.0
        ref_json = sample.ref_json()
        if not isinstance(ref_json, dict):
            return 0.0
        ref_json = cast(UnstructuredMapping, ref_json)

        _, pred_ok = normalize_summary(pred_json, sample.domain_token)
        _, ref_ok = normalize_summary(ref_json, sample.domain_token)
        if not (pred_ok and ref_ok):
            return 0.0

        # Per-category averaging makes this reward sensitive to dropping a key on
        # a small-but-important category (e.g. 电线/捆扎) instead of being diluted
        # by categories that carry many attributes.
        pred_map = _extract_category_attr_map(pred_json, exclude_text=True)
        ref_map = _extract_category_attr_map(ref_json, exclude_text=True)
        if not ref_map:
            return 1.0 if not pred_map else 0.0

        scores: list[float] = []
        for cat, ref_attrs in ref_map.items():
            if not ref_attrs:
                continue
            pred_attrs = pred_map.get(cat, set())
            scores.append(len(pred_attrs & ref_attrs) / len(ref_attrs))
        if not scores:
            # Edge case: reference only had excluded attrs (e.g. 文本-only categories).
            return 1.0 if not pred_map else 0.0
        return float(sum(scores) / len(scores))


class SummaryAttrPathRecallReward(SummaryReward):
    """Recall-only nested attribute path coverage (depth=2) for stable parent keys.

    This is a schema-completeness signal: for categories the model *chooses* to
    emit (and that exist in the reference), does it also emit the expected
    nested attribute paths like `捆扎/整齐`, `可见性/完整`, `符合性/符合`?

    Scoring categories are `pred_categories ∩ ref_categories` (omitting a
    category is not directly penalized by this reward).
    """

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0

        pred_json = sample.pred_json()
        if not isinstance(pred_json, dict):
            return 0.0
        pred_json = cast(UnstructuredMapping, pred_json)

        summary_ref = sample.summary_ref
        if not summary_ref or summary_ref == _IRRELEVANT_TEXT:
            return 0.0

        ref_json = sample.ref_json()
        if not isinstance(ref_json, dict):
            return 0.0
        ref_json = cast(UnstructuredMapping, ref_json)

        _, pred_ok = normalize_summary(pred_json, sample.domain_token)
        _, ref_ok = normalize_summary(ref_json, sample.domain_token)
        if not (pred_ok and ref_ok):
            return 0.0

        pred_cats = _extract_categories(pred_json)
        ref_cats = _extract_categories(ref_json)
        scoring_cats = pred_cats & ref_cats
        if not scoring_cats:
            # Neutral: this reward is only about completeness for emitted
            # categories that also exist in the reference.
            return 0.0

        pred_counts = extract_summary_fact_counts(pred_json, sample.domain_token)
        ref_counts = extract_summary_fact_counts(ref_json, sample.domain_token)
        if pred_counts is None or ref_counts is None:
            return 0.0

        pred_paths = _extract_category_attr_path_map(pred_counts)
        ref_paths = _extract_category_attr_path_map(ref_counts)

        scores: list[float] = []
        for cat in scoring_cats:
            required = ref_paths.get(cat, set())
            if not required:
                # If the reference doesn't contain any stable nested paths for this
                # category, this reward should be neutral for it.
                continue
            predicted = pred_paths.get(cat, set())
            scores.append(len(predicted & required) / len(required))

        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))


class SummaryStructuredContentTverskyReward(SummaryReward):
    """Recall-biased structured fact reward (Tversky index on fact counts)."""

    _BBU_ALPHA = 0.30
    _RRU_ALPHA = 0.60
    _BETA = 1.00

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0
        pred_json = sample.pred_json()
        if pred_json is None:
            return 0.0
        summary_ref = sample.summary_ref
        if not summary_ref or summary_ref == _IRRELEVANT_TEXT:
            return 0.0
        ref_json = sample.ref_json()
        if ref_json is None:
            return 0.0

        pred_counts = extract_summary_fact_counts(pred_json, sample.domain_token)
        ref_counts = extract_summary_fact_counts(ref_json, sample.domain_token)
        if pred_counts is None or ref_counts is None:
            return 0.0

        if sample.domain_token == "BBU":
            pred_filtered = filter_fact_counts(
                pred_counts, exclude_notes=True, exclude_text=True
            )
            ref_filtered = filter_fact_counts(
                ref_counts, exclude_notes=True, exclude_text=True
            )
            alpha = self._BBU_ALPHA
        else:
            pred_filtered = filter_fact_counts(
                pred_counts, exclude_notes=True, exclude_text=False
            )
            ref_filtered = filter_fact_counts(
                ref_counts, exclude_notes=True, exclude_text=False
            )
            alpha = self._RRU_ALPHA

        return float(
            tversky_from_fact_counts(
                pred_filtered, ref_filtered, alpha=alpha, beta=self._BETA
            )
        )


class SummaryTextBBUReward(SummaryReward):
    """BBU-only OCR `文本` reward: lower-bound recall with +2 unique overflow guard."""

    _OVERFLOW_PENALTY_PER_EXTRA = 0.10
    _OVERFLOW_FREE_EXTRA = 2

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0
        if sample.domain_token != "BBU":
            return 0.0
        pred_json = sample.pred_json()
        if pred_json is None:
            return 0.0
        summary_ref = sample.summary_ref
        if not summary_ref or summary_ref == _IRRELEVANT_TEXT:
            return 0.0
        ref_json = sample.ref_json()
        if ref_json is None:
            return 0.0

        pred_counts = extract_summary_fact_counts(pred_json, "BBU")
        ref_counts = extract_summary_fact_counts(ref_json, "BBU")
        if pred_counts is None or ref_counts is None:
            return 0.0

        def _to_text_counts(counts: Mapping[Tuple[str, ...], int]) -> dict[str, int]:
            out: dict[str, int] = {}
            for key, value in counts.items():
                if not value:
                    continue
                if len(key) < 4 or key[0] != "统计" or key[2] != "文本":
                    continue
                text = normalize_bbu_ocr_text(str(key[3]))
                if not text:
                    continue
                out[text] = out.get(text, 0) + int(value)
            return out

        pred_text = _to_text_counts(pred_counts)
        ref_text = _to_text_counts(ref_counts)

        pred_unique = len(pred_text)
        ref_unique = len(ref_text)
        ref_total = sum(ref_text.values())

        overlap = 0
        for key in pred_text.keys() & ref_text.keys():
            overlap += min(pred_text[key], ref_text[key])

        if ref_total > 0:
            score = float(overlap) / float(ref_total)
        else:
            score = 0.0

        overflow_allow = ref_unique + self._OVERFLOW_FREE_EXTRA
        extra_unique = max(0, pred_unique - overflow_allow)
        if extra_unique > 0:
            score -= self._OVERFLOW_PENALTY_PER_EXTRA * float(extra_unique)

        return float(max(-1.0, min(1.0, score)))


class SummaryNotesBBUReward(SummaryReward):
    """BBU-only notes reward: recall when GT has notes; penalize spurious notes otherwise."""

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0
        if sample.domain_token != "BBU":
            return 0.0
        pred_json = sample.pred_json()
        if not isinstance(pred_json, dict):
            return 0.0
        summary_ref = sample.summary_ref
        if not summary_ref or summary_ref == _IRRELEVANT_TEXT:
            return 0.0
        ref_json = sample.ref_json()
        if not isinstance(ref_json, dict):
            return 0.0

        def _notes(obj: UnstructuredMapping) -> set[str]:
            obj = require_mapping(obj, context="summary.notes")
            notes = obj.get("备注")
            if not isinstance(notes, list):
                return set()
            out: set[str] = set()
            for note in cast(list[object], notes):
                if note is None:
                    continue
                norm = normalize_bbu_ocr_text(str(note))
                if norm:
                    out.add(norm)
            return out

        ref_notes = _notes(cast(UnstructuredMapping, ref_json))
        pred_notes = _notes(cast(UnstructuredMapping, pred_json))

        if not ref_notes:
            return -1.0 if pred_notes else 0.0

        overlap = len(pred_notes & ref_notes)
        return float(overlap / len(ref_notes))


class SummaryContentF1Reward(SummaryReward):
    """Partial-match content reward based on count-weighted fact overlap."""

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0
        pred_json = sample.pred_json()
        if pred_json is None:
            return 0.0
        summary_ref = sample.summary_ref
        if not summary_ref or summary_ref == _IRRELEVANT_TEXT:
            return 0.0
        ref_json = sample.ref_json()
        if ref_json is None:
            return 0.0

        pred_counts = extract_summary_fact_counts(pred_json, sample.domain_token)
        ref_counts = extract_summary_fact_counts(ref_json, sample.domain_token)
        if pred_counts is None or ref_counts is None:
            return 0.0

        return float(f1_from_fact_counts(pred_counts, ref_counts))


class SummaryCategoryF1Reward(SummaryReward):
    """Dense category coverage reward (set-F1 over `统计[*].类别`)."""

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0
        pred_json = sample.pred_json()
        if not isinstance(pred_json, dict):
            return 0.0
        pred_json = cast(UnstructuredMapping, pred_json)
        summary_ref = sample.summary_ref
        if not summary_ref or summary_ref == _IRRELEVANT_TEXT:
            return 0.0
        ref_json = sample.ref_json()
        if not isinstance(ref_json, dict):
            return 0.0
        ref_json = cast(UnstructuredMapping, ref_json)

        _, pred_ok = normalize_summary(pred_json, sample.domain_token)
        _, ref_ok = normalize_summary(ref_json, sample.domain_token)
        if not (pred_ok and ref_ok):
            return 0.0

        pred_cats = _extract_categories(pred_json)
        ref_cats = _extract_categories(ref_json)
        return float(f1_from_sets(pred_cats, ref_cats))


class SummaryNotesPresenceReward(SummaryReward):
    """BBU-only: reward if ref has non-empty `备注` and pred also includes non-empty `备注`."""

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0
        if sample.domain_token != "BBU":
            return 0.0
        summary_ref = sample.summary_ref
        if not summary_ref or summary_ref == _IRRELEVANT_TEXT:
            return 0.0
        ref_json = sample.ref_json()
        if not isinstance(ref_json, dict):
            return 0.0
        ref_json = cast(dict[str, object], ref_json)
        ref_notes = ref_json.get("备注")
        ref_has_notes = isinstance(ref_notes, list) and any(
            normalize_free_text(str(v))
            for v in cast(list[object], ref_notes)
            if v is not None
        )
        if not ref_has_notes:
            return 0.0

        pred_json = sample.pred_json()
        if not isinstance(pred_json, dict):
            return 0.0
        pred_json = cast(dict[str, object], pred_json)
        pred_notes = pred_json.get("备注")
        pred_has_notes = isinstance(pred_notes, list) and any(
            normalize_free_text(str(v))
            for v in cast(list[object], pred_notes)
            if v is not None
        )
        return 1.0 if pred_has_notes else 0.0


class SummaryGroupStatsPresenceReward(SummaryReward):
    """RRU-only: reward if ref has non-empty `分组统计` and pred also includes it."""

    def score(self, sample: SummarySample) -> float:
        if sample.is_irrelevant:
            return 0.0
        if sample.domain_token != "RRU":
            return 0.0
        summary_ref = sample.summary_ref
        if not summary_ref or summary_ref == _IRRELEVANT_TEXT:
            return 0.0
        ref_json = sample.ref_json()
        if not isinstance(ref_json, dict):
            return 0.0
        ref_json = cast(dict[str, object], ref_json)
        ref_group = ref_json.get("分组统计")
        if not isinstance(ref_group, dict):
            return 0.0
        ref_group_map = cast(dict[str, object], ref_group)
        ref_has_groups = any(str(k).strip() for k in ref_group_map.keys())
        if not ref_has_groups:
            return 0.0

        pred_json = sample.pred_json()
        if not isinstance(pred_json, dict):
            return 0.0
        pred_json = cast(dict[str, object], pred_json)
        pred_group = pred_json.get("分组统计")
        if not isinstance(pred_group, dict):
            return 0.0
        pred_group_map = cast(dict[str, object], pred_group)
        pred_has_groups = any(str(k).strip() for k in pred_group_map.keys())
        return 1.0 if pred_has_groups else 0.0


__all__ = [
    "SummaryAttrKeyRecallReward",
    "SummaryCategoryF1Reward",
    "SummaryCategoryRecallReward",
    "SummaryContentEqReward",
    "SummaryContentF1Reward",
    "SummaryDatasetReward",
    "SummaryFormatReward",
    "SummaryGroupStatsPresenceReward",
    "SummaryHeaderReward",
    "SummaryNoDupKeysPenalty",
    "SummaryNotesBBUReward",
    "SummaryNotesPresenceReward",
    "SummaryParsePenalty",
    "SummaryStrictPenaltyReward",
    "SummaryStructuredContentTverskyReward",
    "SummaryTextBBUReward",
]
