"""Summary-mode label normalization helpers."""

from __future__ import annotations

import re
from typing import Any, Optional, cast

from .base import BasePreprocessor
from ..contracts import ConversationRecord


class SummaryLabelNormalizer(BasePreprocessor):
    """Collapse identifiable 标签/* entries into a single 标签/可以识别 bucket.

    The summary field typically contains segments such as
    ``标签/信号线标签×1`` or ``标签/无法识别×3`` separated by ``，``.
    When enabled, all label entries whose descriptor does *not* contain
    ``无法识别`` are merged into ``标签/可以识别×N`` while preserving the
    total counts for both identifiable and unidentifiable labels.
    """

    def __init__(
        self,
        *,
        label_prefix: str = "标签",
        unknown_token: str = "无法识别",
        merged_token: str = "可以识别",
        delimiter: str = "，",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.label_prefix = label_prefix
        self.unknown_token = unknown_token
        self.merged_token = merged_token
        self.delimiter = delimiter or "，"
        # Allow either the × symbol or ascii x/X between descriptor and count.
        self._pattern = re.compile(
            rf"^{re.escape(self.label_prefix)}/(.+)[×xX]\s*(\d+)$"
        )

    def preprocess(self, row: ConversationRecord) -> Optional[ConversationRecord]:
        summary = row.get("summary")
        if not isinstance(summary, str):
            return row

        # Split on the primary delimiter; fall back to comma if needed.
        segments = [seg.strip() for seg in summary.split(self.delimiter) if seg.strip()]
        if len(segments) == 1 and "," in summary and self.delimiter != ",":
            segments = [seg.strip() for seg in summary.split(",") if seg.strip()]

        kept: list[str] = []
        first_label_idx: Optional[int] = None
        identifiable = 0
        unknown = 0

        for seg in segments:
            match = self._pattern.match(seg)
            if not match:
                kept.append(seg)
                continue

            desc = match.group(1).strip()
            try:
                count = int(match.group(2))
            except (TypeError, ValueError):
                kept.append(seg)
                continue

            if first_label_idx is None:
                first_label_idx = len(kept)

            if self.unknown_token in desc:
                unknown += count
            else:
                identifiable += count

        # No label entries detected; keep summary unchanged.
        if first_label_idx is None:
            return row

        label_segments: list[str] = []
        if identifiable > 0:
            label_segments.append(
                f"{self.label_prefix}/{self.merged_token}×{identifiable}"
            )
        if unknown > 0:
            label_segments.append(
                f"{self.label_prefix}/{self.unknown_token}×{unknown}"
            )

        new_segments = (
            kept + label_segments
            if first_label_idx is None
            else kept[:first_label_idx] + label_segments + kept[first_label_idx:]
        )

        row_copy = dict(row)
        row_copy["summary"] = self.delimiter.join(new_segments)
        return cast(ConversationRecord, cast(object, row_copy))


__all__ = ["SummaryLabelNormalizer"]
