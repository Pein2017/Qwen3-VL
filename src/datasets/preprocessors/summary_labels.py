from __future__ import annotations

from typing import Optional

from ..contracts import ConversationRecord
from .base import BasePreprocessor


class SummaryLabelNormalizer(BasePreprocessor):
    def preprocess(self, row: ConversationRecord) -> Optional[ConversationRecord]:
        return row


__all__ = ["SummaryLabelNormalizer"]
