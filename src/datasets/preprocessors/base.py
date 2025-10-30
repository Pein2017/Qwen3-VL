"""Base preprocessor following ms-swift patterns"""

from typing import Any, Mapping, Optional

from ..contracts import ConversationRecord, validate_conversation_record


class BasePreprocessor:
    """Base preprocessor for row-level transformations.

    Follows ms-swift RowPreprocessor pattern for pluggable, composable preprocessing.
    """

    def __init__(self, **kwargs: Any):
        """Initialize preprocessor with optional configuration."""
        self.config = kwargs

    def preprocess(self, row: ConversationRecord) -> Optional[ConversationRecord]:
        """Preprocess a single row."""
        return row

    def __call__(self, row: Mapping[str, Any]) -> Optional[ConversationRecord]:
        """Allow preprocessor to be called as a function."""
        validated = validate_conversation_record(row)
        return self.preprocess(validated)


__all__ = ["BasePreprocessor"]
