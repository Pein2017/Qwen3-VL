"""Base builder interface"""

from typing import Any, Dict, Iterable, Mapping

from ..contracts import ConversationRecord, validate_conversation_record


class BaseBuilder:
    """Base class for message builders.

    Builders construct messages from paired records for training.
    """

    def __init__(self, **kwargs: Any):
        """Initialize builder with configuration."""
        self.config = kwargs

    def build(
        self,
        record_a: ConversationRecord,
        record_b: ConversationRecord,
    ) -> Dict[str, Any]:
        """Build messages from two records."""
        raise NotImplementedError("Subclasses must implement build()")

    # Optional: for N-way grouping
    def build_many(self, records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        items = [validate_conversation_record(r) for r in records]
        if not items:
            raise ValueError("build_many requires at least one record")
        if len(items) == 1:
            return self.build(items[0], items[0])
        return self.build(items[0], items[1])

    def __call__(
        self,
        record_a: Mapping[str, Any],
        record_b: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Allow builder to be called as a function."""
        rec_a = validate_conversation_record(record_a)
        rec_b = validate_conversation_record(record_b)
        return self.build(rec_a, rec_b)


__all__ = ["BaseBuilder"]
